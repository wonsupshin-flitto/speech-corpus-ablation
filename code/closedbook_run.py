#!/usr/bin/env python3
"""폐쇄형 응시 — 지정 모델이 문서 없이 답한다 (논문1-1 E3 교차 설계).

배경
----
E3 는 `claude-sonnet-5` 하나만 응시시켰고, 그 답을 `gpt-5.6-sol` 이 채점했다.
그래서 신규성이 "한 모델이 모르는 것" 에 머문다. 두 모델이 각각 응시하고 서로를
채점하면 대칭이 맞고, "두 모델 다 모르는 것" 이라는 더 단단한 하한이 나온다.

이 스크립트는 그 추가분, 곧 `gpt-5.6-sol` 의 응시를 담당한다. 채점은
`build_novelty_judge.py` + 서브에이전트(`claude-sonnet-5`) 가 맡는다.

입력은 기존 배치를 그대로 쓴다. 질문만 들어 있고 조건·도메인 식별자가 없다
(cell-blind, `.claude/rules/judge_model_policy.md`).

지시문은 `report/260827_novelty_closedbook/INSTRUCTIONS.md` 의 서브에이전트용
문안을 API 용으로 옮긴 것이다. 문구를 바꾸면 두 응시자의 조건이 달라지므로
**수정 금지**.

사용:
    python scripts/rag/closedbook_run.py \
      --batch-dir paper_works/1-1.speech_corpus_rag/report/260827_novelty_closedbook/batches \
      --out-dir   paper_works/1-1.speech_corpus_rag/report/260906_novelty_cross/closedbook_sol \
      --model openai/gpt-5.6-sol
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ── INSTRUCTIONS.md 와 같은 조건. 수정 금지 ──────────────────────────────
SYS = (
    "당신은 폐쇄형(closed-book) 질의응답을 수행합니다. "
    "외부 정보를 조회할 수 없으며, 당신이 학습한 지식만으로 답해야 합니다.\n\n"
    "각 질문에 한국어로 3~5문장, 400자 내외의 답변을 씁니다.\n"
    "모르면 모른다고 쓰십시오. 이 측정의 핵심은 '최신 AI 가 이미 아는 지식인가' "
    "입니다. 모르는 것을 아는 척 지어내면 측정이 망가집니다. "
    "모를 때는 '제공된 정보만으로는 알 수 없습니다' 로 시작해 아는 범위까지만 씁니다.\n"
    "질문에 특정 강연·회사·제품·코스 이름이 나오는데 실제로 모른다면 추측하지 말고 "
    "모른다고 쓰십시오. 일반론으로 얼버무리는 것이 가장 나쁜 실패입니다.\n"
    "아는 질문이면 아는 만큼 구체적으로 씁니다.\n\n"
    'JSON 배열만 출력합니다: [{"qid":"..","closedbook":".."}]. 다른 말은 쓰지 마십시오.'
)
CHUNK = 10          # 답변이 길어 배치보다 작게 쪼갠다
TEMPERATURE = 0
MAX_TOKENS = 8000
# ────────────────────────────────────────────────────────────────────────


def load_key(env_path: str) -> str:
    k = os.environ.get("DW_OPENROUTER_API_KEY")
    if k:
        return k
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            if line.startswith("DW_OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        f"DW_OPENROUTER_API_KEY 를 찾을 수 없다 ({env_path}).\n"
        "  OPENAI_API_KEY 는 OpenAI 직결 키라 OpenRouter 에서 401 이다."
    )


def price(model: str, pt: int, ct: int) -> float:
    """OpenRouter 공개 단가 (USD per 1M)."""
    table = {"openai/gpt-5.6-sol": (1.75, 14.0)}
    i, o = table.get(model, (2.0, 10.0))
    return pt / 1e6 * i + ct / 1e6 * o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--env", default=".env")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="배치 수 제한 (0=전체)")
    args = ap.parse_args()

    key = load_key(args.env)
    os.makedirs(args.out_dir, exist_ok=True)
    batches = sorted(glob.glob(os.path.join(args.batch_dir, "batch_*.json")))
    if args.limit:
        batches = batches[: args.limit]

    # 재개: 이미 답한 qid 는 다시 묻지 않는다.
    done: dict[str, str] = {}
    for f in glob.glob(os.path.join(args.out_dir, "cb_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                d = json.loads(line)
                done[d["qid"]] = d["closedbook"]

    items: list[dict] = []
    for b in batches:
        for it in json.load(open(b, encoding="utf-8")):
            if it["qid"] not in done:
                items.append(it)

    print(f"배치 {len(batches)} | 완료 {len(done)} | 남은 {len(items)} | "
          f"호출 {(len(items) + CHUNK - 1) // CHUNK}회 | {args.model}")
    if not items:
        return

    def call(chunk):
        body = json.dumps({
            "model": args.model, "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": json.dumps(chunk, ensure_ascii=False)},
            ],
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=body,
            headers={"Authorization": "Bearer " + key,
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        txt = d["choices"][0]["message"]["content"]
        m = re.search(r"\[.*\]", txt, re.S)
        if not m:
            raise ValueError(f"JSON 배열 없음: {txt[:150]}")
        return json.loads(m.group(0)), d.get("usage", {})

    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    lock = threading.Lock()
    out_path = os.path.join(args.out_dir, "cb_sol.jsonl")
    state = {"n": 0, "cost": 0.0}

    def work(idx_chunk):
        idx, chunk = idx_chunk
        try:
            got, usage = call(chunk)
        except Exception as e:  # noqa: BLE001
            print(f"  chunk {idx}: {type(e).__name__} {e}"[:160])
            return
        c = price(args.model, usage.get("prompt_tokens", 0),
                  usage.get("completion_tokens", 0))
        with lock:
            # 증분 저장. 중단되어도 여기까지는 남는다.
            with open(out_path, "a", encoding="utf-8") as f:
                for g in got:
                    if g.get("qid") and g.get("closedbook"):
                        f.write(json.dumps(
                            {"qid": g["qid"], "closedbook": g["closedbook"]},
                            ensure_ascii=False) + "\n")
            state["n"] += len(got)
            state["cost"] += c
            if idx % 20 == 0 or idx == len(chunks) - 1:
                print(f"  [{idx + 1}/{len(chunks)}] 누적 {state['n']}건 "
                      f"${state['cost']:.2f}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, enumerate(chunks)))

    print(f"\n완료 {state['n']}건 | ${state['cost']:.2f} | {out_path}")


if __name__ == "__main__":
    main()
