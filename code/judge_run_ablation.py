#!/usr/bin/env python3
"""cell-blind judge 실행 — OpenRouter cross-provider (논문1-1 E1/E2).

원본 `judge_run.py` 를 고친 것. 바뀐 것은 두 가지뿐이다.

1. **키 경로**: `~/.openrouter_env` → 레포 `.env` 의 `DW_OPENROUTER_API_KEY`.
   원본 경로가 로컬·서버 어디에도 없어 그대로는 못 돈다(260826 확인).
   ⚠️ 같은 `.env` 의 `OPENAI_API_KEY` 는 OpenAI 직결 키라 OpenRouter 에서 401 이다.
2. **경로 인자화**: 서버 경로 하드코딩 제거.

프롬프트·온도·배치 크기는 원본 그대로 둔다(기존 실험 A 수치와의 비교 가능성).

judge 모델 (260826 확정)
    openai/gpt-5.6-sol
    anthropic/claude-sonnet-5
둘 다 실호출 검증됨. `openai/gpt-5.6-codex` 는 존재하지 않는 슬러그다.

사용:
    python scripts/rag/judge_run_ablation.py \\
      --batch <judge입력>/correctness_batch.json \\
      --model openai/gpt-5.6-sol \\
      --out <출력>/corr_gpt.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ── 원본 그대로. 수정 금지 ────────────────────────────────────────────────
CORR_SYS = (
    "You are a strict reference-based grader for a Korean expert-QA benchmark. "
    "Do NOT reward longer answers; judge only factual agreement with the reference gold. "
    "'correct' if the answer's core content matches gold (same entity/value/concept; paraphrase fine), "
    "'partial' if right area but misses the key point / ambiguous, "
    "'incorrect' if wrong/unrelated/empty. "
    "Output ONLY a JSON array [{\"jid\":\"..\",\"verdict\":\"correct|partial|incorrect\"}], no prose.")
CHUNK = 25
TEMPERATURE = 0
MAX_TOKENS = 6000
# ────────────────────────────────────────────────────────────────────────


def load_key(env_path: str) -> str:
    """레포 .env 에서 OpenRouter 키를 읽는다."""
    k = os.environ.get("DW_OPENROUTER_API_KEY")
    if k:
        return k
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            if line.startswith("DW_OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        f"DW_OPENROUTER_API_KEY 를 찾을 수 없다 ({env_path}).\n"
        "  ⚠️ OPENAI_API_KEY 는 OpenAI 직결 키라 OpenRouter 에서 401 이다.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default=".env")
    # 호출끼리 독립이라 병렬화해도 판정이 달라지지 않는다.
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    key = load_key(args.env)
    batch = json.load(open(args.batch, encoding="utf-8"))
    # 재개: 이미 채점된 항목은 다시 묻지 않는다.
    verdicts_done = (json.load(open(args.out, encoding="utf-8"))
                     if os.path.exists(args.out) else {})
    todo = [it for it in batch if it["jid"] not in verdicts_done]
    print(f"전체 {len(batch)} | 완료 {len(verdicts_done)} | 남은 {len(todo)} | "
          f"호출 {(len(todo)+CHUNK-1)//CHUNK}회 | {args.model}")

    def call(items):
        body = json.dumps({
            "model": args.model, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": CORR_SYS},
                         {"role": "user",
                          "content": "Grade these items:\n" + json.dumps(items, ensure_ascii=False)}],
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        txt = d["choices"][0]["message"]["content"]
        # 코드펜스로 감싸는 모델이 있다(sonnet-5 실측). 배열만 뽑아낸다.
        m = re.search(r"\[.*\]", txt, re.S)
        if not m:
            raise ValueError(f"JSON 배열 없음: {txt[:120]}")
        return json.loads(m.group(0)), d.get("usage", {})

    verdicts, cost = {}, 0.0
    verdicts.update(verdicts_done)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    lock = threading.Lock()
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    done_n = [0]

    def work(idx_chunk):
        idx, chunk = idx_chunk
        nonlocal cost
        for attempt in range(4):
            try:
                vs, usage = call(chunk)
                with lock:
                    for v in vs:
                        verdicts[v["jid"]] = v["verdict"]
                    cost += float(usage.get("cost") or 0)
                    done_n[0] += 1
                    # 저장은 락 안에서. 부분 파일이 깨지면 재개가 무너진다.
                    json.dump(verdicts, open(args.out, "w"), ensure_ascii=False)
                    est = cost / done_n[0] * len(chunks) if done_n[0] else 0
                    print(f"  [{done_n[0]}/{len(chunks)}] {len(verdicts)}/{len(batch)}  "
                          f"누적 ${cost:.4f}  예상 ${est:.2f}", flush=True)
                return
            except Exception as e:
                print(f"  retry {idx} {attempt}: {str(e)[:110]}", flush=True)
                time.sleep(5 * (attempt + 1))
        print(f"  ❌ 배치 {idx} 4회 실패, 건너뜀", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, enumerate(chunks)))

    json.dump(verdicts, open(args.out, "w"), ensure_ascii=False, indent=1)
    miss = len(batch) - len(verdicts)
    print(f"완료 {args.out} | 판정 {len(verdicts)}건"
          f"{f' | 누락 {miss}건' if miss else ''} | 비용 ${cost:.4f}")


if __name__ == "__main__":
    main()
