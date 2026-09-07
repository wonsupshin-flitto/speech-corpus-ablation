#!/usr/bin/env python3
"""E3 신규성 판정 — 유효신규율 (논문1-1).

★ 신규성만 재면 안 된다
------------------------
정제하지 않은 exp0 에는 전사 오류·환각·잡담이 섞여 있고, **이것들은 어떤 참조에도
없으므로 100% "신규"로 집계된다.** 신규성만 재면 exp0 가 exp2 보다 높게 나와
ablation 주장이 거꾸로 뒤집힌다.

그래서 세 축을 함께 판정하고 교차한다.

    유효신규율 = (공개 지식에 없음) ∧ (전사문에 근거함) ∧ (사실로 옳음)

기대되는 그림: exp0 는 신규성 총량이 높지만 근거·정확성이 낮아 유효신규율이 낮고,
exp2 는 반대. **"정제가 가짜 신규성을 걷어낸다"** 는 서사가 되고 ablation 과 맞는다.

판정 축
-------
    novel      프론티어 폐쇄형 답변이 gold 의 핵심을 못 맞혔는가
               (맞혔으면 공개 지식 → not novel)
    grounded   gold 가 제공된 전사문 근거로 뒷받침되는가
    correct    gold 가 사실로 옳은가 (자기모순·비상식 없음)

cell-blind
----------
judge 입력에 조건·도메인 식별자를 넣지 않는다. jid 는 순번뿐이고 매핑은 별도 파일.
`.claude/rules/judge_model_policy.md` 참조.

사용:
    # 1) 배치 생성
    python scripts/rag/judge_novelty.py build \\
      --closedbook <RUN>/closedbook/exp2_medical.jsonl --cond exp2 --domain medical \\
      --out <RUN>/judge_inputs
    # 2) 채점
    python scripts/rag/judge_novelty.py run \\
      --batch <RUN>/judge_inputs/novelty_batch.json \\
      --model openai/gpt-5.6-sol --out <RUN>/scores/nov_gpt.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request

FORBIDDEN = {"model", "provider", "cond", "condition", "experiment", "domain",
             "cell", "group", "source", "origin", "treatment"}

SYS = (
    "You grade Korean expert-domain QA items on three independent axes. "
    "For each item you get: `question`, `gold` (the answer we produced), and "
    "`closedbook` (what a frontier model answered from its own knowledge, no documents).\n"
    "Grade each axis independently:\n"
    "  novel:    true if `closedbook` FAILS to convey the core content of `gold` "
    "(different entity/value/concept, or vague/evasive). false if it conveys the same "
    "core content (paraphrase is fine). Do NOT reward longer answers.\n"
    "  correct:  true if `gold` is factually sound and internally consistent for the "
    "question. false if self-contradictory, nonsensical, or clearly wrong.\n"
    "  grounded: true if `gold` reads as a substantive domain answer to `question`. "
    "false if it is filler, transcription noise, off-topic chatter, or does not answer.\n"
    "Output ONLY a JSON array "
    "[{\"jid\":\"..\",\"novel\":true|false,\"correct\":true|false,\"grounded\":true|false}], "
    "no prose.")
CHUNK = 20
MAX_TOKENS = 8000


def load_key(env_path: str) -> str:
    k = os.environ.get("DW_OPENROUTER_API_KEY")
    if k:
        return k
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            if line.startswith("DW_OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("DW_OPENROUTER_API_KEY 없음")


def cmd_build(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    bf = os.path.join(args.out, "novelty_batch.json")
    mf = os.path.join(args.out, "novelty_map.json")
    batch = json.load(open(bf, encoding="utf-8")) if os.path.exists(bf) else []
    cmap = json.load(open(mf, encoding="utf-8")) if os.path.exists(mf) else {}
    i = len(batch)

    for line in open(args.closedbook, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        i += 1
        jid = f"n{i:06d}"                       # 순번만. 출처 인코딩 금지
        batch.append({"jid": jid, "question": r["question"],
                      "gold": r["gold"], "closedbook": r.get("closedbook", "")})
        cmap[jid] = {"cond": args.cond, "domain": args.domain, "idx": r["idx"]}

    json.dump(batch, open(bf, "w"), ensure_ascii=False, indent=1)
    json.dump(cmap, open(mf, "w"), ensure_ascii=False, indent=1)

    leak = sorted({k for it in batch for k in it if k in FORBIDDEN})
    bad = [it["jid"] for it in batch if not it["jid"][1:].isdigit()][:3]
    print(f"누적 {len(batch)}건 | cell-blind: {'LEAK ' + str(leak) if leak else 'OK'}"
          f"{' | jid 이상 ' + str(bad) if bad else ''}")
    if leak or bad:
        raise SystemExit("cell-blind 위반. 중단.")


def cmd_run(args) -> None:
    key = load_key(args.env)
    batch = json.load(open(args.batch, encoding="utf-8"))
    done = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    todo = [it for it in batch if it["jid"] not in done]
    print(f"전체 {len(batch)} / 완료 {len(done)} / 남은 {len(todo)} | {args.model}")

    def call(items):
        body = json.dumps({
            "model": args.model, "temperature": 0, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": "Grade these items:\n"
                          + json.dumps(items, ensure_ascii=False)}],
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=body,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        txt = d["choices"][0]["message"]["content"]
        m = re.search(r"\[.*\]", txt, re.S)      # 코드펜스 감싸는 모델 대응
        if not m:
            raise ValueError(f"JSON 배열 없음: {txt[:110]}")
        return json.loads(m.group(0)), d.get("usage", {})

    cost = 0.0
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        for attempt in range(4):
            try:
                vs, usage = call(chunk)
                for v in vs:
                    done[v["jid"]] = {k: v.get(k) for k in ("novel", "correct", "grounded")}
                cost += float(usage.get("cost") or 0)
                break
            except Exception as e:
                print(f"  retry {i} {attempt}: {str(e)[:100]}")
                time.sleep(5 * (attempt + 1))
        else:
            print(f"  ❌ 배치 {i} 실패, 건너뜀")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(done, open(args.out, "w"), ensure_ascii=False)
        n = i + len(chunk)
        est = cost / n * len(todo) if n else 0
        print(f"  {len(done)}/{len(batch)}  누적 ${cost:.3f}  예상 ${est:.2f}")
    print(f"완료 {args.out} | ${cost:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--closedbook", required=True)
    b.add_argument("--cond", required=True)
    b.add_argument("--domain", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run")
    r.add_argument("--batch", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--env", default=".env")
    r.set_defaults(fn=cmd_run)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
