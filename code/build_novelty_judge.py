#!/usr/bin/env python3
"""E3 신규성 채점 배치 생성 — 중복 제거 구조를 원래 조건별로 되돌려 붙인다.

왜 별도 단계가 필요한가
-----------------------
폐쇄형 답변은 **고유 질문 단위**로 받았다(같은 질문을 여러 조건이 공유하므로).
그런데 채점은 **조건별로** 집계해야 한다. 그래서 여기서 다시 펼친다.

    cb_*.jsonl        qid -> closedbook          (고유 질문 6886건)
    usage_map.json    qid -> [{cond, domain, idx, gold}, ...]
    question_map.json qid -> question
        ↓
    novelty_batch.json  (조건별로 펼친 10695건)

cell-blind
----------
채점 입력에는 `question` / `gold` / `closedbook` 만 넣는다. 조건·도메인·모델은
넣지 않고 `novelty_map.json` 에만 둔다. jid 는 순번뿐이다.
`.claude/rules/judge_model_policy.md` 참조.

⚠️ 순번 id 는 이 실행 안에서만 유효하다. 다른 실행의 채점 결과와 섞지 말 것
(260827 사고: 실행이 다른 매핑·채점을 순번 키로 조인해 라벨이 밀렸다).

사용:
    python scripts/rag/build_novelty_judge.py --run <RUN> --out <RUN>/judge_inputs
"""
from __future__ import annotations

import argparse
import glob
import json
import os

FORBIDDEN = {"model", "provider", "generator", "cond", "condition", "experiment",
             "exp_id", "domain", "cell", "cell_id", "group", "source", "origin",
             "treatment", "judge_target"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    usage = json.load(open(f"{args.run}/usage_map.json", encoding="utf-8"))
    qmap = json.load(open(f"{args.run}/question_map.json", encoding="utf-8"))

    cb = {}
    for p in sorted(glob.glob(f"{args.run}/closedbook/cb_*.jsonl")):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cb[r["qid"]] = r.get("closedbook", "")

    print(f"고유 질문 {len(qmap)} / 폐쇄형 확보 {len(cb)}")
    missing = [q for q in qmap if q not in cb]
    if missing:
        print(f"⚠️ 폐쇄형 미확보 {len(missing)}건 — 해당 항목은 제외한다")

    batch, cmap = [], {}
    i = 0
    for qid, uses in usage.items():
        if qid not in cb:
            continue
        for u in uses:
            i += 1
            jid = f"n{i:06d}"                      # 순번만. 출처 인코딩 금지
            batch.append({"jid": jid, "question": qmap[qid],
                          "gold": u["gold"], "closedbook": cb[qid]})
            cmap[jid] = {"cond": u["cond"], "domain": u["domain"],
                         "idx": u["idx"], "qid": qid}

    json.dump(batch, open(f"{args.out}/novelty_batch.json", "w"),
              ensure_ascii=False, indent=1)
    json.dump(cmap, open(f"{args.out}/novelty_map.json", "w"), ensure_ascii=False)

    leak = sorted({k for it in batch for k in it if k in FORBIDDEN})
    bad = [it["jid"] for it in batch if not it["jid"][1:].isdigit()][:3]
    print(f"채점 항목 {len(batch)}건")
    print(f"cell-blind: {'LEAK ' + str(leak) if leak else 'OK'}"
          f"{' | jid 이상 ' + str(bad) if bad else ''}")
    if leak or bad:
        raise SystemExit("cell-blind 위반. 중단.")

    from collections import Counter
    c = Counter((v["cond"], v["domain"]) for v in cmap.values())
    for k in sorted(c):
        print(f"  {k[0]}/{k[1]}: {c[k]}")


if __name__ == "__main__":
    main()
