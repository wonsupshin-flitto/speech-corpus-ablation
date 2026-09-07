#!/usr/bin/env python3
"""RAG ablation judge 입력 생성 — cell-blind (논문1-1 E1/E2).

원본 `build_judge_A.py` 의 조건 확장판. 조건 목록이 고정 리스트였던 것을 인자로 뺐다.

cell-blind 가 왜 필수인가
-------------------------
judge 입력에 어느 조건·모델이 만든 답인지 보이면 그 선입견으로 채점한다. 특히
cross-provider 채점에서 크다(`.claude/rules/judge_model_policy.md`).

그래서 judge 에는 `{jid, question, gold, answer}` 만 보내고, jid → (모델/도메인/조건)
매핑은 **별도 파일**에 두어 집계 단계에서만 붙인다. jid 는 순번뿐이라 출처를
역추적할 수 없다.

원본과 동일하게 **금지 필드 누출 검사**를 내장한다.

사용:
    python scripts/rag/build_judge_ablation.py \\
      --gen-dir <gen_rag_ablation 출력> \\
      --questions .../endpoint_qa_43.jsonl \\
      --conditions base,exp0,exp1_1,exp1_2,exp2,mismatched \\
      --out <judge 입력 디렉토리>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# judge 입력에 절대 들어가면 안 되는 키 (judge_model_policy.md)
FORBIDDEN = {"model", "model_name", "provider", "generator", "judge_target",
             "cond", "condition", "experiment", "exp_id", "treatment",
             "cell", "cell_id", "group", "source", "origin"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--out", required=True)
    # 기존 43문항 채점이 a00001.. 을 이미 썼다. 확대분을 같은 접두로 만들면 두 실행의
    # 산출물을 합칠 때 id 가 충돌해 조용히 덮어써진다(260828 실측 사고 재발 방지).
    ap.add_argument("--jid-prefix", default="a")
    args = ap.parse_args()
    if len(args.jid_prefix) != 1 or not args.jid_prefix.isalpha():
        raise SystemExit("--jid-prefix 는 알파벳 한 글자여야 한다(jid 검사가 그 전제)")

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    os.makedirs(args.out, exist_ok=True)

    qmap = {q["id"]: q for q in
            (json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip())}

    batch, cmap = [], {}
    i = 0
    missing = []
    for f in sorted(glob.glob(os.path.join(args.gen_dir, "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for cond in conds:
                if cond not in r:
                    missing.append((r.get("model"), r.get("id"), cond))
                    continue
                i += 1
                jid = f"{args.jid_prefix}{i:05d}"   # 순번만. 출처 정보 인코딩 금지
                batch.append({"jid": jid,
                              "question": qmap[r["id"]]["question"],
                              "gold": qmap[r["id"]]["gold"],
                              "answer": r[cond]})
                cmap[jid] = {"model": r["model"], "domain": r["domain"],
                             "qid": r["id"], "cond": cond,
                             # 질문 출처. judge 입력이 아니라 집계용이다.
                             "qorigin": qmap[r["id"]].get("origin", "base43")}

    json.dump(batch, open(os.path.join(args.out, "correctness_batch.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump(cmap, open(os.path.join(args.out, "correctness_map.json"), "w"),
              ensure_ascii=False, indent=1)

    # ── cell-blind 검사 ───────────────────────────────────────────────────
    leak_keys = sorted({k for it in batch for k in it if k in FORBIDDEN})
    # jid 에 출처가 인코딩됐는지도 본다(`gpt4o_q001` 같은 접두 금지)
    bad_jid = [it["jid"] for it in batch if not it["jid"][1:].isdigit()][:3]

    print(f"항목 {len(batch)}건 | 조건 {conds}")
    print(f"cell-blind: {'LEAK ' + str(leak_keys) if leak_keys else 'OK'}"
          f"{' | jid 이상 ' + str(bad_jid) if bad_jid else ''}")
    if missing:
        print(f"⚠️ 생성 누락 {len(missing)}건 (앞 3): {missing[:3]}")
    if leak_keys or bad_jid:
        raise SystemExit("cell-blind 위반. dispatch 중단.")


if __name__ == "__main__":
    main()
