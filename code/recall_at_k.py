#!/usr/bin/env python3
"""세션 단위 recall@k — 검색기가 정답 근거를 실제로 꺼내오는가 (논문1-1).

왜 필요한가
-----------
리뷰 r1-H6: "검색 실험인데 recall@3 이 없다." 조건마다 인덱스를 갈아끼우고 답변
정확도만 봤으므로, 차이가 **내용** 때문인지 **검색기 거동** 때문인지 분리되지 않았다.

recall 은 답변 생성 없이 계산된다. LLM judge 호출이 0회다. 따라서 judge 편향과
무관하게 기계적으로 재현되고, 이 논문에서 유일하게 판정 모델에 의존하지 않는 지표다.

무엇을 재는가
-------------
질문은 특정 세션의 전사문에서 만들었다. 그 질문의 근거는 **같은 세션에서 생성된 QA
쌍**이다. 그래서

    recall@k = top-k 안에 질문의 출처 세션에서 나온 항목이 하나라도 있는 비율

세션 단위로 재는 이유: 정제 조건마다 QA 쌍의 문면이 달라 항목 단위 정답 매칭이
불가능하다. exp0 과 exp2 의 질문 중복은 1.5% 뿐이다. 세션은 조건이 바뀌어도 보존되는
유일한 공통 좌표다.

보조 지표
---------
    sess@1     top-1 이 출처 세션인 비율. 더 엄격
    mean_rank  출처 세션 항목이 처음 나타나는 순위(없으면 k+1로 절단)

검색 설정은 생성 실험과 동일하게 고정한다(bge-m3, 정규화 내적). 다르면 비교가 깨진다.

사용:
    python scripts/rag/recall_at_k.py \
      --corpus-dir <코퍼스> --provenance <provenance.json> \
      --questions <질문 jsonl> --out <결과 json> --k 3
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np


def load_corpus(corpus_dir: str, cond: str, domain: str):
    path = os.path.join(corpus_dir, f"{cond}_{domain}.jsonl")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = {x["role"]: x["content"] for x in json.loads(line)["messages"]}
            out.append((m.get("user", ""), m.get("assistant", "")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--provenance", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--conditions",
                    default="exp0,exp1_1,exp1_2,exp2,"
                            "whisper_medium_exp0,whisper_medium_exp2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    prov = json.load(open(args.provenance, encoding="utf-8"))
    qs = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(f"질문 {len(qs)}건 / 조건 {len(conds)}개 / k={args.k}")

    emb = SentenceTransformer("BAAI/bge-m3", device=args.device)

    def enc(texts):
        return emb.encode(texts, normalize_embeddings=True, batch_size=48,
                          show_progress_bar=False)

    # 질문은 조건과 무관하므로 한 번만 임베딩한다.
    qv_all = enc([q["question"] for q in qs])

    results = defaultdict(list)     # (cond, domain) -> per-question 기록
    for cond in conds:
        for domain in ("medical", "finance"):
            key = f"{cond}_{domain}"
            pm = prov.get(key)
            if pm is None:
                print(f"  ⚠️ {key}: 출처 매핑 없음, 건너뜀")
                continue
            corpus = load_corpus(args.corpus_dir, cond, domain)
            # 코퍼스 각 항목의 출처 세션. 매핑에 없으면 None (recall 판정에서 미스)
            owners = [pm.get(q.strip()) for q, _ in corpus]
            unmapped = sum(1 for o in owners if not o)
            cv = enc([q + " " + a for q, a in corpus])

            idxs = [i for i, q in enumerate(qs) if q["domain"] == domain]
            for i in idxs:
                sims = cv @ qv_all[i]
                top = np.argsort(-sims)[:max(args.k, 10)]
                si = str(qs[i]["si_seq"])
                rank = None
                for r, j in enumerate(top):
                    o = owners[j]
                    if o and si in o:
                        rank = r + 1
                        break
                results[key].append({
                    "qid": qs[i]["id"], "si_seq": si,
                    "hit_at_k": bool(rank is not None and rank <= args.k),
                    "hit_at_1": bool(rank == 1),
                    "rank": rank if rank is not None else args.k + 1,
                })
            print(f"  {key:28} 코퍼스 {len(corpus):>5}  미매핑 {unmapped:>3}  "
                  f"질문 {len(idxs):>3}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), ensure_ascii=False)

    print()
    print("=" * 74)
    print(f"세션 단위 recall@{args.k}")
    print("=" * 74)
    print(f"{'코퍼스':30}{'n':>6}{f'recall@{args.k}':>12}{'sess@1':>10}{'평균순위':>10}")
    agg = defaultdict(list)
    for key in sorted(results):
        rs = results[key]
        n = len(rs)
        rk = sum(r["hit_at_k"] for r in rs) / n * 100
        r1 = sum(r["hit_at_1"] for r in rs) / n * 100
        mr = sum(r["rank"] for r in rs) / n
        print(f"{key:30}{n:>6}{rk:>11.1f}%{r1:>9.1f}%{mr:>10.2f}")
        agg[key.rsplit("_", 1)[0]] += rs

    print()
    print(f"{'조건 (도메인 합산)':30}{'n':>6}{f'recall@{args.k}':>12}{'sess@1':>10}")
    for cond in sorted(agg):
        rs = agg[cond]
        n = len(rs)
        print(f"{cond:30}{n:>6}"
              f"{sum(r['hit_at_k'] for r in rs)/n*100:>11.1f}%"
              f"{sum(r['hit_at_1'] for r in rs)/n*100:>9.1f}%")
    print(f"\n저장 {args.out}")
    print("※ judge 호출 0회. 이 지표만은 판정 모델에 의존하지 않는다.")


if __name__ == "__main__":
    main()
