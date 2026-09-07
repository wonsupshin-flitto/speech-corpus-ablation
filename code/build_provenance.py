#!/usr/bin/env python3
"""코퍼스 항목 -> 세션(si_seq) 출처 매핑 (논문1-1 recall@3 용).

왜 필요한가
-----------
리뷰 r1-H6: "검색 실험인데 recall@3 이 없다. 인덱스를 바꿨을 때 정답 근거 쌍이 실제로
검색되는지 미보고." 내용의 차이와 검색기 거동의 차이가 분리되지 않는다는 지적이다.

recall 을 재려면 "이 질문의 정답 근거가 코퍼스의 어느 항목인가"를 알아야 한다.
질문은 세션 전사문에서 만들었으므로, **그 세션에서 나온 QA 쌍**이 근거 후보다.
그런데 학습용 JSONL(`training/main/*.jsonl`)은 messages 만 담고 있어 출처가 없다.

출처는 raw 레이아웃에 디렉토리로 남아 있다.

    raw/{domain}/{si_seq}/{condition}_{YYMMDD}.jsonl

그래서 raw 를 훑어 질문 텍스트 -> si_seq 사전을 만든다. 질문 텍스트는 raw 와 training
사이에서 그대로 보존된다(260828 확인).

주의: 같은 질문 텍스트가 두 세션에 나오면 모호하다. 그런 항목은 세션 집합을 전부
담아 두고, recall 판정에서 "질문의 세션이 그 집합에 있으면 히트"로 처리한다.

사용:
    python scripts/rag/build_provenance.py \
      --raw paper_works/1.expert_qa/data/raw --out <출력 json>
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

# raw 디렉토리의 도메인 폴더명 -> 코퍼스 파일명의 도메인
DOMAINS = ("medical", "finance")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # (condition, domain) -> {question_text: [si_seq, ...]}
    prov = defaultdict(lambda: defaultdict(set))
    stat = Counter()

    for dom in DOMAINS:
        d = Path(args.raw) / dom
        if not d.is_dir():
            print(f"⚠️ {d} 없음")
            continue
        for sess in sorted(d.iterdir()):
            if not sess.is_dir():
                continue
            si = sess.name
            for f in sorted(sess.glob("*.jsonl")):
                if "needs_review" in f.name:
                    continue                      # 학습셋에서 제외되는 파일
                cond = f.name.rsplit("_", 1)[0]   # exp1_1_260429.jsonl -> exp1_1
                for line in f.open(encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    q = (r.get("question") or "").strip()
                    if q:
                        prov[(cond, dom)][q].add(si)
                        stat[(cond, dom)] += 1

    out = {}
    amb = 0
    for (cond, dom), m in prov.items():
        key = f"{cond}_{dom}"
        out[key] = {q: sorted(s) for q, s in m.items()}
        amb += sum(1 for s in m.values() if len(s) > 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False)

    print(f"{'코퍼스':26}{'고유질문':>10}{'raw행':>10}{'세션':>8}")
    for key in sorted(out):
        cond, dom = key.rsplit("_", 1)
        n_sess = len({s for ss in out[key].values() for s in ss})
        print(f"{key:26}{len(out[key]):>10}{stat[(cond, dom)]:>10}{n_sess:>8}")
    print(f"\n두 세션 이상에 걸친 질문 {amb}건 (recall 판정에서 집합으로 처리)")
    print(f"저장 {args.out}")


if __name__ == "__main__":
    main()
