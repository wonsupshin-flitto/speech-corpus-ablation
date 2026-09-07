#!/usr/bin/env python3
"""Phase 2 신규성을 비공유 항목만으로 재계산 (논문1-1).

왜 필요한가
-----------
리뷰 r1: "Phase 2 신규성 null 은 상당 부분 기계적이다. 69~89% 가 동일 문자열이라
같은 closed-book 답변이 사상된다. 비공유 항목만으로 재계산이 필요하다."

지적이 맞다. Phase 2 는 필터라서 대부분의 QA 쌍을 그대로 통과시킨다. 통과한 쌍은
질문·정답이 글자 그대로 같으므로, 폐쇄형 답변도 같고 판정도 같다. 그 항목들을 포함해
평균을 내면 **두 조건이 같아지도록 강제된 부분**이 결과를 지배한다. null 이 나오는
것은 당연하고, 그 null 은 Phase 2 에 대해 아무것도 말해주지 않는다.

그래서 **Phase 2 가 실제로 손댄 항목만**으로 다시 잰다.

    exp0 → exp1_2   Phase 2 만 켠 대비 (Phase 0 꺼짐)
    exp1_1 → exp2   Phase 2 만 켠 대비 (Phase 0 켜짐)

각 대비에서 질문 텍스트가 양쪽에 다 있는 항목은 "공유"로 보고 제외한다. 남은 것이
Phase 2 가 떨어뜨렸거나 새로 만든 항목이다.

읽는 법
-------
비공유만 남기면 n 이 크게 줄어 CI 가 넓어진다. 그것이 정직한 그림이다. 넓은 CI 는
"차이가 없다"가 아니라 "이 설계로는 못 잰다"를 뜻한다.

사용:
    python scripts/rag/novelty_nonshared.py --run <RUN 디렉토리>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statlib import bootstrap_ci

AXES = ("novel", "grounded", "correct")
# Phase 2 만 토글하는 대비. (기준, 처치)
PAIRS = [("exp0", "exp1_2"), ("exp1_1", "exp2")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--novel-judge", default="gpt")
    args = ap.parse_args()

    ji = os.path.join(args.run, "judge_inputs")
    cmap = json.load(open(os.path.join(ji, "novelty_map.json"), encoding="utf-8"))
    qmap = json.load(open(os.path.join(args.run, "question_map.json"), encoding="utf-8"))

    import glob
    files = sorted(glob.glob(os.path.join(args.run, "scores", "*.json")))
    judges = [json.load(open(f, encoding="utf-8")) for f in files]
    nov = [json.load(open(f, encoding="utf-8")) for f in files
           if args.novel_judge in os.path.basename(f)] or judges

    # 판정 병합 (기존 집계와 같은 규칙: 축별 AND, novel 만 전용 judge)
    merged = {}
    for jid in cmap:
        vs = [v for v in (j.get(jid) for j in judges) if v]
        nv = [v for v in (j.get(jid) for j in nov) if v]
        if vs and nv:
            merged[jid] = {
                "novel": all(bool(v.get("novel")) for v in nv),
                "grounded": all(bool(v.get("grounded")) for v in vs),
                "correct": all(bool(v.get("correct")) for v in vs),
            }

    # 조건 -> {질문텍스트: 유효신규 여부}
    by = defaultdict(dict)
    for jid, m in merged.items():
        c = cmap[jid]
        qt = qmap.get(c["qid"])
        if qt is None:
            continue
        eff = m["novel"] and m["grounded"] and m["correct"]
        by[c["cond"]][qt.strip()] = 1.0 if eff else 0.0

    print("=" * 78)
    print("Phase 2 대비 — 전체 vs 비공유 항목만")
    print("=" * 78)
    for base, treat in PAIRS:
        A, B = by.get(base, {}), by.get(treat, {})
        if not A or not B:
            print(f"  {base}->{treat}: 데이터 없음")
            continue
        shared = set(A) & set(B)
        onlyA = [A[q] for q in A if q not in shared]
        onlyB = [B[q] for q in B if q not in shared]

        ma, la, ha = bootstrap_ci(list(A.values()))
        mb, lb, hb = bootstrap_ci(list(B.values()))
        print(f"\n  {base} -> {treat}   (Phase 2 만 토글)")
        print(f"    공유 항목 {len(shared)}건 "
              f"= {base} 의 {len(shared)/len(A)*100:.0f}%, "
              f"{treat} 의 {len(shared)/len(B)*100:.0f}%")
        print(f"    전체     {base} {ma*100:.1f}%  {treat} {mb*100:.1f}%  "
              f"델타 {(mb-ma)*100:+.1f}pp")

        if len(onlyA) < 2 or len(onlyB) < 2:
            print(f"    비공유   너무 적다 ({len(onlyA)} / {len(onlyB)}). 재계산 불가")
            continue
        na, lna, hna = bootstrap_ci(onlyA)
        nb, lnb, hnb = bootstrap_ci(onlyB)
        print(f"    비공유   {base} {na*100:.1f}% [{lna*100:.1f},{hna*100:.1f}] n={len(onlyA)}  "
              f"{treat} {nb*100:.1f}% [{lnb*100:.1f},{hnb*100:.1f}] n={len(onlyB)}")
        # 비공유 집합은 서로 다른 항목이라 대응이 아니다. 두 독립 평균의 차이를
        # 부트스트랩해 CI 를 직접 낸다. 각 평균의 CI 가 겹친다고 차이가 없다는
        # 뜻은 아니므로(흔한 오독), 차이 자체의 CI 를 봐야 한다.
        import random as _r
        rng = _r.Random(777)
        ds = []
        for _ in range(10000):
            sa = sum(onlyA[rng.randrange(len(onlyA))] for _ in onlyA) / len(onlyA)
            sb = sum(onlyB[rng.randrange(len(onlyB))] for _ in onlyB) / len(onlyB)
            ds.append(sb - sa)
        ds.sort()
        lo, hi = ds[250], ds[9750]
        sig = "0 배제" if lo > 0 or hi < 0 else "0 포함"
        print(f"             델타 {(nb-na)*100:+.1f}pp  "
              f"95% CI [{lo*100:+.1f}, {hi*100:+.1f}]  {sig}")
        # 공유 항목만의 델타는 정의상 0 이어야 한다. 검산.
        sh = [(A[q], B[q]) for q in shared]
        d = sum(b - a for a, b in sh) / len(sh) if sh else 0.0
        print(f"    검산     공유 항목 델타 {d*100:+.2f}pp "
              f"(0 이 아니면 같은 텍스트에 판정이 갈렸다는 뜻)")

    print()
    print("※ 비공유만 남기면 n 이 줄어 CI 가 넓어진다. 넓은 CI 는 '차이가 없다'가")
    print("  아니라 '이 설계로는 못 잰다'를 뜻한다.")


if __name__ == "__main__":
    main()
