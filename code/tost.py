#!/usr/bin/env python3
"""등가성 검정 (TOST) — 사전 등록대로 실행 (논문1-1).

사전 등록: `report/260828_equivalence_prereg/PREREGISTRATION.md` (결과 보기 전 커밋)

    SESOI = 0.045   품질 1-5 척도의 +0.18 이 하류 0-1 척도로 일대일 전이될 때의 값
    단위  = 셀(모델 × 도메인)
    판정  = 90% CI 가 [-SESOI, +SESOI] 안에 완전히 들어가면 등가

왜 90% CI 인가
--------------
TOST 는 양쪽 경계에 각각 단측 검정을 한다(각 alpha=0.05). 두 단측 검정을 동시에
통과하는 것은 **90% 신뢰구간이 경계 안에 들어가는 것**과 같다. 95% 를 쓰면 실제
유의수준이 0.025 가 되어 사전 등록과 달라진다.

해석 규칙도 사전에 못 박았다. 어느 쪽이 나오든 그대로 쓴다.

    CI ⊂ [-s,+s]              전이 없음을 통계로 말할 수 있다
    CI 가 경계를 넘고 0 포함    등가도 차이도 못 보였다. null 을 주장하지 않는다
    CI 가 0 배제               전이가 있다(부호대로 보고)

사용:
    python scripts/rag/tost.py --map <map.json> --score-dir <scores> \
      --contrast exp2-exp0 --sesoi 0.045
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

SC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

# t 분포 임계값. scipy 없이 돌리기 위해 필요한 자유도만 표로 둔다.
# 단측 0.05 (= 90% 양측 CI)
T90 = {3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833,
       10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753,
       16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725, 23: 1.714,
       29: 1.699, 39: 1.685, 59: 1.671, 119: 1.658}


def t_crit(df: int) -> float:
    if df in T90:
        return T90[df]
    ks = sorted(T90)
    if df < ks[0]:
        return T90[ks[0]]
    if df > ks[-1]:
        return 1.645                       # 정규 근사
    lo = max(k for k in ks if k <= df)
    hi = min(k for k in ks if k >= df)
    if lo == hi:
        return T90[lo]
    w = (df - lo) / (hi - lo)
    return T90[lo] * (1 - w) + T90[hi] * w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--score-dir", required=True)
    ap.add_argument("--contrast", default="exp2-exp0")
    ap.add_argument("--sesoi", type=float, default=0.045)
    args = ap.parse_args()

    cmap = json.load(open(args.map, encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(args.score_dir, "*.json")))
    judges = [json.load(open(f, encoding="utf-8")) for f in files]
    print(f"judge {len(judges)}종: {[os.path.basename(f) for f in files]}")

    val = {}
    for jid in cmap:
        vs = [SC[j[jid]] for j in judges if jid in j and j[jid] in SC]
        if vs:
            val[jid] = sum(vs) / len(vs)

    a, b = args.contrast.split("-", 1)
    cube = defaultdict(dict)
    for jid, s in val.items():
        c = cmap[jid]
        cube[(c["model"], c["domain"], c["qid"])][c["cond"]] = s

    cell = defaultdict(list)
    for (mo, dom, q), cs in cube.items():
        if a in cs and b in cs:
            cell[(mo, dom)].append(cs[a] - cs[b])

    ms = [st.mean(v) for v in sorted(cell.values(), key=len)]
    n = len(ms)
    if n < 3:
        raise SystemExit(f"셀이 {n}개다. 검정 불가.")
    m = st.mean(ms)
    se = st.stdev(ms) / math.sqrt(n)
    tc = t_crit(n - 1)
    lo, hi = m - tc * se, m + tc * se
    s = args.sesoi

    print(f"\n채점 항목 {len(val)} / 셀 {n}개 / 셀당 질문 "
          f"{st.median([len(v) for v in cell.values()]):.0f}\n")
    print("=" * 70)
    print(f"TOST — {args.contrast},  SESOI ±{s}")
    print("=" * 70)
    print(f"  차이 평균        {m:+.4f}")
    print(f"  표준오차         {se:.4f}   (셀 {n}개, t({n-1})={tc:.3f})")
    print(f"  90% CI           [{lo:+.4f}, {hi:+.4f}]")
    print(f"  등가 경계        [{-s:+.4f}, {+s:+.4f}]")

    inside = lo > -s and hi < s
    excl0 = lo > 0 or hi < 0
    print()
    if inside:
        print("  판정: 등가. 90% CI 가 경계 안에 완전히 들어간다.")
        print("        정제의 하류 전이가 없다고 통계로 말할 수 있다.")
    elif excl0:
        print(f"  판정: 차이 있음 ({'양' if m > 0 else '음'}의 방향). CI 가 0 을 배제한다.")
        print("        사전 등록 규칙대로 그대로 보고한다.")
    else:
        print("  판정: 등가도 차이도 보이지 않았다.")
        print("        null 을 주장하지 않는다. 검정력 한계로 기술한다.")
        need = (abs(m) + tc * se)
        print(f"        등가 판정에 필요했을 SESOI: {need:.4f}")
        print(f"        현재 SESOI 로 등가를 보려면 셀 표준오차가 "
              f"{(s - abs(m)) / tc:.4f} 이하여야 한다.")

    # 참고: 95% CI 도 같이 낸다. 본문에서 통상 구간을 물을 수 있다.
    print(f"\n  참고 95% CI      [{m - 1.96*se:+.4f}, {m + 1.96*se:+.4f}]")


if __name__ == "__main__":
    main()
