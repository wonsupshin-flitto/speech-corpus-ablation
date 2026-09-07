#!/usr/bin/env python3
"""분산 분해 — 질문을 늘리면 검정력이 오르는가 (논문1-1, 실행 전 판단용).

왜 이걸 먼저 하는가
-------------------
두 리뷰가 중심 null 의 검정력 부족을 최우선으로 지적했고, 대응책으로 질문 수 확대를
잡았다. 그런데 **질문을 늘려도 검정력이 안 오르는 경우가 있다.**

현재 분석 단위는 셀(모델 × 도메인) 12개다. 셀 단위에서 자유도는 12로 고정이고,
질문을 늘리면 셀 *안*의 잡음만 줄어든다. 셀 *사이*의 분산이 지배적이면 질문을 네 배로
늘려도 셀 단위 CI 는 거의 그대로다. 그 경우 재생성에 쓰는 GPU 시간이 헛돈이다.

그래서 기존 43문항 결과에서 분산을 두 성분으로 쪼갠다.

    between  셀마다 고유한 차이 (모델·도메인이 만드는 것). 질문을 늘려도 안 줄어든다
    within   같은 셀 안에서 질문마다 흔들리는 양. 질문 수 n 에 반비례해 줄어든다

셀 평균의 분산은 `between + within/n` 이다. 질문을 n1 -> n2 로 늘렸을 때 셀 단위
표준오차가 얼마나 줄어드는지 이 식으로 바로 계산된다.

판단
----
  줄어드는 폭이 작다  -> 셀 단위 TOST 는 확대해도 소용없다. 혼합효과 모형으로 간다
  줄어드는 폭이 크다  -> 확대가 값을 한다. 재생성할 이유가 있다

사용:
    python scripts/rag/variance_decomp.py \
      --map <RUN>/judge_inputs/correctness_map.json --score-dir <RUN>/scores \
      --contrast exp2-exp0 --target-n 163
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

SC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--score-dir", required=True)
    ap.add_argument("--contrast", default="exp2-exp0")
    ap.add_argument("--target-n", type=int, default=163)
    args = ap.parse_args()

    cmap = json.load(open(args.map, encoding="utf-8"))
    files = sorted(glob.glob(os.path.join(args.score_dir, "*.json")))
    judges = [json.load(open(f, encoding="utf-8")) for f in files]
    print(f"judge {len(judges)}종: {[os.path.basename(f) for f in files]}")

    # 2-judge 평균. 한쪽만 있으면 그것만 쓴다.
    val = {}
    for jid in cmap:
        vs = [SC[j[jid]] for j in judges if jid in j and j[jid] in SC]
        if vs:
            val[jid] = sum(vs) / len(vs)
    print(f"채점된 항목 {len(val)}/{len(cmap)}\n")

    a, b = args.contrast.split("-", 1)
    # (모델, 도메인, 질문) -> 조건별 점수
    cube = defaultdict(dict)
    for jid, s in val.items():
        c = cmap[jid]
        cube[(c["model"], c["domain"], c["qid"])][c["cond"]] = s

    # 셀 안에서 질문별 대비값을 모은다. 같은 질문의 두 조건이라 대응(paired)이다.
    cell = defaultdict(list)
    for (mo, dom, q), cs in cube.items():
        if a in cs and b in cs:
            cell[(mo, dom)].append(cs[a] - cs[b])

    cells = sorted(cell)
    if len(cells) < 2:
        raise SystemExit("셀이 2개 미만이다.")
    ns = [len(cell[k]) for k in cells]
    means = [st.mean(cell[k]) for k in cells]
    n1 = st.median(ns)

    # within: 셀 안 대비값의 분산을 셀들에 걸쳐 pool
    within = st.mean([st.variance(cell[k]) for k in cells if len(cell[k]) > 1])
    # 관측된 셀평균 분산 = between + within/n  ->  between 을 역산
    obs = st.variance(means)
    between = max(0.0, obs - within / n1)

    print("=" * 76)
    print(f"대비 {args.contrast} — 셀 {len(cells)}개, 셀당 질문 중앙값 {n1:.0f}")
    print("=" * 76)
    print(f"  셀평균의 평균          {st.mean(means):+.4f}")
    print(f"  관측 셀평균 분산       {obs:.5f}   (표준편차 {obs**0.5:.4f})")
    print(f"  within  (질문 잡음)    {within:.5f}")
    print(f"  between (셀 고유차)    {between:.5f}")
    share = between / obs * 100 if obs > 0 else 0
    print(f"  between 비중           {share:.0f}%")

    def se(n):
        return ((between + within / n) / len(cells)) ** 0.5

    s1, s2 = se(n1), se(args.target_n)
    print()
    print(f"  셀 단위 표준오차  n={n1:.0f}  {s1:.4f}")
    print(f"                    n={args.target_n}  {s2:.4f}")
    print(f"  감소폭            {(1 - s2 / s1) * 100:.1f}%")
    print(f"  95% CI 반폭       {1.96*s1:.4f}  ->  {1.96*s2:.4f}")

    # 질문을 무한히 늘려도 남는 하한
    floor = (between / len(cells)) ** 0.5
    print(f"  질문 무한대 극한  {floor:.4f}  (셀 수가 정하는 하한)")

    print()
    if share >= 60:
        print("판단: between 이 지배적이다. 질문을 늘려도 셀 단위 CI 는 거의 안 좁아진다.")
        print("      셀 단위 TOST 를 위해 재생성할 이유가 없다. 확대분의 값은")
        print("      질문·세션을 랜덤효과로 넣는 혼합효과 모형에서만 나온다.")
    elif share <= 30:
        print("판단: within 이 지배적이다. 질문 확대가 셀 단위 CI 를 실제로 좁힌다.")
        print("      재생성이 검정력을 산다.")
    else:
        print("판단: 두 성분이 비슷하다. 감소폭 수치를 보고 비용 대비로 결정한다.")

    # 셀 단위 TOST 를 지금 데이터로 미리 돌려본다 (SESOI 는 인자로 받는 게 아니라
    # 여기서는 감도만 본다). 어느 SESOI 부터 등가 판정이 되는지 역산한다.
    m = st.mean(means)
    need1 = abs(m) + 1.796 * s1     # t(11), 단측 0.05
    need2 = abs(m) + 1.796 * s2
    print()
    print(f"등가 판정에 필요한 최소 SESOI   n={n1:.0f}  {need1:.3f}")
    print(f"                                n={args.target_n}  {need2:.3f}")
    print("  (SESOI 가 이 값보다 커야 '차이 없음'을 통계로 말할 수 있다)")


if __name__ == "__main__":
    main()
