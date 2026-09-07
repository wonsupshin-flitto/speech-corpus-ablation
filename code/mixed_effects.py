#!/usr/bin/env python3
"""혼합효과 모형 — 군집을 제대로 처리한 조건 효과 (논문1-1).

왜 필요한가
-----------
두 리뷰가 같은 지점을 지적했다.

    r1: "다중비교 회계가 감사 불가능", "항목이 세션 안에 군집"
    r2: "12셀이 같은 질문 집합을 공유하므로 독립 추출이 아니다.
         계층/혼합효과 모형(질문·세션 랜덤효과, 대응 대비)으로 셀 평균 요약을 대체하라"

셀 평균 요약은 두 가지를 못 잡는다. 같은 질문이 12셀에 반복 등장하는 것, 그리고
질문들이 40개 세션 안에 군집된 것. 둘 다 유효 표본을 실제보다 크게 보이게 한다.

모형
----
    score ~ condition + (1|session/question) + (1|model)

statsmodels 의 MixedLM 은 중첩 랜덤효과를 직접 못 받으므로, 세션을 그룹으로 두고
질문을 그 안의 variance component 로 넣는다. 모델은 별도 vc 로 넣는다.

    groups = session
    vc     = {"question": "0 + C(qid)", "model": "0 + C(model)"}

읽는 법
-------
조건 계수는 exp0 대비 차이다. 셀 평균 요약과 **부호가 같고 구간이 넓어지면** 군집을
무시했던 만큼 과신했다는 뜻이다. 좁아지면 셀 요약이 보수적이었다는 뜻이다.

사용:
    python scripts/rag/mixed_effects.py \
      --map <correctness_map.json> --score-dir <scores> \
      --questions <질문 jsonl> --conditions exp0,exp1_1,exp1_2,exp2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

SC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--score-dir", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--conditions", default="exp0,exp1_1,exp1_2,exp2")
    ap.add_argument("--baseline", default="exp0")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    cmap = json.load(open(args.map, encoding="utf-8"))
    judges = [json.load(open(f, encoding="utf-8"))
              for f in sorted(glob.glob(os.path.join(args.score_dir, "*.json")))]
    qmeta = {q["id"]: q for q in
             (json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip())}

    rows = []
    for jid, c in cmap.items():
        if c["cond"] not in conds:
            continue
        vs = [SC[j[jid]] for j in judges if jid in j and j[jid] in SC]
        if not vs:
            continue
        q = qmeta.get(c["qid"])
        if q is None:
            continue
        rows.append({"score": sum(vs) / len(vs), "cond": c["cond"],
                     "model": c["model"], "domain": c["domain"],
                     "qid": c["qid"], "session": str(q["si_seq"])})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("데이터 없음")
    df["cond"] = pd.Categorical(df["cond"], categories=[args.baseline] +
                                [c for c in conds if c != args.baseline])
    print(f"관측 {len(df)}건 | 조건 {df['cond'].nunique()} | 질문 "
          f"{df['qid'].nunique()} | 세션 {df['session'].nunique()} | "
          f"모델 {df['model'].nunique()}")

    # 질문·모델을 variance component 로, 세션을 그룹으로. 세션이 최상위 군집이다.
    vc = {"question": "0 + C(qid)", "model": "0 + C(model)"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("score ~ C(cond)", df, groups=df["session"],
                         vc_formula=vc, re_formula="0")
        try:
            r = md.fit(method="lbfgs", maxiter=200)
        except Exception as e:
            raise SystemExit(f"수렴 실패: {e}")

    print("\n" + "=" * 72)
    print(f"혼합효과: score ~ condition + (1|session) + (1|question) + (1|model)")
    print("=" * 72)
    print(f"{'항':30}{'계수':>10}{'표준오차':>10}{'95% CI':>22}{'p':>9}")
    for name in r.params.index:
        if not name.startswith("C(cond)"):
            continue
        lab = name.split("T.")[-1].rstrip("]")
        b, se_ = r.params[name], r.bse[name]
        print(f"  {lab + ' vs ' + args.baseline:28}{b:>+10.4f}{se_:>10.4f}"
              f"{f'[{b-1.96*se_:+.4f}, {b+1.96*se_:+.4f}]':>22}{r.pvalues[name]:>9.3f}")

    # re_formula="0" 이면 cov_re 가 비어 있고 분산은 전부 vcomp 에 들어간다.
    print("\n랜덤효과 분산")
    names = list(vc.keys())
    vc_arr = getattr(r, "vcomp", None)
    vcomp = list(vc_arr) if vc_arr is not None else []
    for k, v in zip(names, vcomp):
        print(f"  {k:20}{v:>10.5f}")
    if len(vcomp) > len(names):
        print(f"  {'session':20}{vcomp[-1]:>10.5f}")
    print(f"  {'잔차':20}{r.scale:>10.5f}")

    print("\n※ 셀 평균 요약과 부호가 같고 구간이 넓어지면, 군집을 무시했던 만큼")
    print("  과신했다는 뜻이다. 좁아지면 셀 요약이 보수적이었다는 뜻이다.")


if __name__ == "__main__":
    main()
