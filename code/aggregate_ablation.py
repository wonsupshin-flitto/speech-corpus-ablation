#!/usr/bin/env python3
"""RAG ablation 집계 — 조건별 correctness + paired bootstrap (논문1-1 E1/E2).

원본 `aggregate_A.py` 의 조건 확장판. 원본은 base/ours/mismatched 3조건 고정이었다.
채점 환산(correct/partial/incorrect → 1.0/0.5/0.0), bootstrap(10,000회, seed 777)은
**원본 그대로** 둔다(기존 실험 A 수치와의 비교 가능성).

왜 paired 인가
--------------
같은 (모델, 도메인, 질문) 셀에서 조건만 바꾼 값을 짝지어 뺀다. 질문 난이도·모델
실력 같은 공통 요인이 상쇄돼 조건 차이만 남는다.

사용:
    python scripts/rag/aggregate_ablation.py \\
      --judge-dir <judge입력> --score-dir <judge출력> \\
      --gen-dir <생성물> \\
      --conditions base,exp0,exp1_1,exp1_2,exp2,mismatched \\
      --contrasts exp2-exp0,exp2-mismatched,exp2-base,exp1_1-exp0,exp1_2-exp0
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import glob
import json
import os
from collections import defaultdict

SC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
NBOOT = 10000
SEED = 777


from statlib import bootstrap_ci as bootstrap, paired_t  # 깨진 자체 LCG 대체(260827)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-dir", required=True)
    ap.add_argument("--score-dir", required=True)
    ap.add_argument("--gen-dir")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--contrasts", default="",
                    help="쉼표 구분, 각 항목 'A-B' 형식. 비우면 첫 조건 대비 전부")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    cmap = json.load(open(os.path.join(args.judge_dir, "correctness_map.json"), encoding="utf-8"))

    # judge 출력 파일을 모두 읽어 평균(2-judge soft mean)
    judge_files = sorted(glob.glob(os.path.join(args.score_dir, "*.json")))
    judges = [json.load(open(f, encoding="utf-8")) for f in judge_files]
    print(f"judge {len(judges)}종: {[os.path.basename(f) for f in judge_files]}")

    # ── 가드: 서로 다른 실행의 매핑·채점을 조인하면 안 된다 ────────────────
    # 260827 사고: 매핑 258건과 채점 774건을 순번 키로 조인해 모델 라벨이 밀렸다.
    # 키가 겹쳐 조인은 성공하고 결손도 안 뜬다. 값도 그럴듯해 사후 검증으로 안 걸린다.
    jmax = max((len(j) for j in judges), default=0)
    if jmax > len(cmap) * 1.05:
        raise SystemExit(
            f"❌ 매핑 {len(cmap)}건 < 채점 {jmax}건. 다른 실행의 산출물일 수 있다.\n"
            f"   순번 id 는 실행 안에서만 유효하다. 같은 실행의 짝인지 확인할 것.")

    score = {}
    for jid in cmap:
        vs = [SC.get(j.get(jid)) for j in judges]
        vs = [v for v in vs if v is not None]
        if vs:
            score[jid] = sum(vs) / len(vs)
    print(f"판정된 항목 {len(score)}/{len(cmap)}")

    # (model, domain, qid) -> {cond: score}
    cell = defaultdict(dict)
    for jid, m in cmap.items():
        if jid in score:
            cell[(m["model"], m["domain"], m["qid"])][m["cond"]] = score[jid]

    # ── 조건별 평균 ───────────────────────────────────────────────────────
    print("\n=== 조건별 correctness (2-judge soft mean) ===")
    hdr = f"{'cell':26} {'n':>4}" + "".join(f"{c[:10]:>11}" for c in conds)
    print(hdr)
    for (model, domain) in sorted({(m, d) for (m, d, q) in cell}):
        rows = [cc for (mo, do, q), cc in cell.items() if mo == model and do == domain]
        rows = [r for r in rows if all(c in r for c in conds)]
        if not rows:
            continue
        line = f"{model+'/'+domain:26} {len(rows):>4}"
        for c in conds:
            line += f"{sum(r[c] for r in rows)/len(rows):>11.3f}"
        print(line)

    # ── 대조 ──────────────────────────────────────────────────────────────
    contrasts = [c.strip() for c in args.contrasts.split(",") if c.strip()]
    if not contrasts and len(conds) > 1:
        ref = conds[0]
        contrasts = [f"{c}-{ref}" for c in conds[1:]]

    print("\n=== paired 대조 (95% CI) ===")
    print(f"{'대조':28} {'n':>5} {'평균':>9} {'95% CI':>22}")
    for con in contrasts:
        a, b = con.split("-", 1)
        vals = [cc[a] - cc[b] for cc in cell.values() if a in cc and b in cc]
        if not vals:
            print(f"{con:28} {'(데이터 없음)':>5}")
            continue
        m, lo, hi = bootstrap(vals)
        sig = "  ✅" if (lo > 0 or hi < 0) else ""
        print(f"{con:28} {len(vals):>5} {m:>+9.3f}   [{lo:+.3f}, {hi:+.3f}]{sig}")
    print("\n✅ = 95% CI 가 0 을 포함하지 않음")

    # ── 셀 단위 대조 (보수적 판정) ────────────────────────────────────────
    # 위 질문 단위 검정은 같은 모델의 질문들이 서로 독립이 아니라 CI 가 좁게 나온다.
    # 모델×도메인 셀로 평균을 올려 다시 잰다. **논문에는 이쪽을 쓴다.**
    print("\n=== 셀 단위 대조 (모델×도메인 평균, 보수적) ===")
    print(f"{'대조':28} {'n':>5} {'평균':>9} {'95% CI':>22}  {'t':>7}")
    for con in contrasts:
        a, b = con.split("-", 1)
        g = defaultdict(list)
        for (mo, do, _), cc in cell.items():
            if a in cc and b in cc:
                g[(mo, do)].append(cc[a] - cc[b])
        cells = [sum(v) / len(v) for v in g.values()]
        if len(cells) < 2:
            print(f"{con:28} {'(셀 부족)':>5}")
            continue
        mm, _sd, t, _df, lo, hi, isg = paired_t(cells)
        pos = sum(1 for x in cells if x > 0)
        print(f"{con:28} {len(cells):>5} {mm:>+9.3f}   [{lo:+.3f}, {hi:+.3f}]"
              f"{'  ✅' if isg else '    '}{t:>7.2f}   양수 {pos}/{len(cells)}")
    print("  ※ 질문 단위 CI 는 반복 측정 때문에 좁다. 이 표가 논문용 정본이다.")

    # ── 길이 (length bias 배제 근거) ──────────────────────────────────────
    if args.gen_dir:
        alen = defaultdict(list)
        for f in glob.glob(os.path.join(args.gen_dir, "*.jsonl")):
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                for c in conds:
                    if c in r:
                        alen[c].append(len(r[c]))
        print("\n=== 답변 길이 (평균 글자수) ===")
        for c in conds:
            if alen[c]:
                print(f"  {c:20} {sum(alen[c])/len(alen[c]):.0f}")


if __name__ == "__main__":
    main()
