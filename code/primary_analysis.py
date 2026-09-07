#!/usr/bin/env python3
"""주분석 — 사전 등록대로 등가성 검정 (논문1-1).

사전 등록: `report/260828_equivalence_prereg/PREREGISTRATION.md` (수정 3)

    SESOI  ±0.045
    단위   셀(모델 × 도메인)
    주분석 중립 표현 문항만 (|gold-코퍼스 중첩차| ≤ 10.9pp)
    판정   등가 = 90% CI ⊂ 경계 / 차이 = 95% CI 가 0 배제

왜 표현 필터가 필요한가
-----------------------
확대 문항 120개는 전사문을 읽고 질문과 정답을 함께 썼다. 그 전사문은 Phase 0 이
교정한 판본이라, 정답 표현이 정제된 코퍼스와 겹치기 쉽다. 채점은 정답과의 일치를
보므로 정제 코퍼스가 유리해진다. **파이프라인의 성질이 아니라 문항을 만든 방식의
결함이다.**

필터 기준(10.9pp)은 **교란이 없는 전문가 43문항의 |중첩차| 80분위**다. 확대 문항의
정확도를 보지 않고 정한 값이다.

사용:
    python scripts/rag/primary_analysis.py
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

R = Path("paper_works/1-1.speech_corpus_rag/report")
SC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
SESOI = 0.045
THR = 0.109          # 사전 확정. 전문가 43문항 |중첩차| 80분위
T90, T95 = 1.796, 1.96   # t(11) 단측 0.05 / 정규 양측 0.05


def two_sided_p(t: float, df: int) -> float:
    """t 분포 양측 p. scipy 없이 불완전베타로 직접 낸다."""
    import math as _m
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    def betacf(a, b, x, it=200):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        h = d
        for m in range(1, it):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
            c = 1.0 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c
            d = 1.0 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
            c = 1.0 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c
            d = 1.0 / d
            de = d * c; h *= de
            if abs(de - 1.0) < 3e-12:
                break
        return h

    lbeta = _m.lgamma(a) + _m.lgamma(b) - _m.lgamma(a + b)
    front = _m.exp(a * _m.log(x) + b * _m.log(1 - x) - lbeta) if 0 < x < 1 else 0.0
    ib = front * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else \
         1.0 - _m.exp(b * _m.log(1 - x) + a * _m.log(x) - lbeta) * betacf(b, a, 1 - x) / b
    return min(1.0, max(0.0, ib))


def grams(s: str, n: int = 4):
    s = "".join(s.split())
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def load_corpus(cond: str, dom: str):
    p = Path(f"paper_works/1.expert_qa/data/training/main/{cond}_{dom}.jsonl")
    out = []
    for line in p.open(encoding="utf-8"):
        if line.strip():
            m = {x["role"]: x["content"] for x in json.loads(line)["messages"]}
            out.append((m.get("user", "").strip(), m.get("assistant", "").strip()))
    return out


def main() -> None:
    prov = json.loads((R / "260828_recall/provenance.json").read_text(encoding="utf-8"))
    qs = {json.loads(l)["id"]: json.loads(l)
          for l in (R / "260828_question_expansion/endpoint_qa_163.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()}
    corp = {(c, d): load_corpus(c, d) for c in ("exp0", "exp2")
            for d in ("medical", "finance")}

    # 문항별 gold-코퍼스 중첩차 (exp2 − exp0)
    def ovd(q):
        si, dom = str(q["si_seq"]), q["domain"]
        gg = grams(q["gold"])
        if not gg:
            return None
        o = {}
        for c in ("exp0", "exp2"):
            pm = prov.get(f"{c}_{dom}", {})
            cg = set()
            for u, a in corp[(c, dom)]:
                if si in (pm.get(u.strip()) or []):
                    cg |= grams(u + " " + a)
            if not cg:
                return None
            o[c] = len(gg & cg) / len(gg)
        return o["exp2"] - o["exp0"]

    OV = {qid: ovd(q) for qid, q in qs.items()}
    neutral = {q for q, v in OV.items() if v is not None and abs(v) <= THR}
    b43 = {q for q, v in qs.items() if v["origin"] == "base43"}
    n120 = {q for q, v in qs.items() if v["origin"] == "expand120"}
    print(f"중립 통과: 전문가 {len(neutral & b43)}/{len(b43)}, "
          f"신규 {len(neutral & n120)}/{len(n120)}, 합 {len(neutral)}")

    def collect(run: str, scores: list[str]):
        cmap = json.loads((R / run / "judge_inputs/correctness_map.json")
                          .read_text(encoding="utf-8"))
        js = [json.loads((R / run / "scores" / f).read_text(encoding="utf-8"))
              for f in scores]
        out = defaultdict(lambda: defaultdict(dict))
        for jid, c in cmap.items():
            vs = [SC[j[jid]] for j in js if jid in j and j[jid] in SC]
            if vs:
                out[(c["model"], c["domain"])][c["qid"]][c["cond"]] = sum(vs) / len(vs)
        return out

    A = collect("260827_rag_ablation", ["corr_gpt.json", "corr_sonnet.json"])
    B = collect("260828_rag_expand", ["corr_gpt.json", "corr_sonnet.json"])
    merged = defaultdict(dict)
    for src in (A, B):
        for k, qd in src.items():
            merged[k].update(qd)

    def tost(cells, label, filt=None, a="exp2", b="exp0"):
        ms, tot = [], 0
        for k in sorted(cells):
            d = [cs[a] - cs[b] for q, cs in cells[k].items()
                 if a in cs and b in cs and (filt is None or q in filt)]
            if d:
                ms.append(st.mean(d))
                tot += len(d)
        if len(ms) < 3:
            print(f"{label:32} 셀 부족")
            return
        n = len(ms)
        m = st.mean(ms)
        se = st.stdev(ms) / math.sqrt(n)
        lo90, hi90 = m - T90 * se, m + T90 * se
        lo95, hi95 = m - T95 * se, m + T95 * se
        eq = lo90 > -SESOI and hi90 < SESOI
        df = lo95 > 0 or hi95 < 0
        t = m / se if se else 0.0
        p = two_sided_p(t, n - 1)
        # 아홉 개 대비를 재므로 0.05/9 로 보정한다(사전 등록).
        mark = "✅" if p < 0.05 / 9 else ("△" if p < 0.05 else "✗")
        print(f"{label:32}셀{n:>3} 문항{tot // n:>4}  {m:+.4f}  "
              f"90%[{lo90:+.4f},{hi90:+.4f}]  95%[{lo95:+.4f},{hi95:+.4f}]  "
              f"{'등가 ✅' if eq else '등가 ✗'}  p={p:.4f} {mark}")

    print("\n" + "=" * 118)
    print("exp2 − exp0  (SESOI ±0.045, 2-judge 평균)")
    print("=" * 118)
    tost(merged, "★ 주분석: 중립 표현 문항", neutral)
    print("-" * 118)
    tost(A, "  전문가 43 전체")
    tost(A, "  전문가 43 중 중립", neutral)
    tost(B, "  신규 120 전체 (결함)")
    tost(B, "  신규 120 중 중립", neutral)
    tost(merged, "  163 전체 (필터 없음)")

    print("\n" + "=" * 118)
    print("다른 대비 (주분석과 같은 중립 필터)")
    print("=" * 118)
    for a, lab in (("exp1_1", "exp1_1 − exp0 (재작성만)"),
                   ("exp1_2", "exp1_2 − exp0 (필터만)")):
        tost(merged, f"  {lab}", neutral, a=a)
    tost(merged, "  exp2 − base (코퍼스 가치)", neutral, a="exp2", b="base")
    tost(merged, "  exp2 − mismatched", neutral, a="exp2", b="mismatched")

    # 전사 원천 대비. whisper 조건은 전문가 43문항으로만 생성했으므로 A 에서만 잰다.
    # 중립 필터는 exp2−exp0 축에서 정의된 값이라 이 축엔 사전 적용 대상이 아니다.
    # 그래도 같은 문항집합을 쓰는지 확인할 수 있게 둘 다 낸다.
    print("\n" + "=" * 118)
    print("전사 원천 (whisper 조건은 전문가 43문항으로만 생성)")
    print("=" * 118)
    for a, b, lab in (("exp0", "whisper_medium_exp0", "단계 off"),
                      ("exp2", "whisper_medium_exp2", "단계 on")):
        tost(A, f"  in-house − whisper, {lab} · 43 전체", None, a=a, b=b)
        tost(A, f"  in-house − whisper, {lab} · 중립만", neutral, a=a, b=b)


if __name__ == "__main__":
    main()
