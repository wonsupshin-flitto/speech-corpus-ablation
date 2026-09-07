#!/usr/bin/env python3
"""교차 설계 신규성 집계 — 두 응시자를 각각, 그리고 함께 본다 (논문1-1).

배경
----
기존 E3 는 `claude-sonnet-5` 하나만 폐쇄형에 응시시켰다. 그래서 신규성이
"한 모델이 모르는 것" 에 머물렀다. 교차 설계에서는 `gpt-5.6-sol` 도 응시하고
서로를 채점한다. 자기 채점이 없고, "두 모델 다 모르는 것" 이라는 더 단단한
하한이 나온다.

    응시 claude-sonnet-5  → 채점 gpt-5.6-sol   (기존 실행 260827)
    응시 gpt-5.6-sol      → 채점 claude-sonnet-5 (이번 실행 260906)

세 값을 낸다
------------
    상한  둘 중 하나라도 못 맞힘   (관대)
    각각  응시자별 신규율          (비교용)
    하한  둘 다 못 맞힘            (엄격)

`grounded` 와 `correct` 는 우리 QA 자체를 보는 축이라 응시자와 무관하다.
두 실행의 판정을 AND 로 합쳐 보수적으로 쓴다.

사용:
    python scripts/rag/aggregate_novelty_cross.py \
      --old <260827_RUN> --new <260906_RUN>
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import glob
import json
from collections import defaultdict

from statlib import bootstrap_ci as bootstrap

CONDS = ["exp0", "exp1_1", "exp1_2", "exp2"]
NBOOT = 10000
SEED = 777


def load_scores(path_glob: str) -> dict:
    """채점 파일들을 jid -> {axis: bool} 로 모은다."""
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(path_glob)):
        try:
            rows = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ {f}: {type(e).__name__}")
            continue
        if isinstance(rows, dict):          # {jid: {...}} 형식
            rows = [{"jid": k, **v} for k, v in rows.items()]
        for r in rows:
            j = r.get("jid")
            if j:
                out[j] = {a: bool(r.get(a)) for a in ("novel", "grounded", "correct")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="260827 실행 디렉토리")
    ap.add_argument("--new", required=True, help="260906 교차 실행 디렉토리")
    args = ap.parse_args()

    # 기존 실행: sonnet 응시 → gpt 채점 (novel 축 정본)
    old = load_scores(f"{args.old}/scores/nov_gpt.json")
    old_map = json.load(open(f"{args.old}/judge_inputs/novelty_map.json", encoding="utf-8"))

    # 이번 실행: sol 응시 → sonnet 채점
    new = load_scores(f"{args.new}/scored/*.json")
    new.update(load_scores(f"{args.new}/scored_small/*.json"))
    new_map = json.load(open(f"{args.new}/judge_inputs/novelty_map.json", encoding="utf-8"))

    print(f"기존(sonnet 응시) {len(old):,}건 | 교차(sol 응시) {len(new):,}건")

    # 두 실행의 jid 는 서로 다른 순번이다. qid+cond+idx 로 맞춘다.
    def key(m, j):
        v = m[j]
        return (v["qid"], v["cond"], v["domain"], v["idx"])

    old_by = {key(old_map, j): v for j, v in old.items() if j in old_map}
    new_by = {key(new_map, j): v for j, v in new.items() if j in new_map}
    shared = sorted(set(old_by) & set(new_by))
    print(f"양쪽 모두 채점된 항목 {len(shared):,}건 "
          f"(교차 채점 진행률 {len(new_by)/len(old_by)*100:.0f}%)")
    if not shared:
        raise SystemExit("겹치는 항목이 없다. 매핑을 확인하라.")

    # 조건별 집계
    rows = defaultdict(list)
    for k in shared:
        o, n = old_by[k], new_by[k]
        # gating 축은 두 판정의 AND (보수적)
        gate = (o["grounded"] and n["grounded"]) and (o["correct"] and n["correct"])
        rows[k[1]].append({
            "sonnet_missed": o["novel"],      # sonnet 이 못 맞힘
            "sol_missed": n["novel"],         # sol 이 못 맞힘
            "gate": gate,
        })

    def pct(vals):
        return 100.0 * sum(vals) / len(vals) if vals else 0.0

    print(f"\n{'조건':<8}{'n':>7}{'sonnet':>9}{'sol':>8}{'둘중하나':>10}{'둘다':>8}")
    print("-" * 50)
    summary = {}
    for c in CONDS:
        r = rows.get(c, [])
        if not r:
            continue
        n = len(r)
        son = [x["sonnet_missed"] and x["gate"] for x in r]
        sol = [x["sol_missed"] and x["gate"] for x in r]
        either = [(x["sonnet_missed"] or x["sol_missed"]) and x["gate"] for x in r]
        both = [(x["sonnet_missed"] and x["sol_missed"]) and x["gate"] for x in r]
        summary[c] = {"n": n, "sonnet": pct(son), "sol": pct(sol),
                      "either": pct(either), "both": pct(both),
                      "_both": both, "_either": either}
        print(f"{c:<8}{n:>7,}{pct(son):>8.1f}%{pct(sol):>7.1f}%"
              f"{pct(either):>9.1f}%{pct(both):>7.1f}%")

    # 대비 (Exp 0 기준), 두 기준 각각
    print("\n대비 (vs Exp~0), 95% CI")
    for label, fld in (("둘 중 하나 (상한)", "_either"), ("둘 다 (하한)", "_both")):
        print(f"  [{label}]")
        base = summary["exp0"][fld]
        for c in CONDS[1:]:
            if c not in summary:
                continue
            cur = summary[c][fld]
            # 두 조건은 같은 질문 집합을 쓰므로 항목별 차이를 재표집한다.
            diffs = [(1.0 if a else 0.0) - (1.0 if b else 0.0)
                     for a, b in zip(cur, base)]
            m, lo, hi = bootstrap(diffs, nboot=NBOOT, seed=SEED)
            print(f"    {c:<8} {m*100:+6.1f}pp  [{lo*100:+.1f}, {hi*100:+.1f}]")

    out = {c: {k: v for k, v in d.items() if not k.startswith("_")}
           for c, d in summary.items()}
    p = f"{args.new}/aggregate_cross.json"
    _os.makedirs(_os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"\n저장 {p}")


if __name__ == "__main__":
    main()
