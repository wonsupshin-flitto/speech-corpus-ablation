#!/usr/bin/env python3
"""4개 조건을 가로지르는 전 지표 종합표 (논문1-1).

조건이 진행될수록 무엇이 좋아지고 무엇이 안 변하는지를 한 자리에서 본다.
논문1(코퍼스 품질)과 논문1-1(downstream)의 수치를 같은 축에 놓는다.

조건
    exp0     정제 없음
    exp1_1   Phase 0 만 (전사문 정제)
    exp1_2   Phase 2 만 (QA 품질 정제)
    exp2     둘 다
"""
from __future__ import annotations

import glob
import json
import statistics as st
from collections import defaultdict

CONDS = ["exp0", "exp1_1", "exp1_2", "exp2"]
SC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
R1 = "paper_works/1-1.speech_corpus_rag/report/260827_rag_ablation"
R3 = "paper_works/1-1.speech_corpus_rag/report/260827_novelty_closedbook"
CORPUS = "paper_works/1.expert_qa/data/training/main"

# 논문1 측정 (코퍼스 품질). 도메인 평균.
PAPER1 = {
    "LLM judge 4종":   {"exp0": 3.805, "exp1_1": 3.895, "exp1_2": 3.840, "exp2": 3.990},
    "프론티어 judge 2종": {"exp0": 3.405, "exp1_1": 3.500, "exp1_2": 3.445, "exp2": 3.645},
    "전문가 6인":       {"exp0": 3.359, "exp1_1": 3.464, "exp1_2": 3.459, "exp2": 3.579},
}


def rag():
    cmap = json.load(open(f"{R1}/judge_inputs/correctness_map.json", encoding="utf-8"))
    js = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(f"{R1}/scores/*.json"))]
    cell = defaultdict(dict)
    for jid, m in cmap.items():
        vs = [SC.get(j.get(jid)) for j in js]
        vs = [v for v in vs if v is not None]
        if vs:
            cell[(m["model"], m["domain"], m["qid"])][m["cond"]] = sum(vs) / len(vs)
    out = {}
    for c in CONDS:
        v = [d[c] for d in cell.values() if c in d]
        out[c] = st.mean(v)
    return out


def novelty():
    cmap = json.load(open(f"{R3}/judge_inputs/novelty_map.json", encoding="utf-8"))
    g = json.load(open(f"{R3}/scores/nov_gpt.json", encoding="utf-8"))
    s = json.load(open(f"{R3}/scores/nov_sonnet.json", encoding="utf-8"))
    eff, cor, noise = defaultdict(list), defaultdict(list), defaultdict(list)
    for jid, m in cmap.items():
        a, b = g.get(jid), s.get(jid)
        if not a or not b:
            continue
        gr = bool(a.get("grounded")) and bool(b.get("grounded"))
        co = bool(a.get("correct")) and bool(b.get("correct"))
        eff[m["cond"]].append(1.0 if (bool(a.get("novel")) and gr and co) else 0.0)
        cor[m["cond"]].append(1.0 if co else 0.0)
        noise[m["cond"]].append(0.0 if gr else 1.0)
    return ({c: st.mean(eff[c]) * 100 for c in CONDS},
            {c: st.mean(cor[c]) * 100 for c in CONDS},
            {c: st.mean(noise[c]) * 100 for c in CONDS})


def corpus_size():
    out = {}
    for c in CONDS:
        n = 0
        for dom in ("medical", "finance"):
            n += sum(1 for l in open(f"{CORPUS}/{c}_{dom}.jsonl", encoding="utf-8") if l.strip())
        out[c] = n
    return out


def ans_len():
    out = {}
    for c in CONDS:
        L = []
        for dom in ("medical", "finance"):
            for line in open(f"{CORPUS}/{c}_{dom}.jsonl", encoding="utf-8"):
                if line.strip():
                    m = {x["role"]: x["content"] for x in json.loads(line)["messages"]}
                    L.append(len(m.get("assistant", "")))
        out[c] = st.mean(L)
    return out


def row(label, d, fmt="{:.3f}", better=None):
    vals = [d[c] for c in CONDS]
    best = max(vals) if better == "high" else (min(vals) if better == "low" else None)
    cells = []
    for c in CONDS:
        s = fmt.format(d[c])
        cells.append(f"**{s}**" if best is not None and d[c] == best else s)
    delta = d["exp2"] - d["exp0"]
    return f"| {label} | " + " | ".join(cells) + f" | {delta:+.3f} |"


def main() -> None:
    r, (nov, cor, noise) = rag(), novelty()
    sz, al = corpus_size(), ans_len()
    print("| 지표 | exp0 | exp1_1 | exp1_2 | exp2 | Δ(exp2−exp0) |")
    print("|---|---:|---:|---:|---:|---:|")
    print("| **코퍼스 품질 (논문1)** | | | | | |")
    for k, v in PAPER1.items():
        print(row(k, v, "{:.3f}", "high"))
    print("| **코퍼스 구성** | | | | | |")
    print(row("QA 개수", sz, "{:.0f}"))
    print(row("답변 길이(자)", al, "{:.0f}"))
    print(row("잡음 비율(%)", noise, "{:.2f}", "low"))
    print("| **downstream (논문1-1)** | | | | | |")
    print(row("RAG 정확도", r, "{:.3f}", "high"))
    print(row("지식 신규성(%)", nov, "{:.1f}", "high"))
    print(row("사실 정확성(%)", cor, "{:.1f}", "high"))


if __name__ == "__main__":
    main()
