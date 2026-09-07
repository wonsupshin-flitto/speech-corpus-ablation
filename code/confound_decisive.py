#!/usr/bin/env python3
"""어휘 교란이 E1 결과를 **설명하는가** — 결정적 검정 (논문1-1).

`check_lexical_confound.py` 는 교란의 **존재**를 확인했다(exp0 의 어휘 중첩이 높다).
그러나 존재한다고 해서 결과를 설명하는 것은 아니다.

조건별로 보면 순서가 어긋난다.

    exp1_1   정답률 −0.045   어휘 −0.0023 (유의하지 않음)
    exp1_2   정답률 +0.022   어휘 −0.0032 (유의)
    exp2     정답률 −0.030   어휘 −0.0055 (유의)

어휘가 정답률을 끌어내린 것이라면 어휘가 더 많이 떨어진 exp1_2 가 exp1_1 보다
나빠야 하는데 **반대**다. 그래서 질문 단위로 직접 상관을 본다.

    질문 q, 조건 c 에 대해
        Δ어휘(q,c)   = 어휘중첩(q,c) − 어휘중첩(q,exp0)
        Δ정답률(q,c) = 정답률(q,c)   − 정답률(q,exp0)
    이 둘의 상관이 0 이면 교란이 결과를 설명하지 못한다.

사용:
    python scripts/rag/confound_decisive.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statlib import paired_t

RUN = "paper_works/1-1.speech_corpus_rag/report/260827_rag_ablation"
CORPUS = "paper_works/1.expert_qa/data/training/main"
QF = "paper_works/1.expert_qa/report/260723_format_mismatch/release/endpoint_qa_43.jsonl"
SC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
CONDS = ["exp1_1", "exp1_2", "exp2", "whisper_medium_exp0", "whisper_medium_exp2"]
K, N = 3, 3


def grams(s):
    s = "".join(s.split())
    return {s[i:i + N] for i in range(max(0, len(s) - N + 1))}


def lex_scores():
    qs = [json.loads(l) for l in open(QF, encoding="utf-8") if l.strip()]
    out = defaultdict(dict)
    for dom in ("medical", "finance"):
        dqs = [q for q in qs if q["domain"] == dom]
        for cond in ["exp0"] + CONDS:
            corp = []
            with open(os.path.join(CORPUS, f"{cond}_{dom}.jsonl"), encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        m = {x["role"]: x["content"] for x in json.loads(line)["messages"]}
                        corp.append(grams(m.get("user", "") + " " + m.get("assistant", "")))
            for q in dqs:
                g = grams(q["question"])
                s = sorted((len(g & c) / len(g | c) for c in corp), reverse=True)[:K]
                out[cond][q["id"]] = sum(s) / len(s)
    return out


def judge_scores():
    cmap = json.load(open(f"{RUN}/judge_inputs/correctness_map.json", encoding="utf-8"))
    js = [json.load(open(f, encoding="utf-8"))
          for f in sorted(glob.glob(f"{RUN}/scores/*.json"))]
    cell = defaultdict(dict)
    for jid, m in cmap.items():
        vs = [SC.get(j.get(jid)) for j in js]
        vs = [v for v in vs if v is not None]
        if vs:
            cell[(m["model"], m["domain"], m["qid"])][m["cond"]] = sum(vs) / len(vs)
    return cell


def pearson(x, y):
    n = len(x)
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx and dy else 0.0


def main() -> None:
    lex, cell = lex_scores(), judge_scores()

    print("=== 어휘 변화가 정답률 변화를 설명하는가 (질문×모델 단위) ===")
    print(f"{'조건':24}{'n':>6}{'상관 r':>10}{'p(대략)':>10}")
    allx, ally = [], []
    for c in CONDS:
        X, Y = [], []
        for (mo, dom, qid), d in cell.items():
            if c in d and "exp0" in d and qid in lex[c]:
                X.append(lex[c][qid] - lex["exp0"][qid])
                Y.append(d[c] - d["exp0"])
        if len(X) < 5:
            continue
        r = pearson(X, Y)
        t = r * math.sqrt((len(X) - 2) / max(1e-12, 1 - r * r))
        print(f"{c:24}{len(X):>6}{r:>10.3f}{'유의' if abs(t) > 1.97 else '유의하지 않음':>12}")
        allx += X; ally += Y
    r = pearson(allx, ally)
    t = r * math.sqrt((len(allx) - 2) / max(1e-12, 1 - r * r))
    print(f"{'전체':24}{len(allx):>6}{r:>10.3f}{'유의' if abs(t) > 1.97 else '유의하지 않음':>12}")

    print("\n=== STT 민감도 상호작용 (조언 반영) ===")
    print("  (exp0−whisper_exp0) − (exp2−whisper_exp2) 를 셀 단위로 직접 검정")
    g = defaultdict(list)
    for (mo, dom, qid), d in cell.items():
        need = ("exp0", "whisper_medium_exp0", "exp2", "whisper_medium_exp2")
        if all(k in d for k in need):
            g[(mo, dom)].append((d["exp0"] - d["whisper_medium_exp0"])
                                - (d["exp2"] - d["whisper_medium_exp2"]))
    cells = [st.mean(v) for v in g.values()]
    m, _sd, tt, df, lo, hi, sig = paired_t(cells)
    print(f"  n={len(cells)}  {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  t={tt:+.2f}  "
          f"{'유의' if sig else '유의하지 않음'}")
    if not sig:
        print("  → 정제 전후로 STT 민감도가 달라졌다고 말할 수 없다. 표현을 낮춰야 한다.")


if __name__ == "__main__":
    main()
