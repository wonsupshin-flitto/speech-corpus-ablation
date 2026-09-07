#!/usr/bin/env python3
"""정제가 '검색 매칭'을 어렵게 한 것인가? — 어휘 중첩 교란 검사 (논문1-1).

왜 필요한가
-----------
E1 에서 `exp2−exp0 = −0.030`, `exp1_1−exp0 = −0.045` 가 나왔다. 이것은 두 가지로
읽힌다.

    (A) 정제가 코퍼스의 **내용 가치**를 높이지 못한다
    (B) 정제가 원 발화의 표현을 바꿔 **검색 매칭**을 어렵게 한다

두 해석은 같은 수치를 만든다. 질문이 원 전사문 어휘에 가깝게 쓰였다면, 다듬지 않은
exp0 에서 검색이 더 잘 맞아 (B)만으로도 음수가 나온다. 내용 품질과 무관하다.

가르는 법
---------
질문과 각 조건 코퍼스의 **어휘 중첩**을 잰다. exp0 가 체계적으로 높으면 (B)가 실재한다.

한국어는 공백 토큰화가 거칠어 **문자 3-gram Jaccard** 를 쓴다. 조사·어미 변화에
덜 민감하고 형태소 분석기 의존도 없앤다.

측정
    질문마다 각 조건 코퍼스에서 최대 중첩 상위 k 개를 찾아 평균낸다.
    실제 검색기(bge-m3)와 다르지만, **어휘 접근성의 방향**을 보는 데는 충분하다.
    검색기 자체 재현은 GPU 가 필요하고, 여기서 묻는 것은 방향뿐이다.

사용:
    python scripts/rag/check_lexical_confound.py
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from statlib import paired_t

CORPUS = "paper_works/1.expert_qa/data/training/main"
QF = "paper_works/1.expert_qa/report/260723_format_mismatch/release/endpoint_qa_43.jsonl"
CONDS = ["exp0", "exp1_1", "exp1_2", "exp2", "whisper_medium_exp0", "whisper_medium_exp2"]
K = 3
N = 3   # 문자 n-gram 크기


def grams(s: str):
    s = "".join(s.split())
    return {s[i:i + N] for i in range(max(0, len(s) - N + 1))}


def load(cond: str, dom: str):
    p = os.path.join(CORPUS, f"{cond}_{dom}.jsonl")
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = {x["role"]: x["content"] for x in json.loads(line)["messages"]}
            out.append(grams(m.get("user", "") + " " + m.get("assistant", "")))
    return out


def main() -> None:
    qs = [json.loads(l) for l in open(QF, encoding="utf-8") if l.strip()]
    print(f"질문 {len(qs)}건, 문자 {N}-gram Jaccard, 상위 {K} 평균\n")

    per_q = {c: {} for c in CONDS}     # cond -> qid -> score
    for dom in ("medical", "finance"):
        dqs = [q for q in qs if q["domain"] == dom]
        for cond in CONDS:
            corp = load(cond, dom)
            for q in dqs:
                g = grams(q["question"])
                sims = sorted((len(g & c) / len(g | c) for c in corp), reverse=True)[:K]
                per_q[cond][q["id"]] = sum(sims) / len(sims)

    print(f"{'조건':24}{'평균 중첩':>12}{'exp0 대비':>12}")
    base = st.mean(per_q["exp0"].values())
    for c in CONDS:
        m = st.mean(per_q[c].values())
        print(f"{c:24}{m:>12.4f}{m - base:>+12.4f}")

    print("\n=== 대응 검정 (질문 단위) ===")
    for a, b in (("exp2", "exp0"), ("exp1_1", "exp0"), ("exp1_2", "exp0"),
                 ("whisper_medium_exp2", "whisper_medium_exp0")):
        d = [per_q[a][k] - per_q[b][k] for k in per_q[a]]
        m, _sd, t, df, lo, hi, sig = paired_t(d)
        print(f"  {a}−{b:22} {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  t={t:+.2f}  "
              f"{'유의' if sig else '유의하지 않음'}")

    print("\n판정")
    d = [per_q["exp2"][k] - per_q["exp0"][k] for k in per_q["exp2"]]
    m = st.mean(d)
    if m < -0.002:
        print("  ⚠️ exp0 의 어휘 중첩이 높다 → 검색 매칭 교란(B)이 실재한다.")
        print("     E1 의 음수를 '내용 가치가 낮다'로 읽으면 안 된다.")
    elif m > 0.002:
        print("  exp2 의 어휘 중첩이 오히려 높다 → 교란(B)으로 음수를 설명할 수 없다.")
        print("     E1 의 음수는 내용 쪽 해석(A)에 무게가 실린다.")
    else:
        print("  조건 간 어휘 중첩이 평평하다 → 교란(B) 근거 없음.")
        print("     E1 의 음수는 내용 쪽 해석(A)으로 읽을 수 있다.")


if __name__ == "__main__":
    main()
