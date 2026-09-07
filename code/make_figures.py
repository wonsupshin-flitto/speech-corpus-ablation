#!/usr/bin/env python3
"""논문1-1 그림 생성 (논문 draft/v0.1/figs/).

Figure 1  핵심 통찰. 코퍼스 품질은 계단, downstream 은 평평하다.
Figure 3  검색 결과. 코퍼스 자체는 크게 이롭고 조건 간에는 차이가 없다.

Figure 2(파이프라인 도해)는 TikZ 로 본문에 직접 그린다. 벡터 텍스트가
본문 폰트와 맞아야 하고, 상자·화살표는 matplotlib 보다 TikZ 가 깔끔하다.

수치 출처는 report/ 아래 집계 결과다. 이 파일의 상수를 고칠 때는 반드시
그쪽을 다시 확인한다.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

import os, sys
# 대상 버전을 인자로 받는다. 기본은 최신 작업본.
VER = sys.argv[1] if len(sys.argv) > 1 else "v0.8"
OUT = f"paper_works/1-1.speech_corpus_rag/manuscript/draft/{VER}/figs"
os.makedirs(OUT, exist_ok=True)
CONDS = ["Exp 0", "Exp 1-1", "Exp 1-2", "Exp 2"]
X = [0, 1, 2, 3]

# ── 수치 (report/ 집계분) ──────────────────────────────────────────────
QUALITY = {                       # 1-5 척도. 채점원 둘.
    "4 LLM judges":    [3.805, 3.895, 3.840, 3.990],
    "6 human experts": [3.359, 3.464, 3.459, 3.579],
}   # 프론티어 둘은 4 LLM 의 부분집합이라 별개 채점원이 아니다 (부록의 강건성 확인)
RETRIEVAL = [0.767, 0.706, 0.785, 0.748]          # 정확도 (중립 115문항, 2-judge)
RETRIEVAL_CI = [0.018, 0.019, 0.017, 0.019]        # 셀 단위 95% 반폭 근사
# 교차 설계(260906): 두 프론티어 모델이 서로를 채점. 엄격 기준(둘 다 못 맞힘).
NOVELTY = [35.6, 36.1, 36.9, 36.5]                 # %
NOVELTY_CI = [1.8, 1.8, 1.8, 1.9]


def fig1():
    """계단 vs 평평 — 이 논문의 그림."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4))

    # 왼쪽: 코퍼스 품질. 각 측정원을 자기 Exp 0 기준으로 이동시켜 함께 본다.
    ax = axes[0]
    marks = ["o", "^"]
    for (name, v), m in zip(QUALITY.items(), marks):
        ax.plot(X, [x - v[0] for x in v], marker=m, ms=5, lw=1.5, label=name)
    ax.axhline(0, color="0.7", lw=0.6, zorder=0)
    ax.set_xticks(X); ax.set_xticklabels(CONDS)
    ax.set_ylabel("change from Exp 0\n(1--5 scale)")
    ax.set_title("Corpus quality", fontsize=10, pad=4)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-0.06, 0.30)

    # 오른쪽: downstream. 두 지표를 각자 Exp 0 기준 % 변화로 겹쳐 본다.
    ax = axes[1]
    r0, n0 = RETRIEVAL[0], NOVELTY[0]
    ax.errorbar(X, [(x - r0) / r0 * 100 for x in RETRIEVAL],
                yerr=[c / r0 * 100 for c in RETRIEVAL_CI],
                marker="o", ms=5, lw=1.5, capsize=3, label="Retrieval accuracy")
    ax.errorbar(X, [(x - n0) / n0 * 100 for x in NOVELTY],
                yerr=[c / n0 * 100 for c in NOVELTY_CI],
                marker="s", ms=5, lw=1.5, capsize=3, label="Knowledge novelty")
    ax.axhline(0, color="0.7", lw=0.6, zorder=0)
    ax.set_xticks(X); ax.set_xticklabels(CONDS)
    ax.set_ylabel("change from Exp 0 (\\%)")
    ax.set_title("Retrieval and novelty", fontsize=10, pad=4)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(-13, 13)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=9)
    fig.tight_layout(pad=0.5)
    fig.savefig(f"{OUT}/fig1_dissociation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_dissociation.pdf")


def fig3():
    """검색 정확도 전 조건. 코퍼스의 크기 대비 조건 차이의 작음을 한눈에."""
    # whisper 두 조건은 전문가 문항으로만 생성해 중립 33문항에서 재므로 여기 넣지
    # 않는다. 같은 그림에 두면 exp0(115문항, 0.767)과 나란히 놓여 실제로 없는
    # 0.055 격차로 읽힌다. 같은 33문항에서 exp0 은 0.721, whisper 는 0.712 다.
    labels = ["No corpus", "Mismatched", "Exp 0", "Exp 1-1", "Exp 1-2", "Exp 2"]
    vals = [0.231, 0.100, 0.767, 0.706, 0.785, 0.748]
    colors = ["0.55", "0.55", "0.25", "0.25", "0.25", "0.25"]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    y = range(len(labels))[::-1]
    ax.barh(list(y), vals, color=colors, height=0.62)
    for yy, v in zip(y, vals):
        ax.text(v + 0.012, yy, f"{v:.3f}", va="center", fontsize=6.8)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("retrieval accuracy", fontsize=8)
    ax.set_xlim(0, 0.92)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    # 네 조건이 모여 있는 띠를 표시
    ax.axvspan(0.702, 0.768, color="0.25", alpha=0.07, zorder=0)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT}/fig3_retrieval.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_retrieval.pdf")


if __name__ == "__main__":
    fig1()
    fig3()
