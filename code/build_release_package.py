#!/usr/bin/env python3
"""공개 패키지 조립 — 논문1-1.

무엇을 담는가
-------------
`select_public_release.py` 가 고른 샘플 위에 **재현에 필요한 나머지**를 얹는다.

    release_package/
      sample_qa/          코퍼스 200 쌍 (4 조건 × 2 도메인 × 25)
      eval_questions/     평가 질문 39 개 + 참조 정답
      verdicts/           항목별 판정 + id 매핑 (본문 텍스트 없음)
      code/               본문 수치를 내는 분석 스크립트
      prompts/            판정·폐쇄형 프롬프트
      README.md  MANIFEST.md  LICENSE

무엇을 담지 않는가
------------------
**judge 입력 배치는 넣지 않는다.** 그 파일에는 질문·정답·모델 답변이 그대로 들어
있어 발표 원문이 노출된다. 대신 `{jid: 판정}` 과 `{jid: (모델, 도메인, 질문 id,
조건)}` 매핑만 넣는다. 둘을 합치면 본문의 모든 수치를 다시 낼 수 있고, 텍스트는
공개 샘플에 있는 39 개 질문 몫만 드러난다.

사용:
    python scripts/rag/build_release_package.py
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P11 = ROOT / "paper_works/1-1.speech_corpus_rag"
REP = P11 / "report"
OUT = P11 / "release_package"

# 본문 텍스트가 섞이면 안 되는 필드
TEXT_FIELDS = {"question", "gold", "answer", "text", "content", "source_sentences"}

CODE = [
    "primary_analysis.py", "tost.py", "statlib.py", "mixed_effects.py",
    "aggregate_ablation.py", "aggregate_novelty_cross.py", "recall_at_k.py",
    "check_lexical_confound.py", "confound_decisive.py", "variance_decomp.py",
    "master_table.py", "make_figures.py", "novelty_nonshared.py",
    "build_judge_ablation.py", "build_novelty_judge.py", "build_provenance.py",
    "merge_questions.py", "judge_run_ablation.py", "judge_novelty.py",
    "closedbook_prompt.py", "closedbook_run.py",
    "select_public_release.py", "build_release_package.py",
]


def no_text(obj, where: str) -> None:
    """판정·매핑 파일에 본문 텍스트가 섞였는지 본다."""
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            bad = TEXT_FIELDS & set(o)
            if bad:
                raise SystemExit(f"[{where}] 본문 텍스트 필드 {sorted(bad)}")
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o[:200])


def copy_json(src: Path, dst: Path, label: str) -> int:
    d = json.loads(src.read_text())
    no_text(d, label)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    return len(d)


def main() -> None:
    counts: dict[str, int] = {}

    # ── 판정 + 매핑 ──
    V = OUT / "verdicts"
    V.mkdir(parents=True, exist_ok=True)
    for src, dst, label in [
        (REP / "260827_rag_ablation/scores/corr_sonnet.json", "retrieval_expert_sonnet.json", "검색·전문가43·Sonnet"),
        (REP / "260827_rag_ablation/scores/corr_gpt.json", "retrieval_expert_gpt.json", "검색·전문가43·GPT"),
        (REP / "260827_rag_ablation/judge_inputs/correctness_map.json", "retrieval_expert_map.json", "검색·전문가43·매핑"),
        (REP / "260828_rag_expand/scores/corr_sonnet.json", "retrieval_expanded_sonnet.json", "검색·확장120·Sonnet"),
        (REP / "260828_rag_expand/scores/corr_gpt.json", "retrieval_expanded_gpt.json", "검색·확장120·GPT"),
        (REP / "260828_rag_expand/judge_inputs/correctness_map.json", "retrieval_expanded_map.json", "검색·확장120·매핑"),
        (REP / "260906_novelty_cross/judge_inputs/novelty_map.json", "novelty_map.json", "신규성·매핑"),
        (REP / "260828_recall/recall3.json", "recall_at3.json", "세션 재현율"),
    ]:
        counts[label] = copy_json(src, V / dst, label)

    # 신규성 판정은 조각 파일이라 하나로 합친다
    rows = []
    for sub in ("scored", "scored_small"):   # 재개 과정에서 조각이 두 폴더로 나뉘었다
        for f in sorted((REP / "260906_novelty_cross" / sub).glob("*.json")):
            rows += json.loads(f.read_text())
    no_text(rows, "신규성·판정")
    seen, uniq = set(), []
    for r in rows:
        if r["jid"] not in seen:
            seen.add(r["jid"])
            uniq.append(r)
    uniq.sort(key=lambda r: r["jid"])
    with (V / "novelty_verdicts.jsonl").open("w") as fh:
        for r in uniq:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts["신규성·판정"] = len(uniq)

    # ── 코드 ──
    C = OUT / "code"
    C.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in CODE:
        src = ROOT / "scripts/rag" / name
        if src.exists():
            shutil.copy2(src, C / name)
            copied += 1
    counts["분석 스크립트"] = copied

    # ── 프롬프트 ──
    PR = OUT / "prompts"
    PR.mkdir(parents=True, exist_ok=True)
    grabs = [
        ("retrieval_correctness.md", "scripts/rag/judge_run_ablation.py", r"CORR_SYS = \((.*?)\)\n", "검색 답변 채점 (correct / partial / incorrect)"),
        ("novelty.md", "scripts/rag/judge_novelty.py", r"SYS = \((.*?)\)\n", "신규성 3 축 판정 (novel / grounded / correct)"),
        ("closed_book.md", "scripts/rag/closedbook_prompt.py", r'TMPL = """(.*?)"""', "폐쇄형 질의 (문서 없이 답하게 한다)"),
    ]
    for out_name, path, pat, title in grabs:
        src = (ROOT / path).read_text()
        m = re.search(pat, src, re.S)
        body = m.group(1) if m else "(추출 실패)"
        body = re.sub(r'^\s*"|"\s*$', "", body.strip(), flags=re.M)
        (PR / out_name).write_text(f"# {title}\n\n출처: `code/{Path(path).name}`\n\n```\n{body.strip()}\n```\n")
    counts["프롬프트"] = len(grabs)

    # ── 집계 ──
    for k in ("sample_qa", "eval_questions"):
        n = sum(1 for f in (OUT / k).glob("*.jsonl") for _ in f.open())
        counts[k] = n

    print("공개 패키지 구성")
    for k, v in counts.items():
        print(f"  {k:24s} {v:>7,}")
    print(f"\n{OUT}")


if __name__ == "__main__":
    main()
