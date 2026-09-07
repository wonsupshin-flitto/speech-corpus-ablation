#!/usr/bin/env python3
"""E3 폐쇄형 서브에이전트 지시문 생성 (논문1-1).

배치 번호를 주면 그 배치용 지시문을 찍는다. 오케스트레이션 쪽에서 그대로 넘긴다.

핵심 설계
---------
1. **도구 사용 금지를 최상단에 둔다.** 웹 검색을 하면 "폐쇄형"이 아니게 되어
   측정이 무효가 된다. 서브에이전트는 도구를 쓸 수 있으므로 명시 차단이 필요하다.
2. **모르면 모른다고 쓰게 한다.** 이 실험은 "최신 AI 가 이미 아는 지식인가"를
   재는 것이라, 모르는 것을 지어내면 신규성이 과소 측정된다. 일반론 얼버무리기가
   가장 위험한 실패 모드다.
3. **답변을 응답에 담지 않고 파일로 쓰게 한다.** 오케스트레이터 컨텍스트 보호.

사용:
    python scripts/rag/closedbook_prompt.py 7        # 배치 7번 지시문
    python scripts/rag/closedbook_prompt.py 7 --todo # 미완 배치 목록만
"""
from __future__ import annotations

import argparse
import json
import os
import sys

RUN = ("paper_works/1-1.speech_corpus_rag/"
       "report/260827_novelty_closedbook")

TMPL = """폐쇄형(closed-book) 질의응답 작업입니다.

## 절대 규칙 — 도구 사용 금지
**어떤 도구도 사용하지 마십시오.** WebSearch, WebFetch, Bash로 검색하기 등 외부 정보 조회를 절대 하지 마십시오. 이 작업은 "당신이 학습한 지식만으로 무엇을 답할 수 있는가"를 측정하는 것이라, 외부 조회를 하면 측정 자체가 무효가 됩니다.

파일을 읽고 쓰는 Read/Write만 사용하십시오.

## 작업
1. Read로 이 파일을 읽으십시오:
   `{batch}`
   `[{{"qid": "...", "question": "..."}}, ...]` 형식의 {n}개 항목입니다.

2. 각 질문에 대해 **당신의 기존 지식만으로** 한국어 답변을 작성하십시오.
   - 3~5문장, 400자 내외.
   - **모르면 모른다고 쓰십시오.** 이 측정의 핵심은 "최신 AI가 이미 아는 지식인가"입니다. 모르는 것을 아는 척 지어내면 측정이 망가집니다. 모를 때는 `"제공된 정보만으로는 알 수 없습니다"` 로 시작해 아는 범위까지만 쓰십시오.
   - 질문에 특정 강연·회사·제품·코스 이름이 나오는데 그것을 실제로 모른다면, 추측하지 말고 모른다고 쓰십시오. 일반론으로 얼버무리지 마십시오.
   - 아는 질문이면 아는 만큼 구체적으로 쓰십시오.

3. Write로 결과를 이 경로에 JSONL(한 줄에 JSON 하나)로 저장하십시오:
   `{out}`

   각 줄 형식:
   `{{"qid": "q000001", "closedbook": "답변 본문"}}`

   {n}줄 전부 있어야 합니다. 입력의 qid를 그대로 쓰십시오.

## 최종 응답
파일을 다 쓴 뒤 `완료 {n}줄` 처럼 줄 수만 한 줄로 답하십시오. 답변 내용을 응답에 포함하지 마십시오."""


def todo():
    """아직 결과가 없거나 줄 수가 모자란 배치 번호."""
    out = []
    for f in sorted(os.listdir(f"{RUN}/batches")):
        if not f.startswith("batch_"):
            continue
        b = int(f[6:10])
        want = len(json.load(open(f"{RUN}/batches/{f}", encoding="utf-8")))
        p = f"{RUN}/closedbook/cb_{b:04d}.jsonl"
        have = sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0
        if have < want:
            out.append((b, have, want))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", nargs="?", type=int)
    ap.add_argument("--todo", action="store_true")
    a = ap.parse_args()

    if a.todo:
        t = todo()
        print(f"미완 {len(t)}개")
        for b, h, w in t[:200]:
            print(f"  batch {b:04d}  {h}/{w}")
        return

    if a.batch is None:
        sys.exit("배치 번호 필요")
    bp = f"{RUN}/batches/batch_{a.batch:04d}.json"
    n = len(json.load(open(bp, encoding="utf-8")))
    print(TMPL.format(batch=bp, out=f"{RUN}/closedbook/cb_{a.batch:04d}.jsonl", n=n))


if __name__ == "__main__":
    main()
