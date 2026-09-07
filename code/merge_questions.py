#!/usr/bin/env python3
"""검색 평가 질문 병합 — 43 -> 163 (논문1-1).

왜 늘리는가
-----------
두 리뷰가 같은 지점을 최우선으로 지적했다. 중심 주장이 "정제해도 하류 성능이 오르지
않는다"는 null 인데, 43문항으로는 그 null 이 **차이가 없어서인지 못 재서인지** 구분되지
않는다. 등가성 검정(TOST)을 하려면 검정력이 필요하고, 검정력은 문항 수에서 나온다.

새 문항은 **전사문만 보고** 작성했다. 코퍼스(어느 조건의 QA)를 보고 만들면 그 조건에
유리한 질문이 되어 비교가 깨진다. 작성자에게 다른 파일 열람과 검색을 금지했다.

사용:
    python scripts/rag/merge_questions.py \
      --base <원본 43 jsonl> --add-dir <out 디렉토리> --out <병합 jsonl>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

REQ = ("id", "domain", "si_seq", "subdomain", "qtype", "question", "gold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--add-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = [json.loads(l) for l in open(args.base, encoding="utf-8") if l.strip()]
    for r in base:
        r["origin"] = "base43"

    add = []
    for f in sorted(glob.glob(os.path.join(args.add_dir, "*.json"))):
        for it in json.load(open(f, encoding="utf-8")):
            it["si_seq"] = str(it["si_seq"])
            it["origin"] = "expand120"
            add.append(it)

    # id 충돌은 조용히 덮어써지면 문항이 사라진다. 명시적으로 막는다.
    ids = Counter(r["id"] for r in base + add)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise SystemExit(f"❌ id 중복 {len(dup)}건: {dup[:5]}")

    # 원본은 factual, 신규는 factoid 로 같은 유형을 달리 적었다. 두 범주로
    # 갈라지면 유형 층화가 깨진다. 표준 용어로 통일한다.
    merged = base + add
    for r in merged:
        if r["qtype"] == "factual":
            r["qtype"] = "factoid"
    for r in merged:
        miss = [k for k in REQ if k not in r or not str(r[k]).strip()]
        if miss:
            raise SystemExit(f"❌ {r.get('id')}: 필드 누락 {miss}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"병합 {len(base)} + {len(add)} = {len(merged)}  ->  {args.out}")
    print(f"  도메인 {dict(Counter(r['domain'] for r in merged))}")
    print(f"  유형   {dict(Counter(r['qtype'] for r in merged))}")
    bs = {r["si_seq"] for r in base}
    as_ = {r["si_seq"] for r in add}
    print(f"  세션   원본 {len(bs)} / 신규 {len(as_)} / 공유 {len(bs & as_)}")
    if bs - as_:
        print(f"  ⚠️ 신규에 없는 원본 세션 {len(bs - as_)}개: {sorted(bs - as_)[:5]}")
    per = Counter(r["si_seq"] for r in merged)
    print(f"  세션당 문항 최소 {min(per.values())} 최대 {max(per.values())}")


if __name__ == "__main__":
    main()
