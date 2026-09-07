#!/usr/bin/env python3
"""공개용 샘플 선정 — 코퍼스 200 쌍 + 평가 질문 40 개 (논문1-1).

무엇을 고르는가
---------------
공개 대상은 두 가지다.

1. **코퍼스 샘플 200 쌍** — 4 조건 × 2 도메인 × 25 쌍.
   조건마다 같은 수를 내보내므로 조건 간 비교가 가능하다.
2. **평가 질문 40 개** — 163 개 중 도메인·출처(전문가/확장) 비율을 맞춰 뽑는다.

무엇을 거르는가
---------------
발표 원문은 제3자 저작물이고 화자가 특정될 수 있다. 그래서 **지식 진술만 남기고
발표 상황을 가리키는 것은 전부 뺀다.** 하드 필터는 네 갈래다.

- 개인정보 형태 (전화·주민·계좌·이메일·URL)
- 사람 지시 (강연자·교수님·홍길동 씨 …)
- 발표 상황 지시 (오늘·이번 발표·저희·제가 …)
- 기관·사례 식별 (○○병원·주식회사·증례 3 …)

살아남은 것 중에서는 **일반성 점수**가 낮은(= 더 일반 지식에 가까운) 순으로 고르고,
동점은 질문 텍스트 해시로 가른다. 해시 정렬이라 실행할 때마다 같은 결과가 나온다.

조건별 차등을 두지 않는다 (중요)
--------------------------------
논문1 의 `select_public_samples.py` 는 조건마다 답변 길이 목표를 달리 줘서
(`exp0` 280 자, `exp2` 430 자) 전체 파이프라인이 좋아 보이도록 골랐다. 본 논문의
논지는 "품질 점수는 오르는데 하류로 가지 않는다" 이므로, 그런 표본을 내면 표본과
결론이 어긋난다. **여기서는 네 조건에 완전히 같은 필터·같은 정렬을 적용하고,
길이·품질을 선정에 쓰지 않는다.** 조건별 통과율을 함께 찍어 편향이 없음을 보인다.

사용:
    python scripts/rag/select_public_release.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "paper_works/1.expert_qa"
P11 = ROOT / "paper_works/1-1.speech_corpus_rag"
QFILE = P11 / "report/260828_question_expansion/endpoint_qa_163.jsonl"

CONDS = ["exp0", "exp1_1", "exp1_2", "exp2"]
DOMAINS = ["medical", "finance"]
PER_CELL = 25
N_QUESTIONS = 10

# ── 하드 필터 ────────────────────────────────────────────────────────────────
PII = [
    re.compile(r"\d{2,3}-\d{3,4}-\d{4}"),            # 전화
    re.compile(r"\b\d{6}-?\d{7}\b"),                 # 주민등록
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),         # 이메일
    re.compile(r"\d{3,4}-\d{2}-\d{6}"),              # 계좌
    re.compile(r"https?://|www\."),                  # URL
]

PERSON = [
    re.compile(r"강연자|발표자|연자|화자|저자님|교수님|원장님|선생님|박사님|대표님|소장님|위원장님"),
    re.compile(r"[가-힣]{1,3}\s?씨(가|는|의|를|와|과|도|에게|께)"),
    re.compile(r"[가-힣]{1,3}\s?님(이|은|의|을|께|과|도)"),
]

# 발표 상황을 가리키는 표현. 지식 진술이 아니라 그 자리의 발화다.
DEIXIS = [
    re.compile(r"오늘|금일|이번\s?(발표|강연|세션|시간)|본\s?(발표|강연|강의)"),
    re.compile(r"저희|제가|제\s|우리\s?(병원|회사|팀|기관|센터|은행|학회)"),
    re.compile(r"앞서\s|방금|아까|뒤에서\s?다루|말씀드린|말씀드리|발표에서"),
    re.compile(r"이\s?(슬라이드|그림|표|자료|화면)|다음\s?(슬라이드|장)"),
    re.compile(r"질문\s?주신|청중|참석자"),
]

# 기관·사례 식별. 앞에 붙는 말이 일반어면 통과시킨다.
ORG_SUFFIX = r"(병원|의료원|대학교|대학병원|증권|은행|생명|화재|캐피탈|자산운용|투자자문|제약|바이오|그룹|공사|공단)"
GENERIC_PREFIX = {
    "대학", "종합", "상급종합", "지역", "국내", "해외", "일반", "민간", "공공",
    "우리", "저희", "각", "본", "타", "해당", "일부", "다수", "여러", "모든",
    "국공립", "사립", "요양", "전문", "1차", "2차", "3차", "상급",
}
ORG = re.compile(r"([가-힣]{2,6})" + ORG_SUFFIX)
CASE = [
    re.compile(r"증례|케이스\s?\d|사례\s?\d|[Cc]ase\s?\d"),
    re.compile(r"(이|해당|본|그)\s?환자"),
    re.compile(r"환자분"),
    re.compile(r"주식회사|㈜|\(주\)"),
]

# 특정 기업의 영업 정보. 브랜드명은 규칙으로 못 잡으므로 **수치·자기지시** 쪽을 막는다.
# 예: "마크 노바는 B2G 서비스로 16,000가구에 제품과 서비스를 제공" 은 회사명 필터를
# 빠져나가지만 `[숫자]가구` 에서 걸린다.
BIZ = [
    re.compile(r"자사|당사|자체\s?개발|고객사|도입\s?기업|레퍼런스\s?고객"),
    re.compile(r"매출|영업이익|수주|점유율|가입자|밸류에이션|투자\s?유치|계약\s?규모"),
    re.compile(r"[\d,]+\s*(가구|개사|개\s?기업|명의\s?(고객|가입자)|건의?\s?계약)"),
    re.compile(r"[\d,천만억조]+\s*원"),                          # 금액 일반 (5천 원·만 원 포함)
    re.compile(r"(라는|이라는)\s*(기업|회사|스타트업|업체|법인|곳)"),   # "퀀텀 머신즈라는 기업"
    re.compile(r"[가-힣]{2,}(뱅크|카드|페이|텔레콤|전자|머신즈|랩스|웍스|테크놀로지|모빌리티)"),
    re.compile(r"수수료|요금제|구독료|판매가|단가"),
]

# ── 일반성 점수 (낮을수록 일반 지식에 가깝다) ───────────────────────────────
SOFT = [
    (re.compile(r"환자"), 1),
    (re.compile(r"20\d\d년"), 1),      # 발표 시점에 묶인 서술일 수 있다
    (re.compile(r"[A-Z][a-zA-Z]{2,}"), 1),   # 라틴 고유명사 후보
    (re.compile(r"논문|연구에서|보고에\s?따르"), 1),
    (re.compile(r"솔루션|플랫폼|출시|상용화|도입\s?사례"), 3),   # 특정 제품 서술로 기운다
]
QTYPE_BONUS = {"definition": -2, "factoid": -1, "factual": -1, "explanatory": -1}


def blocked(text: str) -> str | None:
    """차단 사유를 돌려준다. 통과면 None."""
    for p in PII:
        if p.search(text):
            return "pii"
    for p in PERSON:
        if p.search(text):
            return "person"
    for p in DEIXIS:
        if p.search(text):
            return "deixis"
    for p in CASE:
        if p.search(text):
            return "case"
    for p in BIZ:
        if p.search(text):
            return "biz"
    for m in ORG.finditer(text):
        if m.group(1) not in GENERIC_PREFIX:
            return "org"
    return None


def genericity(text: str, qtype: str = "") -> int:
    s = sum(w * len(p.findall(text)) for p, w in SOFT)
    return s + QTYPE_BONUS.get((qtype or "").lower(), 0)


def h(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


# ── 코퍼스 로딩 ──────────────────────────────────────────────────────────────
def load_raw_meta(sessions: set[str]) -> dict[str, dict]:
    """raw 레이아웃에서 질문 -> 메타데이터. 질문 텍스트는 raw 와 training 사이에서 보존된다."""
    meta: dict[str, dict] = {}
    for dom in DOMAINS:
        for sdir in (P1 / "data/raw" / dom).iterdir():
            if not sdir.is_dir() or sdir.name not in sessions:
                continue
            for f in sdir.glob("*.jsonl"):
                if "needs_review" in f.name:
                    continue
                for line in f.open():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    q = r.get("question")
                    if q and q not in meta:
                        meta[q] = {
                            "language": r.get("language", "ko"),
                            "domain": dom,
                            "sub_domain": r.get("sub_domain", ""),
                            "difficulty": r.get("difficulty", ""),
                            "question_type": r.get("question_type", ""),
                        }
    return meta


def load_corpus(cond: str, dom: str) -> list[tuple[str, str]]:
    """평가에 실제로 쓰인 학습용 JSONL 에서 (질문, 답) 을 읽는다."""
    f = P1 / f"data/training/main/{cond}_{dom}.jsonl"
    out = []
    for line in f.open():
        msgs = json.loads(line)["messages"]
        q = next(m["content"] for m in msgs if m["role"] == "user")
        a = next(m["content"] for m in msgs if m["role"] == "assistant")
        out.append((q, a))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(P11 / "release_package"))
    args = ap.parse_args()
    out = Path(args.out)

    questions = [json.loads(l) for l in QFILE.open()]
    sessions = {q["si_seq"] for q in questions}
    print(f"세션 {len(sessions)} 개 (질문 파일 기준)")

    meta = load_raw_meta(sessions)
    print(f"raw 메타데이터 {len(meta):,} 개 질문")

    # ── 코퍼스 샘플 ──
    (out / "sample_qa").mkdir(parents=True, exist_ok=True)
    stats: dict[str, Counter] = defaultdict(Counter)
    total_sel = 0
    for cond in CONDS:
        for dom in DOMAINS:
            pairs = load_corpus(cond, dom)
            kept = []
            for q, a in pairs:
                reason = blocked(q + "\n" + a)
                if reason:
                    stats[cond][reason] += 1
                    continue
                m = meta.get(q, {})
                stats[cond]["pass"] += 1
                kept.append({
                    "question": q,
                    "answer": a,
                    "language": m.get("language", "ko"),
                    "domain": dom,
                    "sub_domain": m.get("sub_domain", ""),
                    "difficulty": m.get("difficulty", ""),
                    "question_type": m.get("question_type", ""),
                })
            kept.sort(key=lambda r: (genericity(r["question"] + r["answer"],
                                                r["question_type"]), h(r["question"])))
            # 하위도메인이 한쪽으로 몰리지 않게 라운드로빈으로 채운다
            by_sub: dict[str, list] = defaultdict(list)
            for r in kept:
                by_sub[r["sub_domain"] or "-"].append(r)
            sel, i = [], 0
            while len(sel) < PER_CELL and any(by_sub.values()):
                for k in sorted(by_sub):
                    if by_sub[k] and len(sel) < PER_CELL:
                        sel.append(by_sub[k].pop(0))
                i += 1
                if i > 200:
                    break
            sel.sort(key=lambda r: h(r["question"]))
            fp = out / "sample_qa" / f"{cond}_{dom}.jsonl"
            with fp.open("w") as fh:
                for r in sel:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            total_sel += len(sel)
            print(f"  {cond:7s} {dom:8s}  전체 {len(pairs):5,}  통과 {len(kept):5,}"
                  f"  ({len(kept)/len(pairs):5.1%})  선정 {len(sel)}")

    print(f"\n코퍼스 샘플 {total_sel} 쌍")
    print("\n조건별 차단 사유")
    print(f"{'조건':8s} " + " ".join(f"{k:>7s}" for k in ("pass", "pii", "person", "deixis", "case", "org", "biz")))
    for cond in CONDS:
        c = stats[cond]
        tot = sum(c.values())
        print(f"{cond:8s} " + " ".join(f"{c[k]:7,}" for k in ("pass", "pii", "person", "deixis", "case", "org", "biz"))
              + f"   통과율 {c['pass']/tot:5.1%}")

    # ── 평가 질문 샘플 ──
    (out / "eval_questions").mkdir(parents=True, exist_ok=True)
    qkept, qblocked = [], Counter()
    for q in questions:
        reason = blocked(q["question"] + "\n" + q["gold"])
        if reason:
            qblocked[reason] += 1
            continue
        qkept.append(q)
    # 도메인 × 출처 비율을 원본대로 유지
    want = Counter()
    for q in questions:
        want[(q["domain"], q["origin"])] += 1
    scale = N_QUESTIONS / len(questions)
    picked = []
    for key in sorted(want):
        pool = [q for q in qkept if (q["domain"], q["origin"]) == key]
        pool.sort(key=lambda r: (genericity(r["question"] + r["gold"], r.get("qtype", "")),
                                 h(r["question"])))
        picked += pool[:max(1, round(want[key] * scale))]
    # 반올림 때문에 목표에 못 미치면 큰 층부터 한 개씩 채운다
    chosen = {q["id"] for q in picked}
    if len(picked) < N_QUESTIONS:
        rest = [q for q in qkept if q["id"] not in chosen]
        rest.sort(key=lambda r: (genericity(r["question"] + r["gold"], r.get("qtype", "")),
                                 h(r["question"])))
        picked += rest[:N_QUESTIONS - len(picked)]
    picked.sort(key=lambda r: r["id"])
    with (out / "eval_questions" / "questions_sample.jsonl").open("w") as fh:
        for q in picked:
            fh.write(json.dumps({k: q[k] for k in
                                 ("id", "domain", "subdomain", "qtype", "origin", "question", "gold")},
                                ensure_ascii=False) + "\n")
    print(f"\n평가 질문 {len(questions)} 중 통과 {len(qkept)}, 선정 {len(picked)}")
    print("  차단:", dict(qblocked))
    print("  구성:", dict(Counter((q["domain"], q["origin"]) for q in picked)))


if __name__ == "__main__":
    main()
