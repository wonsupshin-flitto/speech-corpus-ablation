# Speech corpora and retrieval: what preprocessing and refinement buy

Reproducibility package for the paper above.

The paper ablates two stages of a pipeline that turns conference speech into
question–answer pairs, and measures the four resulting corpora three ways:
corpus quality, retrieval utility, and knowledge novelty.

## What is here

| Folder | Contents |
|---|---|
| `sample_qa/` | **200 corpus pairs** — 25 per condition per domain, eight files |
| `eval_questions/` | **10 evaluation questions** with their reference answers |
| `verdicts/` | Per-item judge verdicts for every instrument, with the id maps that attach each verdict to a model, domain, question, and condition |
| `code/` | The scripts that produce every number and figure in the paper |
| `prompts/` | The judge and closed-book prompts, verbatim |

## The samples are filtered, not random

The source sessions are third-party conference talks. We release pairs and
questions that read as **domain knowledge on its own**, and drop anything that
points back at the talk or at a company. A pair is dropped when it contains

- contact-like strings (phone, national ID, account, email, URL),
- a reference to a person (`강연자`, `교수님`, a name plus `씨`/`님`),
- a reference to the occasion (`오늘`, `이번 발표`, `저희`, `이 슬라이드`),
- an organization or case identifier (`○○병원`, `주식회사`, `증례 3`), or
- a company's own business figures (`16,000가구`, `매출`, `고객사`, a company
  name such as `퀀텀 머신즈라는 기업`, an amount in won).

Among what survives we take the pairs whose wording is least tied to a
particular talk. `code/select_public_release.py` is the filter, and it runs
deterministically: the same input gives the same 200 pairs.

**The same filter runs on all four conditions, with no per-condition tuning.**
Pass rates are 72.3, 71.4, 71.0, and 69.9 percent for Exp 0, Exp 1-1, Exp 1-2,
and Exp 2. Nothing in the selection uses answer length, judge score, or any
other quality signal, so the sample does not favor one condition over another.

Because the sample is filtered, it is **not a random draw** from the corpora.
Use it to see what the pipeline produces and how the four conditions differ in
kind. Do not use it to estimate corpus-level statistics.

## Verdicts without text

`verdicts/` holds `{item id: verdict}` and `{item id: (model, domain, question
id, condition)}` as separate files. Joining them reproduces every contrast in
the paper. We do not release the judge input batches, because those carry the
questions, reference answers, and model answers in full, and with them the talk
content.

The separation is also how the judging ran. No judge saw which model or
condition produced an answer.

## Reproducing the numbers

```
python code/primary_analysis.py     # equivalence tests, all nine contrasts
python code/aggregate_novelty_cross.py
python code/recall_at_k.py
python code/make_figures.py
```

The scripts read the internal layout described in `MANIFEST.md`. Paths that
point at the full corpora will not resolve from this package alone; the
verdict files are what make the analysis reproducible here.

## Terms

Two licenses, because the two kinds of content carry different constraints.

| What | License | File |
|---|---|---|
| `code/` | MIT | `LICENSE-code` |
| `sample_qa/`, `eval_questions/`, `verdicts/` | CC BY-NC 4.0 | `LICENSE-data` |

`LICENSE` at the top level says which file governs what. We do not set a single
repository-wide license, because one label would misstate one half or the other.

The sessions come from service data where speakers consented to research and
non-commercial use, so the data carries a non-commercial term. That term is not
an open-source or open-data license, and we do not describe it as one. The
analysis code carries no such constraint and is MIT.
