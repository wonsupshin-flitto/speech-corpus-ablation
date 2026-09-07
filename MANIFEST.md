# Manifest

| File | Items | Size |
|---|---:|---:|
| `eval_questions/questions_sample.jsonl` | 39 | 24 KB |
| `sample_qa/exp0_finance.jsonl` | 25 | 16 KB |
| `sample_qa/exp0_medical.jsonl` | 25 | 14 KB |
| `sample_qa/exp1_1_finance.jsonl` | 25 | 15 KB |
| `sample_qa/exp1_1_medical.jsonl` | 25 | 13 KB |
| `sample_qa/exp1_2_finance.jsonl` | 25 | 18 KB |
| `sample_qa/exp1_2_medical.jsonl` | 25 | 14 KB |
| `sample_qa/exp2_finance.jsonl` | 25 | 15 KB |
| `sample_qa/exp2_medical.jsonl` | 25 | 16 KB |
| `verdicts/novelty_map.json` | 10,695 | 973 KB |
| `verdicts/novelty_verdicts.jsonl` | 10,695 | 726 KB |
| `verdicts/recall_at3.json` | 12 | 103 KB |
| `verdicts/retrieval_expanded_gpt.json` | 4,320 | 95 KB |
| `verdicts/retrieval_expanded_map.json` | 4,320 | 557 KB |
| `verdicts/retrieval_expanded_sonnet.json` | 4,320 | 96 KB |
| `verdicts/retrieval_expert_gpt.json` | 2,064 | 45 KB |
| `verdicts/retrieval_expert_map.json` | 2,064 | 216 KB |
| `verdicts/retrieval_expert_sonnet.json` | 2,064 | 46 KB |

`code/` holds 23 scripts:

```
aggregate_ablation.py  aggregate_novelty_cross.py  build_judge_ablation.py  build_novelty_judge.py  build_provenance.py  build_release_package.py  check_lexical_confound.py  closedbook_prompt.py  closedbook_run.py  confound_decisive.py  judge_novelty.py  judge_run_ablation.py  make_figures.py  master_table.py  merge_questions.py  mixed_effects.py  novelty_nonshared.py  primary_analysis.py  recall_at_k.py  select_public_release.py  statlib.py  tost.py  variance_decomp.py
```

## Internal layout the scripts expect

```
paper_works/1.expert_qa/data/training/main/{cond}_{domain}.jsonl   corpora
paper_works/1.expert_qa/data/raw/{domain}/{si_seq}/                per-pair metadata
paper_works/1-1.speech_corpus_rag/report/{date}_{name}/            per-experiment outputs
```
