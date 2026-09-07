# 신규성 3 축 판정 (novel / grounded / correct)

출처: `code/judge_novelty.py`

```
You grade Korean expert-domain QA items on three independent axes. 
For each item you get: `question`, `gold` (the answer we produced), and 
`closedbook` (what a frontier model answered from its own knowledge, no documents).\n
Grade each axis independently:\n
  novel:    true if `closedbook` FAILS to convey the core content of `gold` 
(different entity/value/concept, or vague/evasive). false if it conveys the same 
core content (paraphrase is fine). Do NOT reward longer answers.\n
  correct:  true if `gold` is factually sound and internally consistent for the 
question. false if self-contradictory, nonsensical, or clearly wrong.\n
  grounded: true if `gold` reads as a substantive domain answer to `question`. 
false if it is filler, transcription noise, off-topic chatter, or does not answer.\n
Output ONLY a JSON array 
[{\"jid\":\"..\",\"novel\":true|false,\"correct\":true|false,\"grounded\":true|false}], 
no prose.
```
