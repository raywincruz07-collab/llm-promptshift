# Contamination probe (generated tables, no interpretation)

Limited behavioural probe for memorisation; not proof of exposure or of its absence.

| Model | Paired n | ROUGE-L general | ROUGE-L guided | Guided − general [95% CI] | Sign test p | Wilcoxon p | Exact matches (gen/guided) |
|---|---:|---:|---:|---|---:|---:|---|
| Gemma | 200 | 0.103 | 0.104 | +0.001 [-0.014, +0.017] | 0.7397 | 0.7926 | 0/0 |
| Qwen | 200 | 0.089 | 0.100 | +0.011 [-0.003, +0.025] | 0.7065 | 0.2333 | 0/0 |
| Llama | 200 | 0.094 | 0.117 | +0.022 [+0.010, +0.036] | 0.0035 | 0.0026 | 0/0 |

4 completions with ROUGE-L ≥ 0.7 listed for manual review in probe_review_CONTAINS_TWEET_TEXT.csv (gitignored).
