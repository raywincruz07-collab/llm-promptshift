# References and what we use them for

"Checked" = verified on 26 Nov 2026 against the publisher, ACL Anthology, AAAI or arXiv page. BibTeX: [references.bib](references.bib).

## Closest prior work (what is already known, so we do not claim it)

| Reference | What it shows | How we use it | Checked |
|---|---|---|---|
| Atreja et al. 2025, ICWSM 19(1):122–145 | Prompt design (incl. adding a definition) changes compliance, accuracy and label distribution in CSS annotation, incl. toxicity | The definition effect is **not** new; we test it on guaranteed-unseen data and report prevalence error | yes |
| Reiss 2023, arXiv 2304.11085 | Small wording changes and repeats of identical input change ChatGPT annotations | Motivates wording + exact-repeat conditions | yes |
| Errica et al. 2025, NAACL, pp. 1543–1558 | Sensitivity (label change across rephrasings, no gold needed) and consistency metrics | Our wording-disagreement measure follows this idea | yes |
| Sclar et al. 2024, ICLR; Mizrahi et al. 2024, TACL | Large sensitivity to prompt format; call for multi-prompt evaluation | Our three one-sentence wordings are a lower bound | standard |
| Casanova, Kocielnik & Alvarez 2026, ICML (arXiv 2606.00467) | Across 9 LLMs and 5 toxicity datasets, performance tracks definition alignment, not memorisation metrics; prompting corrects about a third of zero-shot errors | Closest work. They measure memorisation with metrics; we use data that cannot be memorised and study prompt sensitivity and prevalence | yes |

## Contamination

| Reference | What it shows | How we use it | Checked |
|---|---|---|---|
| Li & Flanigan 2024, AAAI | LLMs do better on datasets released before their training cut-off | Before/after comparison is an established idea; our H2 is a CSS-domain test of it | standard |
| Golchin & Surdeanu 2024, ICLR | Guided vs general completion detects instance-level contamination | Probe P1 | standard |
| Golchin & Surdeanu 2025, TACL (arXiv 2311.06233) | Data Contamination Quiz: pick the original among word-level paraphrases | Probe P5 (3 paraphrases + original, chance 25%) | yes |
| Shi et al. 2024, ICLR | Min-K% Prob membership score | Probe P4 | standard |
| Duan et al. 2024, COLM | Membership inference barely beats chance on LLMs; apparent success often reflects time shift | Why OLID-vs-2026 membership differences are "suggestive only" | standard |
| Zhang et al. 2026, ACL, pp. 36538–36555 ("Test of Time") | Post-cutoff performance decay is a sensitive, not clean, contamination signal | Why we use the new posts as a negative control, not as proof | yes |

## Data, labels and statistics

| Reference | Use | Checked |
|---|---|---|
| Zampieri et al. 2019, NAACL (OLID) | Definition, keywords, annotation procedure | standard |
| Barbieri et al. 2020, Findings of EMNLP (TweetEval) | Pinned OLID test split | standard |
| Pangakis et al. 2023, arXiv 2306.00176 | Validating LLM annotation against humans | standard |
| Krippendorff 2018 | Alpha for inter-annotator agreement | standard |
| Holm 1979 | Multiple-testing correction | standard |

## Model dates (why the new posts are unseen)

| Model (maKI name) | Released | Training data until | Source | Checked |
|---|---|---|---|---|
| `llama-3.3-70b` | 6 Dec 2024 | Dec 2023 | Meta model card | yes |
| `gemma4-26b` | 2 Apr 2026 | Jan 2025 | Gemma 4 model card / technical report | yes |
| `qwen3.6-36b` | Apr 2026 (Qwen3.6-Plus 2 Apr; open-weight 35B-A3B mid-April) | not published | Qwen release announcements | yes |

Dropped: Melis et al. 2025 (cited in v2; title could not be confirmed).
