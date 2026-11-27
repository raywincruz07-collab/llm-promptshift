# Same post, different prompt (PromptShift v3)

**Do prompt wording and a codebook definition change LLM labels of offensive language differently on data the models may have seen in training than on posts written after the models were released?**

IS 617 *Large Language Models for the Economic and Social Sciences*, University of Mannheim, HWS 2026

> **AI-use note.** Code and documentation were drafted with AI assistance (Claude, Anthropic) and reviewed by the team, which is responsible for them. No LLM produced any gold label. The report's contributions section names the tools and tasks, as the course AI-use policy requires.

**Start here:** [docs/PromptShift_v3_Simple.pdf](docs/PromptShift_v3_Simple.pdf) (8-page visual overview) · [PROJECT_MAP.md](PROJECT_MAP.md) (what changed and why) · [PLAN_v3.md](PLAN_v3.md) (frozen plan, hypotheses, deviation log)

---

## Result in four lines

1. **Wording** changes about 3 in 100 labels per pair of wordings, about equally on OLID (2.5–3.0%) and on unseen 2026 posts (2.7–3.5%). Exact repeats change 0–0.75%. **H1 (more sensitivity on unseen posts): not supported for any model**; the size is inconclusive.
2. **Adding OLID's definition** moves every model's share of "offensive" towards the human share and lowers prevalence error on unseen posts. **H3: significant for all three models** (Holm p = 0.0009); clearly larger than the 0.02 minimum for Gemma and Llama, inconclusive in size for Qwen.
3. **Accuracy on old vs new data (H2) is not interpretable:** our annotators agree with OLID's labels only moderately (kappa 0.43 and 0.37, below the 0.60 rule fixed in advance).
4. **Contamination:** five memorisation tests found no sign that OLID is more familiar to the models than unseen posts (no evidence detected, not evidence of absence).

## Research question and hypotheses (frozen 25 Nov 2026, commit `e7610de`)

| ID | Statement (verbatim from the plan) | Smallest effect | Result (Holm across 9 tests) |
|---|---|---|---|
| H1 | Excess wording sensitivity is higher on fresh posts than on OLID | 0.01 | Gemma −0.002, Qwen +0.005, Llama +0.010; none significant |
| H2 | Labelling quality is lower on fresh posts | 0.03 macro-F1 | Not interpretable (gold-label agreement rule not met) |
| H3 | On fresh posts, the definition reduces prevalence error | 0.02 | Gemma −0.090, Qwen −0.032, Llama −0.058; all p = 0.0009 |

Estimands, tests and size verdicts: [PLAN_v3.md §6](PLAN_v3.md). All numbers: [artifacts/results_v3/summary_v3.md](artifacts/results_v3/summary_v3.md).

## Design

| | |
|---|---|
| **Models** | `gemma4-26b`, `qwen3.6-36b`, `llama-3.3-70b` (names as served by the University of Mannheim maKI API); temperature 0, at most 16 output tokens, log probabilities recorded |
| **Conditions** | 3 wordings (W0–W2, first sentence differs only) × with/without OLID's definition (D0–D2), plus exact repeats R0 and RD0; shuffled order, a repeat never follows its original |
| **Old data** | OLID / TweetEval offensive test split, 860 posts (27.9% offensive), pinned revision `b3a375b` → 20,640 calls |
| **New data** | 400 Bluesky posts written 1 May – 15 Nov 2026, after all three models were released → 9,600 calls |
| **Statistics** | Percentile bootstrap (B = 10,000), one-sided tests, Holm across 9 primary tests, smallest effects of interest fixed in advance |

**Why the new posts cannot be in training data.** Llama 3.3 was released on 6 Dec 2024 (training data to Dec 2023); Gemma 4 on 2 Apr 2026 (training data to Jan 2025); Qwen 3.6 in April 2026. The collector rejects any post written before 1 May 2026 (`EARLIEST` in [collect_fresh.py](src/promptshift/collect_fresh.py)). Assumption: maKI serves the published weights; every run stops if the service reports a different model than requested.

**How the new data were built** ([protocol/fresh_collection_fingerprint.json](protocol/fresh_collection_fingerprint.json)):

```
Bluesky search, OLID's 8 keywords, May–Nov 2026   15,561 posts
  − 2,006 quote posts, 1,180 keyword not in text,
    714 too short, 31 outside the window, 225 repeats  11,405
  − near-duplicates (Jaccard ≥ 0.8)                  10,926
  balanced sample (58–59 per keyword, 64–72 per month,
  ≤ 2 per author)                                       470 = 400 main + 40 pilot + 30 calibration
  human labels on the 400 main posts (46–54 per keyword) 98 offensive = 24.5% (OLID test: 27.9%)
```

**Composition.** 24.5% of the new posts are offensive, from 13.2% (*you are*) to 40.4% (*MAGA*) by keyword; median 28 words (OLID 21). Keyword sampling, as in OLID, over-represents political and person-directed posts, so results describe keyword-matched posts, not Bluesky as a whole. Tables: [artifacts/fresh/dataset_description.md](artifacts/fresh/dataset_description.md); full description: [docs/DATA_CARD.md](docs/DATA_CARD.md).

**Labels.** The guide uses OLID's definitions verbatim; its 7 rules and 6 invented examples were drafted with an LLM (Claude) and reviewed by the team. Two team members (A, B) labelled all 400 posts plus 100 OLID anchor posts; they agreed on 365 (91.25%, Krippendorff's alpha 0.78). Of the 35 disagreements, 34 are A = offensive / B = not offensive and 32 were marked "unsure"; a third team member (C) labelled them (26 NOT, 9 OFF) and the final label is the majority. Annotator A reported labelling mainly from his own understanding rather than the full rules. A sensitivity analysis with annotator A's labels alone is reported (H3 stays significant for Gemma and Qwen, not for Llama).

**Contamination battery** ([artifacts/contam/contam_summary.md](artifacts/contam/contam_summary.md)), each on OLID train, OLID test and the new posts as a negative control: guided vs general completion, near-verbatim review, dataset-cue label probe, Min-K% Prob, Data Contamination Quiz.

## Limitations

- **Assumed, not shown:** that OLID is in the training data. No probe detects it, so a null H1 can mean that memorisation does not drive wording sensitivity, or that OLID is not memorised enough to matter.
- **Not a pure contamination contrast:** the datasets also differ in platform (Twitter vs Bluesky), year and topics. A length-matched OLID subsample gives the same H1 result.
- **Power:** H1 has roughly 35–40% power for its smallest effect; we can say "no detectable difference", not "no difference".
- **Three minimal wordings:** the wording effect is a lower bound for real prompt variation.
- **Gold labels from our team**, not OLID's annotators; this is why H2 cannot be interpreted. The guide's rules were LLM-drafted, and annotator A labelled mainly from his own understanding.
- **Keyword sample:** results describe keyword-matched, mostly political posts, not Bluesky as a whole.
- **Model identity** depends on the maKI service; Qwen 3.6 does not publish a training cut-off.

## Reproduce

Run from the repository root (`pip install -r requirements.txt`). Set secrets without echoing them:
`read -s -p "maKI key: " MAKI_API_KEY && export MAKI_API_KEY`

| # | Step | Command | Expected |
|---|---|---|---|
| 1 | Tests (no network) | `python -m pytest tests -q` | `47 passed` |
| 2 | Collect and sample new posts | `python src/promptshift/collect_fresh.py collect` then `... sample` (needs `BSKY_HANDLE`, `BSKY_APP_PASSWORD`) | `SAMPLE_OK` |
| 3 | Labelling sheets, scores, final labels | `python src/promptshift/annotate_fresh.py sheets` → `score` → `finalize` | `FINALIZE_OK` |
| 4 | Model runs on new posts | `python src/promptshift/prepare.py --stage fresh` then `python src/promptshift/run.py --stage fresh --all` | `ALL_REQUESTS_COMPLETE` |
| 5 | Contamination battery | `python src/promptshift/contam.py prepare --groups olid_train olid_test fresh`, `run --part stage1 --groups ... --all`, `quiz-prepare`, `run --part quiz ...`, `minkprob --all`, `analyze` | `CONTAM_ANALYSIS_OK` |
| 6 | Analysis | `python src/promptshift/analyze_v3.py` | `ANALYSIS_V3_OK` |
| 7 | Dataset description and release files | `python src/promptshift/dataset_card.py describe` then `release` | `DESCRIBE_OK`, `RELEASE_OK` |
| 8 | Check the post IDs against Bluesky (no login) | `python src/promptshift/dataset_card.py verify-ids` (live demo: `--sample 3`) | `VERIFY_IDS_OK` (27 Nov: 396 match, 4 deleted, 0 differ) |

OLID runs (v2) use `--stage pilot` and `--stage main` with the same scripts. If a run stops, rerun the same command: it resumes and never re-sends a finished request.

## Where everything is

| What | Where |
|---|---|
| Frozen plan and deviation log | [PLAN_v3.md](PLAN_v3.md) |
| Results, figures, sensitivity analyses | [artifacts/results_v3/](artifacts/results_v3/) |
| Model answers (labels and hashes only, no text) | [artifacts/fresh/](artifacts/fresh/), [artifacts/predictions.jsonl](artifacts/predictions.jsonl) |
| Contamination battery | [artifacts/contam/](artifacts/contam/) |
| Prompts and data fingerprints | [protocol/](protocol/) |
| References (checked) | [docs/references.md](docs/references.md), [docs/references.bib](docs/references.bib) |
| Midterm version | `main` branch |

## Data availability

The Bluesky post IDs (AT URIs and links), keyword, month and all labels (A, B, C, final) are submitted with the report; post texts go to the examiners only. Both files are produced by `dataset_card.py release` under `data/fresh/release/` (gitignored). Posts deleted by their authors can no longer be retrieved. On 27 Nov 2026, 396 of the 400 IDs returned exactly our stored text and date; 4 posts had been deleted ([artifacts/fresh/id_verification.json](artifacts/fresh/id_verification.json)).

## Privacy and ethics

No post text is committed (`data/` and `*CONTAINS_TWEET_TEXT*` are git-ignored). Authors are stored only as salted hashes. Example posts in slides and documents are invented and labelled as such. The raw Bluesky download is deleted after the course; API keys and the Bluesky app password are read from the environment and never written to files.

## Acknowledgement

Model queries ran on the maKI service of the University of Mannheim. *(Exact wording to be confirmed with the instructors.)*
