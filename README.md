# Same post, different prompt

**Wording, definitions and run-to-run noise in LLM offensive-language annotation**
IS 617 *Large Language Models for the Economic and Social Sciences*, University of Mannheim, HWS 2026

> **AI-use note.** The code was drafted with AI assistance (Claude, Anthropic) and
> reviewed by the team, which is responsible for it. The report's contributions
> section names the tools and tasks, as the course AI-use policy requires.

## Research questions and hypotheses

| RQ | Hypothesis | Grounded in | Test |
|---|---|---|---|
| RQ1: Does rewording the instruction change labels more than repeating the identical instruction? | **H1** (each model, one-sided): wording disagreement > repeat disagreement | Reiss 2023; Atreja et al. 2025 | Mean per-post pairwise disagreement W0–W2 minus W0–R0 disagreement |
| RQ2: Does adding OLID's definition change classification quality? | **H2** (each model, two-sided): macro-F1 differs with the definition. Secondary: fewer misses for models that under-detect, more false alarms for models that over-detect | Atreja et al. 2025; Melis et al. 2025 | Mean over wordings of macro-F1(D) − macro-F1(W) |
| RQ3: Does the definition reduce wording sensitivity? | **H3** (exploratory): repeat-adjusted wording disagreement is lower with the definition | Errica et al. 2025 | (D wording − RD0 repeat) − (W wording − R0 repeat) |
| Validity check | Contamination: guided completion overlaps the real text more than general completion | Golchin & Surdeanu 2024 | ROUGE-L, paired |

Key terms: **sensitivity** = labels change across prompts (needs no gold labels). **Correctness** = labels match the human (OLID) labels.

## Design

**Data:** TweetEval *offensive* (OLID level A; Zampieri et al. 2019; Barbieri et al. 2020), pinned to revision `b3a375b…`. Labels: OFFENSIVE / NOT_OFFENSIVE.

| Split | Posts | Purpose |
|---|---|---|
| Validation (pilot) | 40 random | Checks that every prompt and model gives a parseable label. **No hypothesis is tested on it.** |
| Test (main) | all 860 (240 offensive, 27.9%) | Every reported result |

**Models:** `gemma4-26b`, `qwen3.6-36b`, `llama-3.3-70b` through the maKI API; temperature 0, at most 16 output tokens, zero-shot, one request in flight at a time.

**Eight conditions** (3 models × 860 posts × 8 = 20,640 requests):

| | No definition | OLID definition |
|---|---|---|
| Wording P0 | W0 | D0 |
| Wording P1 | W1 | D1 |
| Wording P2 | W2 | D2 |
| Identical repeat of P0 | R0 | RD0 |

The three wordings differ **only in the first sentence**. The definition block is inserted after that sentence; nothing else changes. All requests are sent in a fixed random order, and identical repeats are never sent back to back.

**Statistics:** class-stratified paired percentile bootstrap (B = 10,000); Holm correction across the 6 primary tests (H1 and H2 × 3 models) at α = 0.05; smallest effect of interest 1 percentage point for H1 and 0.02 macro-F1 for H2, reported separately from significance. Primary analysis uses posts with valid labels in all 8 conditions; sensitivity analyses count invalid answers as errors and drop exact-duplicate posts.

**Baselines:** majority class (always NOT_OFFENSIVE) and a random guesser matching the true class shares.

**Human validation:** two team members label, blind, 50 posts on which the models flipped and 50 on which they were stable. Krippendorff's alpha per group shows whether model instability tracks genuine human ambiguity (Pangakis et al. 2023 on validating LLM annotation).

## How this answers the midterm feedback

| Feedback | What this project does |
|---|---|
| No hypotheses; unclear link to prior work | H1–H3, each tied to a specific published finding, registered on AsPredicted before the main run |
| Prompt had no definition of offensive | The definition is now an experimental factor (D conditions, OLID's own wording) |
| No action on data contamination | Guided vs general completion probe on 200 random test posts |
| No example posts | Example posts in the slides (masked slurs, content warning) |
| Purpose of the data splits unclear | Validation is used only for the pilot compliance check; test holds all results |
| Experimental set-up underspecified | Design table above; "shuffled" means request order only; exact request counts per stage |
| Bootstrap claims not demonstrated; CIs not on plots | Every figure draws its CIs; Holm-adjusted and unadjusted results are both reported |
| Inconsistent colours | One fixed, colour-blind-safe colour per model in every figure |
| No baseline comparison | Majority and class-weighted random baselines in every results table |
| Human validation not justified | Tests whether model flips coincide with human disagreement |
| Run-to-run noise not separated from wording (implicit in the setup questions) | Identical-repeat conditions R0 and RD0 |
| Wrong API acknowledgement | To be confirmed with the instructors before the report |

## Steps

Run from the repository root, in the project's Python environment (`pip install -r requirements.txt`).

| # | Command | Expected | Time |
|---|---|---|---|
| 1 | `python -m pytest tests -q` | `18 passed` | seconds |
| 2 | `python src/promptshift/dry_run.py` | `DRY_RUN_OK` | ~30 s |
| 3 | `export MAKI_API_KEY=...` then `python src/promptshift/check_api.py` | `PREFLIGHT_OK` | ~1 min |
| 4 | `python src/promptshift/prepare.py --stage pilot` | `PREPARE_OK` (960 requests) | seconds |
| 5 | `python src/promptshift/run.py --stage pilot --all` | `ALL_REQUESTS_COMPLETE` | ~6 min |
| 6 | `python src/promptshift/analyze.py --stage pilot` | `PILOT_REPORT_OK`, ideally 0 non-compliant | seconds |
| 7 | `python src/promptshift/prepare.py --stage main` then `python src/promptshift/probe.py prepare` | `PREPARE_OK` (20,640), `PROBE_PREPARE_OK` (1,200) | seconds |
| 8 | `git add -A && git commit -m "Freeze protocol"` | note the commit SHA | – |
| 9 | Register on aspredicted.org (hypotheses, design, analysis, commit SHA, hashes from `protocol/`) | save the PDF in `protocol/` | ~30 min |
| 10 | `python src/promptshift/run.py --stage main --limit 20`, then `--all` | `ALL_REQUESTS_COMPLETE` | ~2.2 h |
| 11 | `python src/promptshift/run.py --stage main --all --probe` | `ALL_REQUESTS_COMPLETE` | ~15–30 min |
| 12 | `python src/promptshift/analyze.py --stage main` and `python src/promptshift/probe.py analyze` | `ANALYSIS_OK`, `PROBE_ANALYSIS_OK` | ~1 min |
| 13 | `python src/promptshift/annotate.py export`; two members fill in their sheets alone; `python src/promptshift/annotate.py score` | `ANNOTATION_SCORE_OK` | ~1 h of labelling each |

If a run stops, rerun the same command: it resumes and never re-sends a finished request. If the pilot shows non-compliant answers, fix the cause **before** step 7, and record the change in the registration.

**Outputs** (in `data/main/results/`): `summary.md`, CSV tables, `fig1_primary_tests`, `fig2_wording_vs_repeat`, `fig3_prevalence`, `fig4_recall_tradeoff` (PNG and PDF), `probe_summary.md`, and `annotation/annotation_results.md`.

**What to commit:** code, `protocol/`, `predictions.jsonl` and `attempts.jsonl` (labels and hashes only), the figures and CSV summaries. **Never commit** `sample.jsonl`, `requests.jsonl`, `probe_items.jsonl`, `probe_predictions.jsonl`, the annotation sheets, or anything containing post text; `.gitignore` excludes `data/`.

## When something goes wrong

| Message | Fix |
|---|---|
| `Set the MAKI_API_KEY environment variable first.` | `export MAKI_API_KEY=...` in that terminal |
| `PREFLIGHT_PROBLEM` | Read `data/preflight.json`; an alias is missing or a call failed |
| `StopRuleTriggered` | The API is down; wait, then rerun the same command |
| `ModelMismatchError` | **Stop.** The model behind an alias changed; tell the instructors |
| `requests file changed since prepare step` | Restore the registered `requests.jsonl`; never rebuild mid-run |
| `N requests still have no result` | The run isn't finished yet |

## Layout

```
src/promptshift/   config, prompts, prepare, check_api, run, api, mock_api,
                   analyze, stats, plots, probe, annotate, dry_run, common
tests/             pipeline tests (no network)
protocol/          prompt registry and fingerprints (hashes only)
data/              gitignored: downloaded data, requests, predictions, results
```
