# PromptShift v3: full study plan (FROZEN)

Status: FROZEN, 25 Nov 2026. Binding from the "Freeze plan v3" commit onward. That commit precedes every model call on the fresh posts and the finalisation of their gold labels. Any later change is listed in the deviation log (section 10) with date and reason.

This plan responds to the 19 Nov 2026 meeting feedback:
(1) hypotheses must be sharper; (2) a slide must state how the work differs from prior work, because the RQ is otherwise a replication;
(3) OLID (2019) is almost certainly in the training data, so use data released after the models' cutoffs.

---

## 1. What prior work already answers (so we do not claim it)

| Paper | What it did | Consequence for us |
|---|---|---|
| Zampieri et al. 2019 (OLID) | Keyword-sampled tweets; Level A OFF/NOT definitions; two annotations per tweet, third on disagreement | Source of our definition, keywords, and annotation procedure; we copy the procedure for the fresh data |
| Barbieri et al. 2020 (TweetEval) | Standardised OLID splits | Our pinned OLID test split |
| Reiss 2023 | ChatGPT annotation varies across prompt variants and repeats | Prompt sensitivity is known; repeats as a noise floor are not new by themselves |
| Atreja et al. 2025 (ICWSM) | 4 CSS tasks incl. toxicity; varied definition inclusion, output type, explanation, length | **The definition effect itself is NOT new.** We must not present H2-old as original |
| Errica et al. 2025 | Sensitivity and consistency metrics for prompt variation | Our disagreement measures follow this line |
| Sclar et al. 2024; Mizrahi et al. 2024 | Large prompt-format sensitivity; multi-prompt evaluation | Our minimal one-sentence variation is a lower bound |
| Li & Flanigan 2024 (AAAI) | Zero/few-shot LLMs do better on datasets released before the training-data date than after | Before/after comparison is an established contamination signal; our accuracy contrast (H2) is a CSS-domain test of it, not a new idea |
| Golchin & Surdeanu 2024 (ICLR) | Guided vs general completion to detect instance-level contamination | Our existing probe |
| Golchin & Surdeanu, Data Contamination Quiz (arXiv 2311.06233) | Model picks the original instance among word-level paraphrases | Optional probe if a 4th model is available on maKI |
| Shi et al. 2024 (ICLR), Min-K% Prob | Membership score from the lowest-probability tokens | Optional probe if maKI returns input-token log probabilities |
| Duan et al. 2024 (COLM) | Membership inference barely beats chance on LLMs; apparent success often comes from temporal distribution shift | **Any OLID-vs-2026 difference in membership scores is confounded by time; we treat it as suggestive only** |
| "Test of Time" (ACL 2026) | Post-cutoff performance decay is not always a clean contamination signal | We control difficulty and report the confound |
| Casanova, Kocielnik & Alvarez 2026 (ICML) | Across 9 LLMs and 5 toxicity datasets, performance tracks definition alignment, not text-memorisation metrics; prompting rescues only about a third of zero-shot errors | Closest prior work. They measure memorisation with metrics; we use **data that is guaranteed unseen** and study **prompt sensitivity and prevalence**, which they do not |
| Pangakis et al. 2023 | Uses password-protected datasets to reduce contamination | An alternative strategy; ours is temporal |

To verify before citing: the exact claims of Errica et al. and Reiss in the paper text, and Melis et al. 2025 (title not yet confirmed; dropped from the core list until checked).

## 2. What is new in v3 (the "how we differ" slide)

1. **Guaranteed-unseen contrast for prompt sensitivity.** Prior work measures prompt sensitivity on public benchmarks. We measure it on (a) OLID, likely seen, and (b) posts written after all three models were released, which cannot be in training data. Question: is prompt sensitivity an artefact of memorised data?
2. **Social-science estimand.** We report prevalence error (how wrong the share of offensive posts is), not only F1.
3. **Explicit nondeterminism floor on both datasets** (exact repeats), because Gemma is not deterministic at temperature 0 in our data.
4. **Contamination probes with a negative control.** Each probe is run on OLID train, OLID test and the fresh posts, so we measure whether the probe can detect anything at all.

## 3. Research question

> Do prompt wording and a codebook definition affect LLM annotation of offensive language differently on data the models have likely seen in training (OLID test, 2019) than on posts written after the models' public release (May 2026 onward), and what are the consequences for prevalence estimates?

## 4. Data

### 4.1 OLID (existing)
TweetEval offensive test split, 860 posts, pinned revision; 20,640 predictions already collected (commit 1d75816). Not rerun.

### 4.2 Fresh posts (new) — collected 25 Nov 2026
- **Platform:** Bluesky, public posts, `app.bsky.feed.searchPosts` with an app password.
- **What was collected** (protocol/fresh_collection_fingerprint.json): 15,561 posts returned, 11,405 kept after filters (dropped: 2,006 quote posts, 1,180 keyword only outside the text, 714 too short, 31 outside the window, 225 already in pool), 10,926 after de-duplication; 470 sampled (58–59 per keyword, 64–72 per month, at most 2 per author).
- **Time window:** created 1 May 2026 – 15 Nov 2026. All three models were released before May 2026 (Llama 3.3: Dec 2024, cutoff Dec 2023; Gemma 4: 2 Apr 2026, cutoff Jan 2025; Qwen 3.6: Apr 2026, cutoff undisclosed), so these posts cannot be in pretraining data. Check with maKI that the served weights are the April releases.
- **Sampling:** OLID's retained keywords that exist on Bluesky: "you are", "she is", "he is", "gun control", "conservatives", "antifa", "MAGA", "liberals". Dropped with reason: "to:BreitBartNews" and "-filter:safe" are Twitter-specific operators.
- **Stratification:** equal number per keyword and spread across months, to avoid one news event dominating.
- **Filters:** language English; original posts only (no reposts, no quote posts); at least 3 content words; keyword present in the text; exact and near-duplicate removal (Jaccard ≥ 0.8).
- **Preprocessing as OLID:** mentions → `@USER`, links → `URL`.
- **Size:** 400 main + 40 pilot + 30 annotator-calibration = 470 posts, disjoint.
- **Ethics:** post text never committed; author IDs stored only as salted hashes; raw download deleted after the study; report counts and pattern, never quotes that identify authors.

### 4.3 Human gold labels for fresh posts
- **Guide:** one page, OLID Level A definitions verbatim, 6 worked examples (invented, not from the sample).
- **Calibration:** both annotators label the 30 calibration posts, discuss disagreements to clarify the guide (not the main data).
- **Main labelling:** two annotators (A, B) label the 400 main posts plus 100 OLID anchors (500 items, separate random orders) independently, blind to each other and to model outputs. The 40 pilot posts are not labelled: the pilot only checks answer format.
- **Disagreements:** a third team member (C) labels each disagreement without seeing A's or B's labels; the final label is the majority of the three (mirrors OLID).
- **Blinding to model outputs:** the annotators also run the pipeline. Nobody opens fresh predictions or runs any fresh analysis until `annotate_fresh.py finalize` has produced `gold_main.jsonl`; the runner prints progress only.
- **Anchor check:** the same two annotators also label 100 OLID test posts, mixed in. This measures agreement of our annotators with OLID gold, so the two gold standards are comparable.
- **Reported:** Krippendorff's alpha (fresh), agreement with OLID gold on the 100 anchors.
- Labelling cannot be automated: using an LLM to create the gold labels would make the evaluation circular.

## 5. Conditions (unchanged)
W0–W2, D0–D2, R0, RD0; temperature 0; max 16 tokens; shuffled order; same prompt strings (prompt registry hash a7444f2e…).

## 6. Hypotheses (per model; Holm across the primary family)

What was already seen: the OLID side of every estimand below was observed before this plan (OLID results, commit fa704fd). The fresh side is unseen. The hypotheses are confirmatory because their fresh-side values, and therefore the contrasts, are unknown at freeze time.

| ID | Statement | Estimand (per model) | Test | SESOI |
|---|---|---|---|---|
| H1 | Excess wording sensitivity is higher on fresh posts than on OLID | Per post: mean pairwise disagreement among W0, W1, W2 minus 1[W0 ≠ R0]; EWS = mean over posts; Δ = EWS_fresh − EWS_OLID > 0 | Independent percentile bootstrap (posts resampled within each dataset), B = 10,000, one-sided p = (1 + #{Δ* ≤ 0}) / (B + 1) | 0.01 |
| H2 | Labelling quality is lower on fresh posts | macro-F1 (vs each dataset's gold labels) averaged over W0–W2 and D0–D2; Δ = F1_fresh − F1_OLID < 0 | Independent bootstrap, class-stratified within each dataset, one-sided p = (1 + #{Δ* ≥ 0}) / (B + 1) | 0.03 |
| H3 | On fresh posts, the definition reduces prevalence error | PE_X = mean over the three wordings of \|predicted offensive share − gold offensive share\| under condition set X; Δ = PE_D − PE_W < 0 | Paired bootstrap over posts, class-stratified, one-sided | 0.02 |

Primary family: H1 × 3 + H2 × 3 + H3 × 3 = 9 tests, Holm, alpha 0.05. H3 on OLID is reported alongside but is NOT confirmatory (already observed: all three models moved toward the gold share). Each result is reported as estimate, 95% CI, Holm p and size verdict (as in v2: "materially positive/negative" if the one-sided bound clears the SESOI; "practically equivalent" if the CI lies within ±SESOI; otherwise "inconclusive").

Invalid answers: posts with an invalid answer in any condition entering an estimand are excluded from that estimand (as in v2); sensitivity: invalid counted as disagreement (H1) or error (H2, H3).

Gold-standard comparability (fixed in advance): if either annotator's Cohen's kappa against OLID gold on the 100 anchors is below 0.60, H2 is reported but marked "not interpretable as a contamination effect" (the two gold standards would differ too much). H1 does not depend on gold labels.

Expected precision (approximation from OLID: about 4% of posts flip, per-post EWS sd ≈ 0.13): CI half-width for H1 Δ ≈ ±0.016; for H2 Δ ≈ ±0.06. H1 has roughly 35–40% power for a true Δ of 0.01 and detects Δ ≳ 0.026 reliably; H2 is underpowered for 0.03. Inconclusive results are acceptable and will be reported as such.

### Exploratory (labelled exploratory everywhere)
- E1: does the definition help more on fresh posts than on OLID (difference-in-differences of macro-F1)?
- E2: model confidence (log probabilities) for flipped vs stable posts.
- E3: between-model disagreement on each dataset.
- E4: flip rate on fresh posts where the two annotators disagreed vs agreed (do models flip on posts humans also find ambiguous?). Replaces the separate 50/50 flip-validation annotation.

## 7. Contamination evidence battery (OLID train 200, OLID test 200, fresh 400)

| Probe | Signal | Interpretation rule (fixed in advance) |
|---|---|---|
| P1 Guided vs general completion (existing) | ROUGE-L guided − general | Evidence if OLID-test gain > fresh gain (CI excludes 0). Within-dataset comparison has no time confound |
| P2 Near-verbatim review | Completions with ROUGE-L ≥ 0.7 | Specific (non-formulaic) reproductions count as direct evidence |
| P3 Dataset-cue label probe | Accuracy with "this tweet is from the OLID dataset; give its gold label" minus neutral prompt | Evidence of label memorisation if the gain exists on OLID and not on fresh posts |
| P4 Min-K% Prob (if maKI supports input log probabilities) | Score distribution and AUC, OLID vs fresh | Suggestive only (Duan et al. 2024 time-shift confound); train vs test comparison is unconfounded. Per-item API failures are recorded; a model failing on more than 10% of items is excluded, because the scored items would no longer be a random subset (rule added 20 Nov after Gemma returned NaN errors on some posts, before any P4 result existed) |
| P5 Data Contamination Quiz (paraphrases by ministral-3-14b) | Rate of picking the original among 3 paraphrases, vs 25% chance | Evidence only if OLID accuracy exceeds the FRESH accuracy (CI excludes 0). Above-chance accuracy alone is not evidence: real posts are messier than machine paraphrases, so a model can spot the original without memory. Reported with accuracy by correct position (letter bias) and among valid answers |
| Train vs test | Same probes on 200 OLID train posts | Similar scores → test split exposed like the rest of OLID |

Deviation log (20 Nov): Qwen began its quiz replies with reasoning and was cut off at the 4-token limit in 105 of 391 items, before giving a letter. Qwen is re-asked the identical quiz prompts with a 256-token limit, parsed by a fixed rule (explicit "answer is X", then a bold letter, then the last standalone letter). Both versions are reported; the 256-token version is primary for Qwen, and the fresh posts use the same setting.

Verdict wording is fixed: "evidence of exposure" / "no evidence detected (not evidence of absence)".

Fresh group of the battery: all gold-labelled main posts that meet the probe's length rule (≥ 12 content tokens). P1 and P3 need gold labels and run after `finalize`; P4 and P5 can run earlier but are not analysed before `finalize`.

## 8. Analysis outputs
Tables: compliance, primary tests, per-condition metrics for both datasets, contamination battery, annotation agreement. Figures: OLID vs fresh EWS; macro-F1 by dataset; prevalence error with and without definition; probe scores by group.
Sensitivity: invalid-as-error; duplicates removed; OLID subsample matched on length (keyword is not recorded for OLID posts); fresh posts with annotator disagreement removed.
Analysis code: `analyze_v3.py`, written after this freeze and tested on mock data before `finalize`; it implements section 6 exactly. Any difference between code and plan is logged in section 10.

## 9. Order of work and completion criteria

| Phase | Done when |
|---|---|
| A Contamination evidence (OLID) | P1–P3 (+P4/P5 if supported) run on OLID train and test; P2 reviewed |
| B Framing | This plan reviewed by the team; email sent to professor |
| C Fresh data | 470 posts collected, filtered, hashed; feasibility log saved |
| D Gold labels | Calibration done; all labels in; alpha and anchor agreement computed; adjudication finished |
| E Freeze | Commit containing this plan, data hash, label hash, seeds |
| F Runs | Pilot 100% compliant; main 9,600 calls; probes on fresh posts |
| G Analysis | All tables and figures generated; every number in slides traced to an output file |
| H Deliverables | Slides with novelty slide; two timed rehearsals; report; admin |

## 10. Deviation log

| Date | Change | Reason | Before or after seeing fresh results |
|---|---|---|---|
| 20 Nov | Min-K%: per-item failures recorded; model excluded if > 10% fail | Gemma returned NaN errors on some posts | Before any P4 result |
| 20 Nov | Qwen quiz re-asked with 256-token limit | 105/391 replies cut off at 4 tokens | Before fresh data existed |
| 25 Nov | Pilot posts not labelled; anchors 100; third annotator labels blind, majority vote | Reduce annotation load; mirror OLID | Before fresh model calls |
| 25 Nov | Primary family 9 tests (H3 on OLID descriptive only) | H3 on OLID already observed | Before fresh model calls |
| 25 Nov | Mastodon collector kept in repo but not used | Bluesky worked | Before fresh model calls |
| 25 Nov | protocol/contam_quiz_fingerprint.json restored from data/contam/quiz_meta.json | The test suite had overwritten it with mock values (bug fixed; a guard test now blocks this) | Before fresh model calls |
| 25 Nov | Clarification (no change to tests): simulated 95% CI half-widths at the real sample sizes are about ±0.014 (H1), ±0.046 (H2), ±0.068 (H3); length matching uses only fresh length bins that also contain OLID posts | Precision for H3 was not stated; analyze_v3.py written and tested on synthetic data | Before fresh results; fresh predictions not opened |
| 26 Nov | Added sensitivity analysis: fresh gold = annotator A's labels only (no adjudication); gold file now stores each annotator's label | Lets readers check that results do not depend on annotator B or on the adjudication | Before fresh results; analysis not yet run |
