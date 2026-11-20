# PromptShift v3: full study plan (draft for freeze)

Status: DRAFT, 20 Nov 2026. Becomes binding when committed as the "freeze v3" commit, before any model call on the fresh data.

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

### 4.2 Fresh posts (new)
- **Platform:** Bluesky, public posts, `app.bsky.feed.searchPosts` with an app password.
- **Time window:** created 1 May 2026 – 15 Nov 2026. All three models were released before May 2026 (Llama 3.3: Dec 2024, cutoff Dec 2023; Gemma 4: 2 Apr 2026, cutoff Jan 2025; Qwen 3.6: Apr 2026, cutoff undisclosed), so these posts cannot be in pretraining data. Check with maKI that the served weights are the April releases.
- **Sampling:** OLID's retained keywords that exist on Bluesky: "you are", "she is", "he is", "gun control", "conservatives", "antifa", "MAGA", "liberals". Dropped with reason: "to:BreitBartNews" and "-filter:safe" are Twitter-specific operators.
- **Stratification:** equal number per keyword and spread across months, to avoid one news event dominating.
- **Filters:** language English; original posts only (no reposts); at least 3 words; exact and near-duplicate removal.
- **Preprocessing as OLID:** mentions → `@USER`, links → `URL`.
- **Size:** 400 main + 40 pilot + 30 annotator-calibration = 470 posts, disjoint.
- **Ethics:** post text never committed; author IDs stored only as salted hashes; raw download deleted after the study; report counts and pattern, never quotes that identify authors.

### 4.3 Human gold labels for fresh posts
- **Guide:** one page, OLID Level A definitions verbatim, 6 worked examples (invented, not from the sample).
- **Calibration:** both annotators label the 30 calibration posts, discuss disagreements to clarify the guide (not the main data).
- **Main labelling:** two annotators label all 400 + 40 independently, blind to each other and to model outputs.
- **Disagreements:** a third team member decides, blind to who said what (mirrors OLID: two annotations, third on disagreement).
- **Anchor check:** the same two annotators also label 100 OLID test posts, mixed in. This measures agreement of our annotators with OLID gold, so the two gold standards are comparable.
- **Reported:** Krippendorff's alpha (fresh), agreement with OLID gold on the 100 anchors.
- Labelling cannot be automated: using an LLM to create the gold labels would make the evaluation circular.

## 5. Conditions (unchanged)
W0–W2, D0–D2, R0, RD0; temperature 0; max 16 tokens; shuffled order; same prompt strings (prompt registry hash a7444f2e…).

## 6. Hypotheses (per model; Holm across the primary family)

| ID | Statement | Estimand | Test | SESOI |
|---|---|---|---|---|
| H1 | Excess wording sensitivity is higher on fresh posts than on OLID | EWS = mean pairwise disagreement W0–W2 minus disagreement W0–R0; Δ = EWS_fresh − EWS_OLID > 0 | Independent-sample percentile bootstrap, one-sided | 0.01 |
| H2 | Labelling quality is lower on fresh posts | macro-F1 averaged over the six non-repeat conditions, fresh − OLID < 0 | Independent bootstrap (class-stratified), one-sided | 0.03 |
| H3 | The definition reduces prevalence error, on each dataset | \|prev_D − prev_human\| − \|prev_W − prev_human\| < 0 (averaged over wordings) | Paired bootstrap, one-sided | 0.02 |

Primary family: 3 + 3 + 6 = 12 tests, Holm, alpha 0.05. Each result is reported as an estimate with a 95% CI and a size verdict, whether or not it is significant.

Expected power: with 400 fresh posts, the CI half-width for H1 is roughly ±0.015 [estimate, to confirm by simulation before freezing]. H1 may therefore be inconclusive; that is an acceptable outcome if reported as such.

### Exploratory (labelled exploratory everywhere)
- E1: does the definition help more on fresh posts than on OLID (difference-in-differences of macro-F1)?
- E2: model confidence (log probabilities) for flipped vs stable posts.
- E3: between-model disagreement on each dataset.

## 7. Contamination evidence battery (OLID train 200, OLID test 200, fresh 400)

| Probe | Signal | Interpretation rule (fixed in advance) |
|---|---|---|
| P1 Guided vs general completion (existing) | ROUGE-L guided − general | Evidence if OLID-test gain > fresh gain (CI excludes 0). Within-dataset comparison has no time confound |
| P2 Near-verbatim review | Completions with ROUGE-L ≥ 0.7 | Specific (non-formulaic) reproductions count as direct evidence |
| P3 Dataset-cue label probe | Accuracy with "this tweet is from the OLID dataset; give its gold label" minus neutral prompt | Evidence of label memorisation if the gain exists on OLID and not on fresh posts |
| P4 Min-K% Prob (if maKI supports input log probabilities) | Score distribution and AUC, OLID vs fresh | Suggestive only (Duan et al. 2024 time-shift confound); train vs test comparison is unconfounded |
| P5 Data Contamination Quiz (if a 4th maKI model exists) | Rate of picking the original among 3 paraphrases, vs 25% chance | Evidence if OLID above chance and fresh at chance |
| Train vs test | Same probes on 200 OLID train posts | Similar scores → test split exposed like the rest of OLID |

Verdict wording is fixed: "evidence of exposure" / "no evidence detected (not evidence of absence)".

## 8. Analysis outputs
Tables: compliance, primary tests, per-condition metrics for both datasets, contamination battery, annotation agreement. Figures: OLID vs fresh EWS; macro-F1 by dataset; prevalence error with and without definition; probe scores by group.
Sensitivity: invalid-as-error; duplicates removed; OLID subsample matched on keyword and length; fresh posts with annotator disagreement removed.

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
