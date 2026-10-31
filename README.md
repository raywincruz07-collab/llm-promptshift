# LLM PromptShift

### How stable are LLM decisions when only the prompt changes?

LLM PromptShift studies how small changes in prompt wording affect offensive-language classification. It measures classification consistency and performance across three hosted model aliases and three prompt variants on the same posts. Archived predictions and a reproducible evaluation workflow make the analysis inspectable without new model calls.

| Model aliases | Prompt variants | Held-out test posts | Archived prediction records |
| :---: | :---: | :---: | :---: |
| **3** | **3** | **860** | **10,620** |

The 10,620 records span pilot, development, held-out, and repeatability runs.

[Overview](#overview) · [Results](#results) · [Visualizations](#visualizations) · [Reproduction](#reproduction) · [Methodology](#methodology) · [Structure](#structure) · [Limitations](#limitations)

## Overview

The question is whether closely related zero-shot instructions produce consistent `OFFENSIVE` or `NOT_OFFENSIVE` labels for the same post. The study used 12 pilot posts, 300 development posts from TweetEval's validation split, and 860 held-out test posts. The three prompts (P0, P1, P2) were designed to preserve the task, though their semantic equivalence was not independently established.

The archive contains 108 pilot, 2,700 development, 7,740 held-out, and 72 repeatability prediction records. The principal stability measure is the share of held-out posts for which at least one prompt produces a different label within a model.

## Results

| Hosted maKI API alias | Posts with a differing label | P0 macro-F1 | P1 macro-F1 | P2 macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| `gemma4-26b` | 39 / 860 (4.53%) | 0.6908 | 0.6801 | 0.7002 |
| `qwen3.6-36b` | 41 / 860 (4.77%) | 0.7037 | 0.7199 | 0.6949 |
| `llama-3.3-70b` | 31 / 860 (3.60%) | 0.7789 | 0.7856 | 0.7880 |

![Prompt sensitivity across the three model aliases](figures/figure1_prompt_sensitivity.png)

*Figure 1. Prompt sensitivity in development and held-out data, with confidence intervals.* [Download PDF](figures/figure1_prompt_sensitivity.pdf)

The disagreement rates and macro-F1 scores are descriptive. **All nine post-hoc Bonferroni-adjusted comparison intervals include zero**: six within-model P1/P2 versus P0 macro-F1 differences and three between-model disagreement-rate differences. These comparisons do not establish statistically significant superiority for a prompt or model. The [verified results](docs/verified_results_v2.md) give all nine conditions, intervals, class recalls, and sensitivity results.

## Visualizations

| Held-out macro-F1 | Class-specific recall |
| --- | --- |
| ![Held-out macro-F1 by model and prompt](figures/figure2_macro_f1.png) | ![Offensive and non-offensive recall by model and prompt](figures/figure3_recall_tradeoff_v2.png) |
| [Figure 2 PDF](figures/figure2_macro_f1.pdf): marginal confidence intervals; paired differences are assessed in the results table. | [Figure 3 PDF](figures/figure3_recall_tradeoff_v2.pdf): offensive and non-offensive recall across all nine conditions. |

## Reproduction

### Quick Start

Use Python 3.12 on Ubuntu or WSL (the reference environment used Python 3.12.3):

```bash
git clone https://github.com/raywincruz07-collab/llm-promptshift.git
cd llm-promptshift
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r reproducibility/requirements-lock.txt
python reproducibility/test_public_analysis.py
```

The initial run needs internet access to download the pinned TweetEval files and verify their fingerprints. It reconstructs the historical inputs, analyzes archived predictions, and exports **17 CSV tables** and **6 figure files** to ignored `reproduced_outputs/`. It makes **zero new LLM API calls** and needs no maKI key. Successful output includes `PUBLIC_DATA_BOOTSTRAP_OK`, `PUBLIC_ANALYSIS_REPRODUCTION_OK`, and `REPRODUCED_OUTPUTS_EXPORTED_OK`.

## Methodology

- **Data:** CardiffNLP [TweetEval `offensive`](https://huggingface.co/datasets/cardiffnlp/tweet_eval), pinned to revision `b3a375baf0f409c77e6bc7aa35102b7b3534f8be`. The held-out set contains 620 non-offensive and 240 offensive posts. Raw tweet text is downloaded for reproduction and is not tracked here.
- **Prompts and models:** Three instructions in [prompts.py](src/promptshift/prompts.py) were evaluated with maKI API aliases `gemma4-26b`, `qwen3.6-36b`, and `llama-3.3-70b`. Recorded inference settings include temperature 0 and `max_tokens=16`. These aliases do not establish immutable checkpoint identities; see [model provenance](reproducibility/model_provenance.md).
- **Analysis:** Accuracy, macro-F1, class recall, and within-model prompt disagreement were computed on archived labels. Comparisons use 5,000 stratified paired bootstrap resamples, with P0 as the reference prompt. An exact-text duplicate sensitivity analysis retains 840 singleton posts.
- **Input reconstruction:** [Sample IDs](reproducibility/sample_manifest_v1.json) and [test-text suffixes](reproducibility/test_text_suffix_manifest.json) preserve historical inputs. The original development sampling seed is unknown; recorded test suffixes must be retained to reproduce the archived requests.

## Structure

```text
src/promptshift/          Prompts, preparation, live inference, analysis, plots
reproducibility/          Bootstrap, reproduction test, lockfile, manifests
artifacts/predictions/    Archived JSONL prediction records
figures/                  Three final plots, each as PNG and PDF
docs/                     Verified results and qualitative provenance
reproduced_outputs/       Locally regenerated tables and figures (Git-ignored)
```

Historical `src/promptshift/run_*.py` live-inference scripts are separate from the verified archived-prediction reproduction workflow. They require authorized maKI access and may not reproduce historical responses. The live `run_test.py` entry point still depends on a held-out protocol file that is absent from this repository; that dependency remains unresolved.

## Limitations

The study covers one English social-media dataset, three nearby prompts, and one hosted inference environment. Hosted checkpoint and serving details are unverified, and temperature 0 does not guarantee repeatability. The held-out set includes exact-text duplicates and a conflicting gold-label duplicate; possible model pretraining exposure to TweetEval cannot be excluded. The multiplicity correction was introduced after inspecting test results. Confidence intervals that include zero do not prove equivalence. See the [qualitative-case provenance](docs/qualitative_case_provenance.md).

**Attribution:** Dataset: CardiffNLP TweetEval `offensive`. The project originated in the University of Mannheim IS6170 course, *Large Language Models for the Economic and Social Sciences* (HWS 2026). Tweet content and model access have separate rights and are not bundled. Contributor credit and a code license require team confirmation before publication.
