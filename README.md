# Prompt Robustness in LLM-Based Offensive Language Classification

**IS6170 — Large Language Models for the Economic and Social Sciences**
University of Mannheim · HWS 2026

A reproducible, small-scale empirical investigation of how closely related instructions affect zero-shot offensive-language classification. This repository provides the experimental code, pinned data reconstruction manifests, validated prediction records, numerical analysis, and visualization workflow.

> **Reproduction scope:** The archived-prediction analysis can be reproduced **without a maKI API key**. Repeating *new* inference calls requires authorized maKI access and may produce different outputs if the hosted deployment has changed. The original dataset texts are downloaded separately; they are not distributed here.

## Research question

**How does semantically equivalent prompt wording affect the consistency and accuracy of LLM offensive-language classification across three models?**

We evaluate three closely related prompt formulations (P0, P1, P2) on identical posts. We measure:

- **Prediction disagreement:** the fraction of posts for which at least one of P0/P1/P2 gives a different binary label within the same model.
- **Classification performance:** accuracy, macro-F1, and offensive/non-offensive recall.
- **Uncertainty:** post-level paired bootstrap confidence intervals for metrics and planned comparisons.
- **Sensitivity:** the effect of excluding exact-text duplicate groups from the held-out evaluation.

These prompts were designed to preserve the task, but their semantic equivalence was not established through an independent annotation study.

## Dataset and experimental design

We use the **TweetEval `offensive`** configuration, obtained from the [CardiffNLP dataset on Hugging Face](https://huggingface.co/datasets/cardiffnlp/tweet_eval).

| Partition | Rows | Use |
| --- | ---: | --- |
| Official validation split | 1,324 | Source for development samples |
| Pilot | 12 | Initial checks; 6 posts per gold class |
| Main validation sample | 300 | Development analysis; 196 non-offensive, 104 offensive |
| Official held-out test | 860 | Final analysis; 620 non-offensive, 240 offensive |

The recorded **dataset revision** is:

```text
b3a375baf0f409c77e6bc7aa35102b7b3534f8be
```

The original pilot and main sampling seed was not independently verified. Instead, the repository records the *actual historical row IDs* in `reproducibility/sample_manifest_v1.json` so those exact samples can be reconstructed. The historical test data have one to four additional trailing spaces per post compared with the downloaded source; their exact per-row suffixes are preserved in `reproducibility/test_text_suffix_manifest.json`. The reason for those suffixes is unknown. **Do not strip or alter them when reproducing the archived prompts.**

The reproducibility script verifies source hashes and rebuilds datasets and request files without needing copies of the original researchers' `data/` directory.

## Models, prompts, and inference configuration

The experiments used the following **maKI API aliases** (not independently verified checkpoint identifiers):

| API alias | Prompt conditions |
| --- | --- |
| `gemma4-26b` | P0, P1, P2 |
| `qwen3.6-36b` | P0, P1, P2 |
| `llama-3.3-70b` | P0, P1, P2 |

The prompt templates appear in [`prompts.py`](prompts.py). Each asks for a single label: `OFFENSIVE` or `NOT_OFFENSIVE`. Recorded inference settings include zero-shot instructions, temperature `0`, and `max_tokens=16`. The API base URL was `https://maki.uni-mannheim.de/v1`.

The maKI model-list endpoint confirmed the aliases but did **not** reveal immutable model checkpoint IDs, weight hashes, quantization, inference engine, chat template, or complete serving configuration. The API returned generic metadata fields, which must not be interpreted as definitive model-maker or release information. See [`reproducibility/model_provenance.md`](reproducibility/model_provenance.md). We make **no model-architecture or model-size causal claim** based on these aliases.

### Archived inference quantities

| Stage | Prediction records |
| --- | ---: |
| Pilot | 108 |
| Main validation | 2,700 |
| Held-out test | 7,740 |
| Repeatability | 72 |
| **Total** | **10,620** |

The files in `artifacts/predictions/` contain labels and request/prediction metadata but not original post texts or full rendered prompts. Publication of these artifacts remains subject to the final repository-content and permissions review.

## Verified results

### Posts with at least one differing label across prompts

| API alias | Posts / 860 | Disagreement rate |
| --- | ---: | ---: |
| `gemma4-26b` | 39 | 4.53% |
| `qwen3.6-36b` | 41 | 4.77% |
| `llama-3.3-70b` | 31 | 3.60% |

These are **descriptive** rates. The study did not establish a statistically reliable between-model difference in disagreement rates.

### Held-out macro-F1 (all nine conditions)

| API alias | P0 (reference) | P1 | P2 |
| --- | ---: | ---: | ---: |
| `gemma4-26b` | 0.6908 | 0.6801 | 0.7002 |
| `qwen3.6-36b` | 0.7037 | 0.7199 | 0.6949 |
| `llama-3.3-70b` | 0.7789 | 0.7856 | 0.7880 |

The majority-class (always non-offensive) baseline achieved **72.09% accuracy**, **0.4189 macro-F1**, and **0 offensive recall**.

All **nine planned comparisons** (six within-model P1/P2-versus-P0 macro-F1 differences and three between-model disagreement-rate differences) were recomputed using 5,000 stratified paired bootstrap resamples and Bonferroni adjustment across nine comparisons. **The multiplicity correction was introduced post hoc**, after examining test results. None of the nine adjusted intervals excluded zero.

A notable reporting detail: for `llama-3.3-70b`, **P2 − P0 macro-F1 = +0.00909**, with an **unadjusted 95% confidence interval `[+0.00018, +0.01877]`**, but the **nine-comparison Bonferroni-adjusted interval `[-0.00353, +0.02315]`** includes zero. Therefore we do **not** claim a reliable corrected improvement. No best-performing prompt was preselected using test results.

The singleton sensitivity analysis removes **all 20 rows belonging to exact-text duplicate groups**, retaining 840 rows. Disagreement rates remain similar: 4.40% (Gemma), 4.64% (Qwen), 3.69% (Llama).

For complete numerical tables, statistical intervals, and limitations, read [`docs/verified_results_v2.md`](docs/verified_results_v2.md). Qualitative-case documentation is in [`docs/qualitative_case_provenance.md`](docs/qualitative_case_provenance.md).

## Reproduce the archived results (Ubuntu / WSL)

### A. Prerequisites

The reference environment was **Ubuntu under WSL, Python 3.12.3**, with **35 pinned Python package entries**. This combination passed an independent fresh-virtual-environment test on the researcher's machine. Other operating systems or Python versions have not been equivalently tested.

For Ubuntu 24.04 / WSL Ubuntu, install prerequisites as needed:

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip git
python3.12 --version
```

### B. Clone and create your own virtual environment

Replace the placeholders with the actual repository URL and directory **after the project has been published**:

```bash
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --disable-pip-version-check -r reproducibility/requirements-lock.txt
python -m pip check
```

`python -m pip check` should report `No broken requirements found.`. The dependency lock preserves the package versions from the reference environment; a fresh installation does not imply that future hosted model inference is identical. **Never copy an existing `.venv` directory between computers or commit it to Git.**

### C. Reproduce the public analysis

From the repository root with `.venv` activated:

```bash
python reproducibility/test_public_analysis.py
```

This single command:

1. Downloads the pinned TweetEval validation/test Parquet files when missing (**internet access required for the initial download**).
2. Verifies source-file fingerprints and recreates the exact pilot, main validation, and held-out test inputs.
3. Recreates the 36 pilot, 2,700 main, and 7,740 test prepared requests.
4. Loads the archived model prediction files from `artifacts/predictions/`.
5. Executes pilot/repeatability, validation/test, uncertainty, and nine-comparison analyses.
6. Generates the three final figure sets as PNG and PDF.
7. Exports the derived numerical tables and figures to `reproduced_outputs/`.

Successful output contains:

```text
PUBLIC_DATA_BOOTSTRAP_OK
PUBLIC_FIGURES_REPRODUCED_OK
PUBLIC_ANALYSIS_REPRODUCTION_OK
REPRODUCED_OUTPUTS_EXPORTED_OK
```

Outputs are written to:

```text
reproduced_outputs/
├── tables/       # 17 derived CSV result tables
└── figures/      # Three figures, each in PNG and PDF
```

**Important verification boundary:** The workflow was successfully tested in the original workspace using a separately installed environment and a temporary workspace with no original `data/` directory. The curated release directory has now passed reproduction using a fresh Python 3.12 environment, downloaded source data, and archived predictions. A separate clean Git clone test is still pending. The first run may download public dataset files; it is not fully offline without that cache. It makes **zero LLM inference calls**.

### D. Figures

The authoritative final figures are:

- `figure1_prompt_sensitivity`: fraction of posts with differing predictions, with dataset/class-composition context.
- `figure2_macro_f1`: held-out macro-F1 and **marginal** confidence intervals; overlapping/non-overlapping marginal intervals do not test the paired prompt differences.
- `figure3_recall_tradeoff_v2`: class-specific recall by model and prompt (the `_v2` file supersedes a misleading earlier plot).

The model labels in these final plots use the exact maKI API aliases, not unverified public checkpoint names.

## Optional: repeat live model inference

Reproducing the **saved analysis** needs no maKI credentials. Repeating model calls requires a personally authorized API key, access to `https://maki.uni-mannheim.de/v1`, and currently available model aliases. Historical scripts are included for transparency (`run_pilot.py`, `run_main.py`, `run_test.py`, `run_repeatability.py`). They were not independently validated as a single safe, automatic *fresh-inference* command for public use: inspect their expected paths and prerequisites before using them.

If you have authorized access, configure credentials **in your local shell**, never in a tracked source file:

```bash
export MAKI_API_KEY='YOUR_PERSONAL_API_KEY'
```

Do not post your key in issues, screenshots, commits, or logs. An identical API alias is insufficient to guarantee identical weights or responses across dates. Avoid overwriting archived predictions.

## Repository layout

```text
.
├── README.md
├── .gitignore
├── prompts.py
├── prepare_pilot.py
├── prepare_repeatability.py
├── run_*.py
├── analyze_*.py
├── plot_figure1.py
├── plot_figure2.py
├── plot_figure3_v2.py
├── artifacts/
│   └── predictions/             # Archived label/metadata records
├── figures/                    # Three final figures, PNG + PDF
├── docs/
│   ├── verified_results_v2.md
│   └── qualitative_case_provenance.md
└── reproducibility/
    ├── requirements-lock.txt
    ├── python-version.txt
    ├── bootstrap_public_data.py
    ├── test_public_analysis.py
    ├── sample_manifest_v1.json
    ├── test_text_suffix_manifest.json
    ├── tweeteval_source_manifest.json
    ├── model_provenance.md
    └── maki_model_metadata.json
```

Locally generated data, source caches, logs, secrets, virtual environments, and temporary outputs are excluded via `.gitignore`. The original tweet texts and full prompts are reconstructed from source data when authorized and necessary, rather than committed here.

## Limitations and responsible interpretation

- The study examines **one English social-media dataset**, three close prompt formulations, and three hosted model aliases. Results do not automatically generalize to other languages, tasks, prompts, or providers.
- Exact hosted model checkpoint/serving provenance is not independently known; model deployments may change across experiment dates.
- Temperature 0 does not mathematically guarantee deterministic output. A limited repeatability check observed no changes, but cannot rule out variation for every borderline post.
- One gold-label conflict exists among duplicate observations; class mix differs between validation and test. Exact-text singleton sensitivity is reported separately.
- The original sampling mechanism/seed could not be recovered; preserved row IDs make **historical sample reconstruction** reproducible, not the original random selection procedure.
- Possible pretraining exposure to TweetEval/OLID cannot be excluded.
- Qualitative examples are illustrative; the historical annotation process and number of independent reviewers are not fully documented.
- Multiplicity correction was post hoc and statistical uncertainty remains; **failure to detect a difference does not prove equivalence**.

## Attribution, data rights, and license

Dataset: CardiffNLP TweetEval `offensive`. The dataset and underlying tweet content may be subject to rights and conditions distinct from the research code. The dataset itself is **not bundled** here. Do not infer permission to redistribute underlying content, hosted model weights, or API services.

**Project contributor names and code-license choice must be confirmed by the team before public publication.** Until a license is chosen and added, do not assume general reuse rights for this code. A formal research-paper citation will be added after the final manuscript is complete.

## Reproduction status

- **Verified in the original project:** saved metrics, uncertainty tables, reconstructed inputs, fresh Python environment, and public-data + archived-prediction analysis (including three figure pairs).
- **Verified in the curated release directory:** fresh dependency installation, reconstruction, numerical analysis, 17 exported tables, and six figure files.
- **Still pending:** clean Git-clone test, final staged-file audit, contributor authorization, and artifact redistribution review.
- **Not guaranteed:** exact reproduction of fresh hosted LLM inference when the server checkpoint or infrastructure differs.
