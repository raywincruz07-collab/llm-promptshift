# Contamination evidence battery (generated tables, no interpretation)

A probe that finds nothing means "no evidence detected", not "not in the training data".
Groups present: olid_train, olid_test, fresh.

## P1 Guided minus general completion (ROUGE-L)

| Group | Model | n | Mean diff [95% CI] |
|---|---|---:|---|
| fresh | Gemma | 324 | -0.014 [-0.023, -0.005] |
| fresh | Llama | 324 | +0.020 [+0.013, +0.028] |
| fresh | Qwen | 324 | -0.001 [-0.011, +0.010] |
| olid_test | Gemma | 200 | +0.001 [-0.014, +0.017] |
| olid_test | Llama | 200 | +0.022 [+0.010, +0.036] |
| olid_test | Qwen | 200 | +0.011 [-0.004, +0.025] |
| olid_train | Gemma | 200 | -0.012 [-0.027, +0.003] |
| olid_train | Llama | 200 | +0.029 [+0.016, +0.044] |
| olid_train | Qwen | 200 | +0.017 [+0.001, +0.032] |

P1 group contrasts (independent bootstrap, difference in means):

| Model | Contrast | Difference [95% CI] |
|---|---|---|
| Gemma | olid_test − olid_train | +0.013 [-0.008, +0.035] |
| Gemma | olid_test − fresh | +0.015 [-0.003, +0.033] |
| Gemma | olid_train − fresh | +0.002 [-0.015, +0.020] |
| Qwen | olid_test − olid_train | -0.006 [-0.027, +0.015] |
| Qwen | olid_test − fresh | +0.012 [-0.006, +0.029] |
| Qwen | olid_train − fresh | +0.017 [-0.001, +0.036] |
| Llama | olid_test − olid_train | -0.007 [-0.027, +0.012] |
| Llama | olid_test − fresh | +0.002 [-0.013, +0.018] |
| Llama | olid_train − fresh | +0.009 [-0.007, +0.025] |

## P3 Dataset-cue label probe (accuracy with OLID cue minus matched neutral prompt)

| Group | Model | n | Acc neutral | Acc cue | Cue − neutral [95% CI] |
|---|---|---:|---:|---:|---|
| fresh | Gemma | 324 | 0.858 | 0.849 | -0.009 [-0.049, +0.031] |
| fresh | Llama | 324 | 0.731 | 0.762 | +0.031 [-0.006, +0.068] |
| fresh | Qwen | 324 | 0.815 | 0.827 | +0.012 [-0.012, +0.037] |
| olid_test | Gemma | 200 | 0.745 | 0.750 | +0.005 [-0.040, +0.050] |
| olid_test | Llama | 200 | 0.740 | 0.760 | +0.020 [-0.030, +0.070] |
| olid_test | Qwen | 200 | 0.755 | 0.760 | +0.005 [-0.030, +0.040] |
| olid_train | Gemma | 200 | 0.675 | 0.715 | +0.040 [-0.005, +0.090] |
| olid_train | Llama | 200 | 0.575 | 0.610 | +0.035 [-0.015, +0.090] |
| olid_train | Qwen | 200 | 0.730 | 0.740 | +0.010 [-0.035, +0.055] |

P3 group contrasts (independent bootstrap, difference in means):

| Model | Contrast | Difference [95% CI] |
|---|---|---|
| Gemma | olid_test − olid_train | -0.035 [-0.100, +0.030] |
| Gemma | olid_test − fresh | +0.014 [-0.045, +0.074] |
| Gemma | olid_train − fresh | +0.049 [-0.012, +0.112] |
| Qwen | olid_test − olid_train | -0.005 [-0.060, +0.050] |
| Qwen | olid_test − fresh | -0.007 [-0.050, +0.037] |
| Qwen | olid_train − fresh | -0.002 [-0.053, +0.047] |
| Llama | olid_test − olid_train | -0.015 [-0.085, +0.055] |
| Llama | olid_test − fresh | -0.011 [-0.072, +0.050] |
| Llama | olid_train − fresh | +0.004 [-0.061, +0.068] |

## P4 Min-K% Prob (k = 20%; higher = more familiar)

API failures per model: Gemma 129/724, Qwen 0/724, Llama 0/724.
Excluded (more than 10% of items failed, so the scored items are not a random subset): Gemma.

| Model | Comparison | AUC [95% CI] | Note |
|---|---|---|---|
| Qwen | olid_test vs olid_train | 0.413 [0.357, 0.467] | same era; no time confound |
| Qwen | olid_test vs fresh | 0.359 [0.312, 0.407] | time-shift confound (Duan et al. 2024): suggestive only |
| Qwen | olid_train vs fresh | 0.444 [0.394, 0.497] | time-shift confound: suggestive only |
| Llama | olid_test vs olid_train | 0.511 [0.454, 0.567] | same era; no time confound |
| Llama | olid_test vs fresh | 0.461 [0.412, 0.512] | time-shift confound (Duan et al. 2024): suggestive only |
| Llama | olid_train vs fresh | 0.449 [0.400, 0.498] | time-shift confound: suggestive only |

## P5 Data Contamination Quiz (pick the original among 4; chance = 0.25)

| Group | Model | n | Accuracy [95% CI] | Binomial p (> 0.25) | Picks A/B/C/D/invalid |
|---|---|---:|---|---:|---|
| fresh | Gemma | 322 | 0.528 [0.472, 0.584] | 0.0000 | 227/35/43/15/2 |
| fresh | Llama | 322 | 0.537 [0.481, 0.593] | 0.0000 | 194/30/43/50/5 |
| fresh | Qwen | 322 | 0.602 [0.550, 0.655] | 0.0000 | 71/52/50/49/100 |
| fresh | Qwen (256-token answers) | 322 | 0.702 [0.652, 0.752] | 0.0000 | 89/82/65/85/1 |
| olid_test | Gemma | 196 | 0.439 [0.367, 0.505] | 0.0000 | 151/18/16/11/0 |
| olid_test | Llama | 196 | 0.566 [0.495, 0.633] | 0.0000 | 111/16/27/39/3 |
| olid_test | Qwen | 196 | 0.556 [0.485, 0.628] | 0.0000 | 45/21/29/35/66 |
| olid_test | Qwen (256-token answers) | 196 | 0.668 [0.602, 0.735] | 0.0000 | 59/31/46/60/0 |
| olid_train | Gemma | 195 | 0.441 [0.374, 0.513] | 0.0000 | 154/18/13/10/0 |
| olid_train | Llama | 195 | 0.518 [0.446, 0.590] | 0.0000 | 133/13/22/25/2 |
| olid_train | Qwen | 195 | 0.672 [0.605, 0.738] | 0.0000 | 46/36/41/33/39 |
| olid_train | Qwen (256-token answers) | 195 | 0.749 [0.687, 0.805] | 0.0000 | 57/43/51/44/0 |

P5 diagnostics: accuracy when the original is at position A/B/C/D (a model that favours one letter scores high only there), and accuracy among valid answers.

| Group | Model | Acc if A | Acc if B | Acc if C | Acc if D | Acc among valid |
|---|---|---:|---:|---:|---:|---:|
| fresh | Gemma | 0.99 (n=78) | 0.45 (n=77) | 0.49 (n=88) | 0.19 (n=79) | 0.531 |
| fresh | Llama | 0.88 (n=78) | 0.27 (n=77) | 0.44 (n=88) | 0.56 (n=79) | 0.546 |
| fresh | Qwen | 0.69 (n=78) | 0.57 (n=77) | 0.53 (n=88) | 0.62 (n=79) | 0.874 |
| fresh | Qwen (256-token answers) | 0.76 (n=78) | 0.70 (n=77) | 0.60 (n=88) | 0.76 (n=79) | 0.704 |
| olid_test | Gemma | 1.00 (n=42) | 0.35 (n=49) | 0.33 (n=48) | 0.19 (n=57) | 0.439 |
| olid_test | Llama | 0.90 (n=42) | 0.27 (n=49) | 0.52 (n=48) | 0.61 (n=57) | 0.575 |
| olid_test | Qwen | 0.67 (n=42) | 0.37 (n=49) | 0.58 (n=48) | 0.61 (n=57) | 0.838 |
| olid_test | Qwen (256-token answers) | 0.76 (n=42) | 0.45 (n=49) | 0.71 (n=48) | 0.75 (n=57) | 0.668 |
| olid_train | Gemma | 1.00 (n=45) | 0.31 (n=58) | 0.27 (n=48) | 0.23 (n=44) | 0.441 |
| olid_train | Llama | 1.00 (n=45) | 0.22 (n=58) | 0.44 (n=48) | 0.50 (n=44) | 0.523 |
| olid_train | Qwen | 0.78 (n=45) | 0.50 (n=58) | 0.77 (n=48) | 0.68 (n=44) | 0.840 |
| olid_train | Qwen (256-token answers) | 0.82 (n=45) | 0.59 (n=58) | 0.83 (n=48) | 0.80 (n=44) | 0.749 |

P5 group contrasts (independent bootstrap, difference in means):

| Model | Contrast | Difference [95% CI] |
|---|---|---|
| Gemma | olid_test − olid_train | -0.002 [-0.104, +0.100] |
| Gemma | olid_test − fresh | -0.089 [-0.177, -0.001] |
| Gemma | olid_train − fresh | -0.087 [-0.177, +0.001] |
| Qwen | olid_test − olid_train | -0.116 [-0.213, -0.018] |
| Qwen | olid_test − fresh | -0.046 [-0.134, +0.040] |
| Qwen | olid_train − fresh | +0.069 [-0.016, +0.153] |
| Llama | olid_test − olid_train | +0.048 [-0.054, +0.146] |
| Llama | olid_test − fresh | +0.029 [-0.059, +0.117] |
| Llama | olid_train − fresh | -0.019 [-0.110, +0.070] |
| Qwen (256-token answers) | olid_test − olid_train | -0.080 [-0.167, +0.007] |
| Qwen (256-token answers) | olid_test − fresh | -0.033 [-0.115, +0.050] |
| Qwen (256-token answers) | olid_train − fresh | +0.047 [-0.033, +0.124] |

Quiz items with 3 valid paraphrases: olid_train 195/200, olid_test 196/200, fresh 322/324

6 completions with ROUGE-L ≥ 0.7 listed in contam_review_CONTAINS_TWEET_TEXT.csv (gitignored) for manual review.
