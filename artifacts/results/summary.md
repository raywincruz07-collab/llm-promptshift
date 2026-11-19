# Results (generated tables, no interpretation)

Generated 2026-11-19T11:02:46.101653+00:00 from predictions sha256 `bd22fbf2be4bf1fc…`.
Sample `main`, 860 posts; bootstrap B=10000, seed 20261124.
Primary H2 metric: `macro_f1`. Holm correction across 6 primary tests, alpha 0.05.

## Compliance

Total requests 20640, compliant 20640.

## Primary tests

| Test | Estimate | 95% CI | One-sided 95% bound | p | p (Holm) | Significant | Size verdict | n |
|---|---:|---|---:|---:|---:|---|---|---:|
| H1_Gemma | +0.0260 | [+0.0178, +0.0349] | 0.0190 | 0.0001 | 0.0006 | True | materially positive (lower bound above SESOI) | 860 |
| H1_Qwen | +0.0302 | [+0.0217, +0.0395] | 0.0225 | 0.0001 | 0.0006 | True | materially positive (lower bound above SESOI) | 860 |
| H1_Llama | +0.0248 | [+0.0163, +0.0333] | 0.0178 | 0.0001 | 0.0006 | True | materially positive (lower bound above SESOI) | 860 |
| H2_Gemma | +0.0896 | [+0.0637, +0.1172] |  | 0.0002 | 0.0006 | True | materially positive | 860 |
| H2_Qwen | +0.0501 | [+0.0305, +0.0713] |  | 0.0002 | 0.0006 | True | materially positive | 860 |
| H2_Llama | +0.0037 | [-0.0119, +0.0196] |  | 0.6371 | 0.6371 | False | practically equivalent (CI within ±SESOI) | 860 |

## Baselines (no model)

| Baseline | Accuracy | Macro-F1 | Off. recall | Not-off. recall | Predicted prevalence |
|---|---:|---:|---:|---:|---:|
| majority (always NOT_OFFENSIVE) | 0.7209 | 0.4189 | 0.0000 | 1.0000 | 0.0000 |
| random, weighted by class shares (expected) | 0.5976 | 0.5000 | 0.2791 | 0.7209 | 0.2791 |

## Metrics by condition

| Model | Condition | n | Accuracy | Macro-F1 [95% CI] | Off. recall | Not-off. recall | Predicted prevalence [95% CI] |
|---|---|---:|---:|---|---:|---:|---|
| Gemma | W0 | 860 | 0.7919 | 0.6862 [0.6491, 0.7234] | 0.3792 | 0.9516 | 0.1407 [0.1198, 0.1616] |
| Gemma | W1 | 860 | 0.7907 | 0.6838 [0.6470, 0.7200] | 0.3750 | 0.9516 | 0.1395 [0.1198, 0.1605] |
| Gemma | W2 | 860 | 0.8000 | 0.6979 [0.6606, 0.7343] | 0.3917 | 0.9581 | 0.1395 [0.1198, 0.1605] |
| Gemma | D0 | 860 | 0.8360 | 0.7817 [0.7490, 0.8125] | 0.6042 | 0.9258 | 0.2221 [0.1988, 0.2442] |
| Gemma | D1 | 860 | 0.8337 | 0.7813 [0.7496, 0.8122] | 0.6167 | 0.9177 | 0.2314 [0.2081, 0.2547] |
| Gemma | D2 | 860 | 0.8302 | 0.7736 [0.7400, 0.8052] | 0.5917 | 0.9226 | 0.2209 [0.1977, 0.2442] |
| Gemma | R0 | 860 | 0.7930 | 0.6886 [0.6502, 0.7258] | 0.3833 | 0.9516 | 0.1419 [0.1209, 0.1640] |
| Gemma | RD0 | 860 | 0.8326 | 0.7781 [0.7448, 0.8093] | 0.6042 | 0.9210 | 0.2256 [0.2023, 0.2488] |
| Qwen | W0 | 860 | 0.8023 | 0.7014 [0.6642, 0.7376] | 0.3958 | 0.9597 | 0.1395 [0.1198, 0.1605] |
| Qwen | W1 | 860 | 0.8105 | 0.7199 [0.6827, 0.7558] | 0.4333 | 0.9565 | 0.1523 [0.1314, 0.1733] |
| Qwen | W2 | 860 | 0.8012 | 0.6913 [0.6539, 0.7285] | 0.3667 | 0.9694 | 0.1244 [0.1047, 0.1442] |
| Qwen | D0 | 860 | 0.8302 | 0.7605 [0.7254, 0.7931] | 0.5208 | 0.9500 | 0.1814 [0.1593, 0.2035] |
| Qwen | D1 | 860 | 0.8267 | 0.7534 [0.7177, 0.7870] | 0.5042 | 0.9516 | 0.1756 [0.1547, 0.1977] |
| Qwen | D2 | 860 | 0.8233 | 0.7489 [0.7130, 0.7825] | 0.5000 | 0.9484 | 0.1767 [0.1558, 0.1977] |
| Qwen | R0 | 860 | 0.8023 | 0.7014 [0.6636, 0.7381] | 0.3958 | 0.9597 | 0.1395 [0.1198, 0.1605] |
| Qwen | RD0 | 860 | 0.8302 | 0.7605 [0.7265, 0.7944] | 0.5208 | 0.9500 | 0.1814 [0.1605, 0.2035] |
| Llama | W0 | 860 | 0.8128 | 0.7812 [0.7515, 0.8091] | 0.7750 | 0.8274 | 0.3407 [0.3140, 0.3674] |
| Llama | W1 | 860 | 0.8116 | 0.7814 [0.7525, 0.8097] | 0.7875 | 0.8210 | 0.3488 [0.3233, 0.3744] |
| Llama | W2 | 860 | 0.8186 | 0.7868 [0.7572, 0.8153] | 0.7750 | 0.8355 | 0.3349 [0.3093, 0.3605] |
| Llama | D0 | 860 | 0.8244 | 0.7884 [0.7586, 0.8176] | 0.7375 | 0.8581 | 0.3081 [0.2826, 0.3337] |
| Llama | D1 | 860 | 0.8244 | 0.7874 [0.7571, 0.8177] | 0.7292 | 0.8613 | 0.3035 [0.2779, 0.3291] |
| Llama | D2 | 860 | 0.8244 | 0.7848 [0.7540, 0.8148] | 0.7083 | 0.8694 | 0.2919 [0.2674, 0.3174] |
| Llama | R0 | 860 | 0.8128 | 0.7812 [0.7513, 0.8095] | 0.7750 | 0.8274 | 0.3407 [0.3151, 0.3663] |
| Llama | RD0 | 860 | 0.8244 | 0.7884 [0.7585, 0.8176] | 0.7375 | 0.8581 | 0.3081 [0.2826, 0.3326] |

## Secondary measures

| Model | Measure | Estimate | 95% CI | n |
|---|---|---:|---|---:|
| Gemma | wording disagreement, no definition (mean pairwise W0-W2) | +0.0271 | [+0.0186, +0.0364] | 860 |
| Gemma | wording disagreement, definition (mean pairwise D0-D2) | +0.0233 | [+0.0155, +0.0318] | 860 |
| Gemma | repeat disagreement W0 vs R0 | +0.0012 | [+0.0000, +0.0035] | 860 |
| Gemma | repeat disagreement D0 vs RD0 | +0.0058 | [+0.0012, +0.0116] | 860 |
| Gemma | H3: (wording-repeat with definition) minus (without) | -0.0085 | [-0.0198, +0.0027] | 860 |
| Gemma | legacy any-flip rate across W0-W2 | +0.0407 | [+0.0279, +0.0547] | 860 |
| Gemma | prevalence change: definition minus none (mean over wordings) | +0.0849 | [+0.0690, +0.1016] | 860 |
| Gemma | offensive_recall: definition minus none | +0.2222 | [+0.1750, +0.2709] | 860 |
| Gemma | not_offensive_recall: definition minus none | -0.0317 | [-0.0446, -0.0199] | 860 |
| Gemma | macro_f1: definition minus none | +0.0896 | [+0.0633, +0.1180] | 860 |
| Gemma | accuracy: definition minus none | +0.0391 | [+0.0229, +0.0558] | 860 |
| Gemma | Krippendorff alpha across wordings, no definition | +0.8873 |  | 860 |
| Gemma | Krippendorff alpha across wordings, definition | +0.9333 |  | 860 |
| Qwen | wording disagreement, no definition (mean pairwise W0-W2) | +0.0302 | [+0.0217, +0.0395] | 860 |
| Qwen | wording disagreement, definition (mean pairwise D0-D2) | +0.0318 | [+0.0225, +0.0411] | 860 |
| Qwen | repeat disagreement W0 vs R0 | +0.0000 | [+0.0000, +0.0000] | 860 |
| Qwen | repeat disagreement D0 vs RD0 | +0.0000 | [+0.0000, +0.0000] | 860 |
| Qwen | H3: (wording-repeat with definition) minus (without) | +0.0016 | [-0.0101, +0.0132] | 860 |
| Qwen | legacy any-flip rate across W0-W2 | +0.0453 | [+0.0326, +0.0593] | 860 |
| Qwen | prevalence change: definition minus none (mean over wordings) | +0.0391 | [+0.0275, +0.0519] | 860 |
| Qwen | offensive_recall: definition minus none | +0.1097 | [+0.0750, +0.1472] | 860 |
| Qwen | not_offensive_recall: definition minus none | -0.0118 | [-0.0220, -0.0022] | 860 |
| Qwen | macro_f1: definition minus none | +0.0501 | [+0.0299, +0.0716] | 860 |
| Qwen | accuracy: definition minus none | +0.0221 | [+0.0101, +0.0349] | 860 |
| Qwen | Krippendorff alpha across wordings, no definition | +0.8736 |  | 860 |
| Qwen | Krippendorff alpha across wordings, definition | +0.8914 |  | 860 |
| Llama | wording disagreement, no definition (mean pairwise W0-W2) | +0.0248 | [+0.0171, +0.0333] | 860 |
| Llama | wording disagreement, definition (mean pairwise D0-D2) | +0.0171 | [+0.0101, +0.0248] | 860 |
| Llama | repeat disagreement W0 vs R0 | +0.0000 | [+0.0000, +0.0000] | 860 |
| Llama | repeat disagreement D0 vs RD0 | +0.0000 | [+0.0000, +0.0000] | 860 |
| Llama | H3: (wording-repeat with definition) minus (without) | -0.0078 | [-0.0178, +0.0023] | 860 |
| Llama | legacy any-flip rate across W0-W2 | +0.0372 | [+0.0256, +0.0500] | 860 |
| Llama | prevalence change: definition minus none (mean over wordings) | -0.0403 | [-0.0543, -0.0271] | 860 |
| Llama | offensive_recall: definition minus none | -0.0542 | [-0.0833, -0.0278] | 860 |
| Llama | not_offensive_recall: definition minus none | +0.0349 | [+0.0199, +0.0511] | 860 |
| Llama | macro_f1: definition minus none | +0.0037 | [-0.0123, +0.0190] | 860 |
| Llama | accuracy: definition minus none | +0.0101 | [-0.0039, +0.0236] | 860 |
| Llama | Krippendorff alpha across wordings, no definition | +0.9449 |  | 860 |
| Llama | Krippendorff alpha across wordings, definition | +0.9595 |  | 860 |
| Gemma vs Qwen | between-model disagreement on W0 | +0.0593 | [+0.0442, +0.0756] | 860 |
| Gemma vs Llama | between-model disagreement on W0 | +0.2070 | [+0.1814, +0.2337] | 860 |
| Qwen vs Llama | between-model disagreement on W0 | +0.2058 | [+0.1802, +0.2314] | 860 |

## Sensitivity analyses (primary estimands)

| Analysis | Test | Estimate | 95% CI | p (Holm within analysis) | n |
|---|---|---:|---|---:|---:|
| invalid answers counted as errors | H1_Gemma | +0.0260 | [+0.0178, +0.0349] | 0.0006 | 860 |
| invalid answers counted as errors | H1_Qwen | +0.0302 | [+0.0217, +0.0395] | 0.0006 | 860 |
| invalid answers counted as errors | H1_Llama | +0.0248 | [+0.0163, +0.0333] | 0.0006 | 860 |
| invalid answers counted as errors | H2_Gemma | +0.0896 | [+0.0637, +0.1172] | 0.0006 | 860 |
| invalid answers counted as errors | H2_Qwen | +0.0501 | [+0.0305, +0.0713] | 0.0006 | 860 |
| invalid answers counted as errors | H2_Llama | +0.0037 | [-0.0119, +0.0196] | 0.6371 | 860 |
| exact-duplicate posts removed | H1_Gemma | +0.0250 | [+0.0171, +0.0337] | 0.0006 | 840 |
| exact-duplicate posts removed | H1_Qwen | +0.0294 | [+0.0206, +0.0389] | 0.0006 | 840 |
| exact-duplicate posts removed | H1_Llama | +0.0254 | [+0.0175, +0.0349] | 0.0006 | 840 |
| exact-duplicate posts removed | H2_Gemma | +0.0835 | [+0.0571, +0.1105] | 0.0006 | 840 |
| exact-duplicate posts removed | H2_Qwen | +0.0512 | [+0.0309, +0.0732] | 0.0006 | 840 |
| exact-duplicate posts removed | H2_Llama | +0.0037 | [-0.0121, +0.0192] | 0.6475 | 840 |
