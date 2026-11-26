# PromptShift v3 results (generated tables, no interpretation)

Generated 2026-11-26T18:24:38.050424+00:00. OLID n=860 (gold share 0.279); fresh n=400 (gold share 0.245). Bootstrap B=10000, seed 20261201. Holm across 9 primary tests, alpha 0.05.

## Gold labels (fresh)

Krippendorff alpha (A vs B, fresh): 0.776; raw agreement 0.912; 35 disagreements decided by C.
Agreement with OLID gold on 100 anchors: A accuracy 0.780 (kappa 0.429), B accuracy 0.770 (kappa 0.367). Rule fixed in advance: H2 interpretable only if both kappas ≥ 0.6: **NOT met — H2 is not interpretable as a contamination effect**.

## Primary tests

| Test | Estimate | 95% CI | p | p (Holm) | Significant | Size verdict | fresh / OLID value | n fresh / OLID |
|---|---:|---|---:|---:|---|---|---|---|
| H1_Gemma | -0.0018 | [-0.0181, +0.0149] | 0.5897 | 1.0000 | False | size inconclusive | 0.0242 / 0.0260 | 400 / 860 |
| H2_Gemma | +0.0777 | [+0.0285, +0.1257] | 0.9988 | 1.0000 | False | smaller than SESOI (lower bound above -SESOI) | 0.8118 / 0.7341 | 400 / 860 |
| H3_Gemma | -0.0900 | [-0.1142, -0.0667] | 0.0001 | 0.0009 | True | materially negative (upper bound below -SESOI) | PE_D 0.0417 / PE_W 0.1317 | 400 /  |
| H1_Qwen | +0.0048 | [-0.0123, +0.0224] | 0.3170 | 1.0000 | False | size inconclusive | 0.0350 / 0.0302 | 400 / 860 |
| H2_Qwen | +0.0615 | [+0.0052, +0.1151] | 0.9824 | 1.0000 | False | smaller than SESOI (lower bound above -SESOI) | 0.7907 / 0.7292 | 400 / 860 |
| H3_Qwen | -0.0317 | [-0.0492, -0.0158] | 0.0001 | 0.0009 | True | size inconclusive | PE_D 0.0833 / PE_W 0.1150 | 400 /  |
| H1_Llama | +0.0102 | [-0.0060, +0.0276] | 0.1145 | 0.6869 | False | size inconclusive | 0.0350 / 0.0248 | 400 / 860 |
| H2_Llama | +0.0648 | [+0.0227, +0.1082] | 0.9985 | 1.0000 | False | smaller than SESOI (lower bound above -SESOI) | 0.8497 / 0.7850 | 399 / 860 |
| H3_Llama | -0.0585 | [-0.0802, -0.0384] | 0.0001 | 0.0009 | True | materially negative (upper bound below -SESOI) | PE_D 0.0493 / PE_W 0.1078 | 399 /  |

## Metrics by dataset and condition

| Dataset | Model | Condition | n | Accuracy | Macro-F1 [95% CI] | Off. recall | Not-off. recall | Predicted share | Gold share |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| OLID | Gemma | W0 | 860 | 0.7919 | 0.6862 [0.6489, 0.7223] | 0.3792 | 0.9516 | 0.1407 | 0.2791 |
| OLID | Gemma | W1 | 860 | 0.7907 | 0.6838 [0.6455, 0.7202] | 0.3750 | 0.9516 | 0.1395 | 0.2791 |
| OLID | Gemma | W2 | 860 | 0.8000 | 0.6979 [0.6603, 0.7338] | 0.3917 | 0.9581 | 0.1395 | 0.2791 |
| OLID | Gemma | D0 | 860 | 0.8360 | 0.7817 [0.7492, 0.8131] | 0.6042 | 0.9258 | 0.2221 | 0.2791 |
| OLID | Gemma | D1 | 860 | 0.8337 | 0.7813 [0.7488, 0.8123] | 0.6167 | 0.9177 | 0.2314 | 0.2791 |
| OLID | Gemma | D2 | 860 | 0.8302 | 0.7736 [0.7403, 0.8058] | 0.5917 | 0.9226 | 0.2209 | 0.2791 |
| OLID | Gemma | R0 | 860 | 0.7930 | 0.6886 [0.6508, 0.7252] | 0.3833 | 0.9516 | 0.1419 | 0.2791 |
| OLID | Gemma | RD0 | 860 | 0.8326 | 0.7781 [0.7457, 0.8096] | 0.6042 | 0.9210 | 0.2256 | 0.2791 |
| OLID | Qwen | W0 | 860 | 0.8023 | 0.7014 [0.6642, 0.7375] | 0.3958 | 0.9597 | 0.1395 | 0.2791 |
| OLID | Qwen | W1 | 860 | 0.8105 | 0.7199 [0.6832, 0.7555] | 0.4333 | 0.9565 | 0.1523 | 0.2791 |
| OLID | Qwen | W2 | 860 | 0.8012 | 0.6913 [0.6529, 0.7283] | 0.3667 | 0.9694 | 0.1244 | 0.2791 |
| OLID | Qwen | D0 | 860 | 0.8302 | 0.7605 [0.7262, 0.7935] | 0.5208 | 0.9500 | 0.1814 | 0.2791 |
| OLID | Qwen | D1 | 860 | 0.8267 | 0.7534 [0.7187, 0.7877] | 0.5042 | 0.9516 | 0.1756 | 0.2791 |
| OLID | Qwen | D2 | 860 | 0.8233 | 0.7489 [0.7138, 0.7825] | 0.5000 | 0.9484 | 0.1767 | 0.2791 |
| OLID | Qwen | R0 | 860 | 0.8023 | 0.7014 [0.6635, 0.7375] | 0.3958 | 0.9597 | 0.1395 | 0.2791 |
| OLID | Qwen | RD0 | 860 | 0.8302 | 0.7605 [0.7264, 0.7942] | 0.5208 | 0.9500 | 0.1814 | 0.2791 |
| OLID | Llama | W0 | 860 | 0.8128 | 0.7812 [0.7519, 0.8094] | 0.7750 | 0.8274 | 0.3407 | 0.2791 |
| OLID | Llama | W1 | 860 | 0.8116 | 0.7814 [0.7520, 0.8102] | 0.7875 | 0.8210 | 0.3488 | 0.2791 |
| OLID | Llama | W2 | 860 | 0.8186 | 0.7868 [0.7575, 0.8157] | 0.7750 | 0.8355 | 0.3349 | 0.2791 |
| OLID | Llama | D0 | 860 | 0.8244 | 0.7884 [0.7581, 0.8176] | 0.7375 | 0.8581 | 0.3081 | 0.2791 |
| OLID | Llama | D1 | 860 | 0.8244 | 0.7874 [0.7575, 0.8172] | 0.7292 | 0.8613 | 0.3035 | 0.2791 |
| OLID | Llama | D2 | 860 | 0.8244 | 0.7848 [0.7536, 0.8151] | 0.7083 | 0.8694 | 0.2919 | 0.2791 |
| OLID | Llama | R0 | 860 | 0.8128 | 0.7812 [0.7519, 0.8099] | 0.7750 | 0.8274 | 0.3407 | 0.2791 |
| OLID | Llama | RD0 | 860 | 0.8244 | 0.7884 [0.7586, 0.8176] | 0.7375 | 0.8581 | 0.3081 | 0.2791 |
| fresh | Gemma | W0 | 400 | 0.8525 | 0.7515 [0.6951, 0.8040] | 0.4388 | 0.9868 | 0.1175 | 0.2450 |
| fresh | Gemma | W1 | 400 | 0.8600 | 0.7629 [0.7077, 0.8132] | 0.4490 | 0.9934 | 0.1150 | 0.2450 |
| fresh | Gemma | W2 | 400 | 0.8475 | 0.7374 [0.6785, 0.7905] | 0.4082 | 0.9901 | 0.1075 | 0.2450 |
| fresh | Gemma | D0 | 400 | 0.9075 | 0.8679 [0.8261, 0.9053] | 0.7347 | 0.9636 | 0.2075 | 0.2450 |
| fresh | Gemma | D1 | 400 | 0.9200 | 0.8844 [0.8445, 0.9203] | 0.7449 | 0.9768 | 0.2000 | 0.2450 |
| fresh | Gemma | D2 | 400 | 0.9075 | 0.8669 [0.8252, 0.9061] | 0.7245 | 0.9669 | 0.2025 | 0.2450 |
| fresh | Gemma | R0 | 400 | 0.8500 | 0.7459 [0.6883, 0.7990] | 0.4286 | 0.9868 | 0.1150 | 0.2450 |
| fresh | Gemma | RD0 | 400 | 0.9100 | 0.8720 [0.8311, 0.9103] | 0.7449 | 0.9636 | 0.2100 | 0.2450 |
| fresh | Qwen | W0 | 400 | 0.8575 | 0.7673 [0.7121, 0.8189] | 0.4796 | 0.9801 | 0.1325 | 0.2450 |
| fresh | Qwen | W1 | 400 | 0.8550 | 0.7691 [0.7147, 0.8192] | 0.5000 | 0.9702 | 0.1450 | 0.2450 |
| fresh | Qwen | W2 | 400 | 0.8475 | 0.7403 [0.6811, 0.7939] | 0.4184 | 0.9868 | 0.1125 | 0.2450 |
| fresh | Qwen | D0 | 400 | 0.8775 | 0.8112 [0.7611, 0.8568] | 0.5816 | 0.9735 | 0.1625 | 0.2450 |
| fresh | Qwen | D1 | 400 | 0.8925 | 0.8343 [0.7867, 0.8773] | 0.6122 | 0.9834 | 0.1625 | 0.2450 |
| fresh | Qwen | D2 | 400 | 0.8850 | 0.8220 [0.7719, 0.8664] | 0.5918 | 0.9801 | 0.1600 | 0.2450 |
| fresh | Qwen | R0 | 400 | 0.8575 | 0.7673 [0.7125, 0.8186] | 0.4796 | 0.9801 | 0.1325 | 0.2450 |
| fresh | Qwen | RD0 | 400 | 0.8775 | 0.8112 [0.7606, 0.8575] | 0.5816 | 0.9735 | 0.1625 | 0.2450 |
| fresh | Llama | W0 | 400 | 0.8575 | 0.8315 [0.7945, 0.8671] | 0.9490 | 0.8278 | 0.3625 | 0.2450 |
| fresh | Llama | W1 | 400 | 0.8475 | 0.8197 [0.7828, 0.8577] | 0.9286 | 0.8212 | 0.3625 | 0.2450 |
| fresh | Llama | W2 | 400 | 0.8725 | 0.8448 [0.8070, 0.8807] | 0.9184 | 0.8576 | 0.3325 | 0.2450 |
| fresh | Llama | D0 | 400 | 0.8950 | 0.8668 [0.8298, 0.9019] | 0.8878 | 0.8974 | 0.2950 | 0.2450 |
| fresh | Llama | D1 | 399 | 0.8897 | 0.8612 [0.8238, 0.8966] | 0.8878 | 0.8904 | 0.3008 | 0.2456 |
| fresh | Llama | D2 | 400 | 0.9025 | 0.8752 [0.8388, 0.9090] | 0.8878 | 0.9073 | 0.2875 | 0.2450 |
| fresh | Llama | R0 | 400 | 0.8575 | 0.8315 [0.7956, 0.8664] | 0.9490 | 0.8278 | 0.3625 | 0.2450 |
| fresh | Llama | RD0 | 400 | 0.8950 | 0.8668 [0.8304, 0.9020] | 0.8878 | 0.8974 | 0.2950 | 0.2450 |

## Secondary and exploratory measures

| Dataset | Model | Measure | Estimate | 95% CI | n | Note |
|---|---|---|---:|---|---:|---|
| OLID | Gemma | wording disagreement (mean pairwise W0-W2) | +0.0271 | [+0.0186, +0.0364] | 860 |  |
| OLID | Gemma | repeat disagreement W0 vs R0 | +0.0012 | [+0.0000, +0.0035] | 860 |  |
| OLID | Gemma | repeat disagreement D0 vs RD0 | +0.0058 | [+0.0012, +0.0116] | 860 |  |
| OLID | Gemma | any-flip rate across W0-W2 | +0.0407 | [+0.0279, +0.0547] | 860 |  |
| OLID | Gemma | H3 on OLID (descriptive, not confirmatory): PE_D - PE_W | -0.0849 | [-0.1016, -0.0694] | 860 | observed before freeze |
| OLID | Gemma | E2 confidence (W0 first-token prob): flipped - stable | -0.1193 | [-0.1723, -0.0711] | 860 | exploratory |
| fresh | Gemma | wording disagreement (mean pairwise W0-W2) | +0.0267 | [+0.0150, +0.0400] | 400 |  |
| fresh | Gemma | repeat disagreement W0 vs R0 | +0.0025 | [+0.0000, +0.0075] | 400 |  |
| fresh | Gemma | repeat disagreement D0 vs RD0 | +0.0075 | [+0.0000, +0.0175] | 400 |  |
| fresh | Gemma | any-flip rate across W0-W2 | +0.0400 | [+0.0225, +0.0600] | 400 |  |
| fresh | Gemma | E2 confidence (W0 first-token prob): flipped - stable | -0.0636 | [-0.1147, -0.0215] | 400 | exploratory |
| fresh - OLID | Gemma | E1 definition effect on macro-F1, fresh minus OLID | +0.0329 | [-0.0194, +0.0876] | 1260 | exploratory |
| fresh | Gemma | E4 any-flip rate: annotators disagreed - agreed | +0.0501 | [-0.0329, +0.1534] | 400 | exploratory; 35 posts with annotator disagreement |
| OLID | Qwen | wording disagreement (mean pairwise W0-W2) | +0.0302 | [+0.0209, +0.0395] | 860 |  |
| OLID | Qwen | repeat disagreement W0 vs R0 | +0.0000 | [+0.0000, +0.0000] | 860 |  |
| OLID | Qwen | repeat disagreement D0 vs RD0 | +0.0000 | [+0.0000, +0.0000] | 860 |  |
| OLID | Qwen | any-flip rate across W0-W2 | +0.0453 | [+0.0314, +0.0593] | 860 |  |
| OLID | Qwen | H3 on OLID (descriptive, not confirmatory): PE_D - PE_W | -0.0391 | [-0.0519, -0.0271] | 860 | observed before freeze |
| OLID | Qwen | E2 confidence (W0 first-token prob): flipped - stable | -0.2862 | [-0.3131, -0.2587] | 860 | exploratory |
| fresh | Qwen | wording disagreement (mean pairwise W0-W2) | +0.0350 | [+0.0217, +0.0500] | 400 |  |
| fresh | Qwen | repeat disagreement W0 vs R0 | +0.0000 | [+0.0000, +0.0000] | 400 |  |
| fresh | Qwen | repeat disagreement D0 vs RD0 | +0.0000 | [+0.0000, +0.0000] | 400 |  |
| fresh | Qwen | any-flip rate across W0-W2 | +0.0525 | [+0.0325, +0.0750] | 400 |  |
| fresh | Qwen | E2 confidence (W0 first-token prob): flipped - stable | -0.2728 | [-0.3237, -0.2178] | 400 | exploratory |
| fresh - OLID | Qwen | E1 definition effect on macro-F1, fresh minus OLID | +0.0136 | [-0.0238, +0.0517] | 1260 | exploratory |
| fresh | Qwen | E4 any-flip rate: annotators disagreed - agreed | +0.0051 | [-0.0630, +0.0963] | 400 | exploratory; 35 posts with annotator disagreement |
| OLID | Llama | wording disagreement (mean pairwise W0-W2) | +0.0248 | [+0.0163, +0.0333] | 860 |  |
| OLID | Llama | repeat disagreement W0 vs R0 | +0.0000 | [+0.0000, +0.0000] | 860 |  |
| OLID | Llama | repeat disagreement D0 vs RD0 | +0.0000 | [+0.0000, +0.0000] | 860 |  |
| OLID | Llama | any-flip rate across W0-W2 | +0.0372 | [+0.0256, +0.0500] | 860 |  |
| OLID | Llama | H3 on OLID (descriptive, not confirmatory): PE_D - PE_W | -0.0403 | [-0.0531, -0.0256] | 860 | observed before freeze |
| OLID | Llama | E2 confidence (W0 first-token prob): flipped - stable | -0.1396 | [-0.1947, -0.0888] | 860 | exploratory |
| fresh | Llama | wording disagreement (mean pairwise W0-W2) | +0.0350 | [+0.0217, +0.0500] | 400 |  |
| fresh | Llama | repeat disagreement W0 vs R0 | +0.0000 | [+0.0000, +0.0000] | 400 |  |
| fresh | Llama | repeat disagreement D0 vs RD0 | +0.0000 | [+0.0000, +0.0000] | 399 |  |
| fresh | Llama | any-flip rate across W0-W2 | +0.0525 | [+0.0325, +0.0750] | 400 |  |
| fresh | Llama | E2 confidence (W0 first-token prob): flipped - stable | -0.1047 | [-0.1727, -0.0459] | 400 | exploratory |
| fresh - OLID | Llama | E1 definition effect on macro-F1, fresh minus OLID | +0.0321 | [+0.0030, +0.0610] | 1259 | exploratory |
| fresh | Llama | E4 any-flip rate: annotators disagreed - agreed | +0.0677 | [-0.0290, +0.1820] | 400 | exploratory; 35 posts with annotator disagreement |
| OLID | Gemma vs Qwen | E3 between-model disagreement on W0 | +0.0593 | [+0.0430, +0.0756] | 860 | exploratory |
| OLID | Gemma vs Llama | E3 between-model disagreement on W0 | +0.2070 | [+0.1802, +0.2349] | 860 | exploratory |
| OLID | Qwen vs Llama | E3 between-model disagreement on W0 | +0.2058 | [+0.1791, +0.2326] | 860 | exploratory |
| fresh | Gemma vs Qwen | E3 between-model disagreement on W0 | +0.0700 | [+0.0475, +0.0975] | 400 | exploratory |
| fresh | Gemma vs Llama | E3 between-model disagreement on W0 | +0.2450 | [+0.2025, +0.2875] | 400 | exploratory |
| fresh | Qwen vs Llama | E3 between-model disagreement on W0 | +0.2300 | [+0.1900, +0.2725] | 400 | exploratory |

## Sensitivity analyses (primary estimands; Holm within each analysis)

| Analysis | Test | Estimate | 95% CI | p (Holm) | Size verdict | n fresh / OLID |
|---|---|---:|---|---:|---|---|
| invalid answers counted as disagreement/error | H1_Gemma | -0.0018 | [-0.0181, +0.0149] | 1.0000 | size inconclusive | 400 / 860 |
| invalid answers counted as disagreement/error | H2_Gemma | +0.0777 | [+0.0285, +0.1257] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 400 / 860 |
| invalid answers counted as disagreement/error | H3_Gemma | -0.0900 | [-0.1142, -0.0667] | 0.0009 | materially negative (upper bound below -SESOI) | 400 /  |
| invalid answers counted as disagreement/error | H1_Qwen | +0.0048 | [-0.0123, +0.0224] | 1.0000 | size inconclusive | 400 / 860 |
| invalid answers counted as disagreement/error | H2_Qwen | +0.0615 | [+0.0052, +0.1151] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 400 / 860 |
| invalid answers counted as disagreement/error | H3_Qwen | -0.0317 | [-0.0492, -0.0158] | 0.0009 | size inconclusive | 400 /  |
| invalid answers counted as disagreement/error | H1_Llama | +0.0102 | [-0.0060, +0.0276] | 0.6869 | size inconclusive | 400 / 860 |
| invalid answers counted as disagreement/error | H2_Llama | +0.0645 | [+0.0216, +0.1087] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 400 / 860 |
| invalid answers counted as disagreement/error | H3_Llama | -0.0575 | [-0.0800, -0.0367] | 0.0009 | materially negative (upper bound below -SESOI) | 400 /  |
| OLID exact duplicates removed | H1_Gemma | -0.0008 | [-0.0171, +0.0156] | 1.0000 | size inconclusive | 400 / 840 |
| OLID exact duplicates removed | H2_Gemma | +0.0787 | [+0.0288, +0.1265] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 400 / 840 |
| OLID exact duplicates removed | H3_Gemma | -0.0900 | [-0.1142, -0.0667] | 0.0009 | materially negative (upper bound below -SESOI) | 400 /  |
| OLID exact duplicates removed | H1_Qwen | +0.0056 | [-0.0117, +0.0229] | 1.0000 | size inconclusive | 400 / 840 |
| OLID exact duplicates removed | H2_Qwen | +0.0633 | [+0.0078, +0.1173] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 400 / 840 |
| OLID exact duplicates removed | H3_Qwen | -0.0317 | [-0.0492, -0.0158] | 0.0009 | size inconclusive | 400 /  |
| OLID exact duplicates removed | H1_Llama | +0.0096 | [-0.0068, +0.0270] | 0.8105 | size inconclusive | 400 / 840 |
| OLID exact duplicates removed | H2_Llama | +0.0682 | [+0.0247, +0.1111] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 399 / 840 |
| OLID exact duplicates removed | H3_Llama | -0.0585 | [-0.0802, -0.0384] | 0.0009 | materially negative (upper bound below -SESOI) | 399 /  |
| fresh posts with annotator disagreement removed | H1_Gemma | -0.0050 | [-0.0209, +0.0116] | 1.0000 | smaller than SESOI (upper bound below SESOI) | 365 / 860 |
| fresh posts with annotator disagreement removed | H2_Gemma | +0.0947 | [+0.0439, +0.1421] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 365 / 860 |
| fresh posts with annotator disagreement removed | H3_Gemma | -0.0849 | [-0.1087, -0.0621] | 0.0009 | materially negative (upper bound below -SESOI) | 365 /  |
| fresh posts with annotator disagreement removed | H1_Qwen | +0.0045 | [-0.0127, +0.0226] | 1.0000 | size inconclusive | 365 / 860 |
| fresh posts with annotator disagreement removed | H2_Qwen | +0.0825 | [+0.0272, +0.1377] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 365 / 860 |
| fresh posts with annotator disagreement removed | H3_Qwen | -0.0365 | [-0.0548, -0.0192] | 0.0009 | materially negative (upper bound below -SESOI) | 365 /  |
| fresh posts with annotator disagreement removed | H1_Llama | +0.0062 | [-0.0099, +0.0237] | 1.0000 | size inconclusive | 365 / 860 |
| fresh posts with annotator disagreement removed | H2_Llama | +0.0923 | [+0.0511, +0.1346] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 364 / 860 |
| fresh posts with annotator disagreement removed | H3_Llama | -0.0504 | [-0.0723, -0.0302] | 0.0009 | materially negative (upper bound below -SESOI) | 364 /  |
| OLID subsample matched on post length | H1_Gemma | +0.0074 | [-0.0124, +0.0266] | 1.0000 | size inconclusive | 400 / 239 |
| OLID subsample matched on post length | H2_Gemma | +0.0685 | [+0.0000, +0.1399] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 400 / 239 |
| OLID subsample matched on post length | H3_Gemma | -0.0900 | [-0.1142, -0.0667] | 0.0009 | materially negative (upper bound below -SESOI) | 400 /  |
| OLID subsample matched on post length | H1_Qwen | +0.0015 | [-0.0235, +0.0244] | 1.0000 | size inconclusive | 400 / 239 |
| OLID subsample matched on post length | H2_Qwen | +0.0470 | [-0.0261, +0.1215] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 400 / 239 |
| OLID subsample matched on post length | H3_Qwen | -0.0317 | [-0.0492, -0.0158] | 0.0009 | size inconclusive | 400 /  |
| OLID subsample matched on post length | H1_Llama | +0.0043 | [-0.0185, +0.0266] | 1.0000 | size inconclusive | 400 / 239 |
| OLID subsample matched on post length | H2_Llama | +0.0609 | [+0.0015, +0.1215] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 399 / 239 |
| OLID subsample matched on post length | H3_Llama | -0.0585 | [-0.0802, -0.0384] | 0.0009 | materially negative (upper bound below -SESOI) | 399 /  |
| fresh gold = annotator A only (no adjudication) | H1_Gemma | -0.0018 | [-0.0181, +0.0149] | 1.0000 | size inconclusive | 400 / 860 |
| fresh gold = annotator A only (no adjudication) | H2_Gemma | +0.0322 | [-0.0190, +0.0814] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 400 / 860 |
| fresh gold = annotator A only (no adjudication) | H3_Gemma | -0.0900 | [-0.1150, -0.0658] | 0.0009 | materially negative (upper bound below -SESOI) | 400 /  |
| fresh gold = annotator A only (no adjudication) | H1_Qwen | +0.0048 | [-0.0123, +0.0224] | 1.0000 | size inconclusive | 400 / 860 |
| fresh gold = annotator A only (no adjudication) | H2_Qwen | +0.0036 | [-0.0522, +0.0578] | 1.0000 | size inconclusive | 400 / 860 |
| fresh gold = annotator A only (no adjudication) | H3_Qwen | -0.0317 | [-0.0500, -0.0150] | 0.0009 | size inconclusive | 400 /  |
| fresh gold = annotator A only (no adjudication) | H1_Llama | +0.0102 | [-0.0060, +0.0276] | 0.8014 | size inconclusive | 400 / 860 |
| fresh gold = annotator A only (no adjudication) | H2_Llama | +0.0606 | [+0.0163, +0.1049] | 1.0000 | smaller than SESOI (lower bound above -SESOI) | 399 / 860 |
| fresh gold = annotator A only (no adjudication) | H3_Llama | -0.0317 | [-0.0668, +0.0234] | 0.9053 | size inconclusive | 399 /  |
