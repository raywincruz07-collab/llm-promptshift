# Verified Experimental Results — IS6170

## Research Question

How does semantically equivalent prompt wording affect the consistency and accuracy of LLM offensive-language classification across three models?

## Experimental Setup

- Dataset: TweetEval Offensive.
- Held-out test: 860 posts (620 non-offensive, 240 offensive).
- Three model aliases and three zero-shot prompt variants.
- Inference settings: temperature 0, maximum 16 generated tokens.
- Reference prompt: P0.
- Primary consistency outcome: share of posts with at least one differing prediction across P0, P1 and P2.
- Performance outcomes: macro-F1, accuracy and class recall.

## All Held-Out Performance Results

All nine model-prompt conditions are reported. No prompt was selected as the winner based on held-out test scores.

| Model alias | Prompt | Accuracy | Macro-F1 | Offensive recall | Non-offensive recall |
|---|---|---:|---:|---:|---:|
| Majority-class baseline | - | 0.7209 | 0.4189 | 0.0000 | 1.0000 |
| gemma4-26b | P0 | 0.7953 | 0.6908 | 0.3833 | 0.9548 |
| gemma4-26b | P1 | 0.7895 | 0.6801 | 0.3667 | 0.9532 |
| gemma4-26b | P2 | 0.8012 | 0.7002 | 0.3958 | 0.9581 |
| qwen3.6-36b | P0 | 0.8047 | 0.7037 | 0.3958 | 0.9629 |
| qwen3.6-36b | P1 | 0.8105 | 0.7199 | 0.4333 | 0.9565 |
| qwen3.6-36b | P2 | 0.8035 | 0.6949 | 0.3708 | 0.9710 |
| llama-3.3-70b | P0 | 0.8105 | 0.7789 | 0.7750 | 0.8242 |
| llama-3.3-70b | P1 | 0.8151 | 0.7856 | 0.7958 | 0.8226 |
| llama-3.3-70b | P2 | 0.8198 | 0.7880 | 0.7750 | 0.8371 |

## Prompt Sensitivity

| Model alias | Posts with any flip | Flip rate |
|---|---:|---:|
| gemma4-26b | 39/860 | 4.53% |
| qwen3.6-36b | 41/860 | 4.77% |
| llama-3.3-70b | 31/860 | 3.60% |

These are descriptive within-model disagreement rates. They are not evidence that one model is statistically more robust than another.

## Nine Planned Statistical Comparisons

The following intervals come from 5,000 stratified paired bootstrap resamples (seed 6174). The Bonferroni correction across nine comparisons was introduced **post hoc**, after inspection of test results.

| Comparison | Difference | Unadjusted 95% CI | Bonferroni-adjusted CI |
|---|---:|---|---|
| gemma4-26b: P1 minus P0 | -0.01075 | [-0.03041, +0.00845] | [-0.03779, +0.01597] |
| gemma4-26b: P2 minus P0 | +0.00939 | [-0.01304, +0.03167] | [-0.02117, +0.04200] |
| qwen3.6-36b: P1 minus P0 | +0.01622 | [-0.00353, +0.03674] | [-0.01097, +0.04592] |
| qwen3.6-36b: P2 minus P0 | -0.00875 | [-0.02785, +0.01042] | [-0.03702, +0.01741] |
| llama-3.3-70b: P1 minus P0 | +0.00673 | [-0.00499, +0.01934] | [-0.00938, +0.02508] |
| llama-3.3-70b: P2 minus P0 | +0.00909 | [+0.00018, +0.01877] | [-0.00353, +0.02315] |
| gemma4-26b minus qwen3.6-36b: flip rate | -0.00233 | [-0.01977, +0.01512] | [-0.02674, +0.02209] |
| gemma4-26b minus llama-3.3-70b: flip rate | +0.00930 | [-0.00930, +0.02907] | [-0.01744, +0.03721] |
| qwen3.6-36b minus llama-3.3-70b: flip rate | +0.01163 | [-0.00698, +0.03140] | [-0.01395, +0.03953] |

**Finding:** None of the nine adjusted confidence intervals excludes zero.

The Llama P2-minus-P0 macro-F1 comparison has an unadjusted 95% interval narrowly above zero, but its Bonferroni-adjusted interval includes zero. It must not be presented as a reliable improvement after correction.

## Sensitivity and Limitations

- The exact-text duplicate sensitivity analysis retains 840 singleton posts and produces similar flip rates.
- The test dataset contains a documented conflicting gold-label duplicate.
- The historical pilot and main samples are reconstructed using preserved row IDs; the original sampling seed has not been independently established.
- Test texts contain recorded trailing whitespace that must be preserved during reproduction.
- The university endpoint confirms model aliases, but does not verify underlying checkpoints, quantization or serving configuration.
- Temperature 0 reduces randomness but does not guarantee identical outputs across new API runs.
- The experiment covers one English social-media dataset, three similar prompts and one inference setup.
- Possible model pretraining exposure to TweetEval cannot be excluded.
- Qualitative examples are illustrative rather than a representative sample of linguistic phenomena.

## Conclusion

Small prompt rewordings corresponded to changes in some individual classifications for all three model aliases. However, the planned comparisons did not establish a reliable macro-F1 improvement after post-hoc multiple-comparison correction. These findings should not be generalized beyond the evaluated dataset, prompts and serving environment.

## Reproduction

Primary inputs: final_test_results_with_baseline.csv, test_flips.csv and test_all9_comparisons.csv.

Generate this document with:

```bash
python reproducibility/write_verified_summary.py
```
