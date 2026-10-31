# Qualitative Case Provenance

## Source

- Dataset: TweetEval Offensive, validation sample.
- Original model-post disagreement cases: 58.
- Selected qualitative examples: 12.
- Every selected example was verified against the
  original model, row ID, gold label, text and predictions.

## Observed Selection Structure

The saved sample contains exactly two examples for
each of the six model-by-gold-label combinations.

**Important:** This balance is observable, but the
historical selection procedure and selection seed
are not established by the preserved files.
The examples must not be described as randomly selected.

## Annotations

The approved file contains two populated fields:

- `provisional_linguistic_feature`
- `human_review_note`

These are interpretive annotations, not ground-truth
linguistic classifications.

The preserved files do not independently establish
the annotator identity, number of independent reviewers,
review dates, coding rubric or inter-annotator agreement.
The filename `approved` alone does not establish
an independent adjudication procedure.

## Version Comparison

- Feature values changed from review to approved: 0 of 12.
- Review notes changed from review to approved: 12 of 12.

## Interpretation Limits

These cases illustrate possible sources of disagreement
between semantically similar prompt formulations.
They do not establish causal linguistic mechanisms.
They are not representative frequency estimates.
No inter-annotator reliability claim is made.

## Verification

Run:

```bash
python reproducibility/document_qualitative.py
```

This script verifies the preserved qualitative artifacts
against the archived validation flip cases.
