# Model Provenance — IS6170

## Verified API Information

- Endpoint: `https://maki.uni-mannheim.de/v1`
- Metadata collected: `2026-10-31T16:23:38.417580+00:00`
- Inference interface: OpenAI-compatible maKI API
- Temperature: `0`
- Maximum generated tokens: `16`
- Classification: zero-shot

## Model Aliases

| Requested model ID | Maximum input tokens |
|---|---:|
| `gemma4-26b` | 131072 |
| `qwen3.6-36b` | 262144 |
| `llama-3.3-70b` | 32768 |

## Provenance Limitations

The endpoint confirms the API aliases but does not provide
sufficient information to verify the exact underlying
model checkpoint or serving configuration.

The following were not established:

- Model checkpoint revision or weights hash
- Quantization configuration
- Inference engine and engine version
- Chat template
- Thinking-mode configuration
- Whether the deployment changed between experiment dates

The returned `owned_by` and `created` fields must not be
treated as authoritative model-developer or release metadata.

Experiments are attributed to the exact maKI API aliases.
No claims about model size, architecture, or checkpoint
identity are inferred solely from those aliases.

## Reproducibility

Offline metric reproduction uses archived predictions.
Fresh API inference requires authorized maKI access.
Exact prediction reproduction cannot be guaranteed
without identifying and preserving the serving configuration.

See `maki_model_metadata.json` for the original response.
