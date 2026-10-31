"""Figure 2: Held-out macro-F1 with bootstrap confidence intervals."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parents[2] / "figures"
OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(str(Path(__file__).resolve().parents[2] / "data" / "test_f1_confidence.csv"))

models = ["gemma4-26b", "qwen3.6-36b", "llama-3.3-70b"]
labels = ["gemma4-26b", "qwen3.6-36b", "llama-3.3-70b"]

fig, ax = plt.subplots(figsize=(9, 4.8))

y = np.arange(3)

for prompt, offset, marker in [
    ("P0", -0.22, "o"),
    ("P1", 0.00, "s"),
    ("P2", 0.22, "^"),
]:
    subset = (
        df[df["prompt_id"] == prompt]
        .set_index("model")
        .loc[models]
    )

    center = subset["macro_f1"].to_numpy()

    errors = np.vstack([
        center - subset["ci_low"].to_numpy(),
        subset["ci_high"].to_numpy() - center
    ])

    ax.errorbar(
        center,
        y + offset,
        xerr=errors,
        fmt=marker,
        markersize=7,
        capsize=4,
        linewidth=1.7,
        label=prompt,
    )

ax.set_yticks(y, labels)
ax.invert_yaxis()

ax.set_xlim(0.55, 0.86)
ax.set_xlabel("Macro-F1 (95% bootstrap confidence interval)")
ax.set_title("Held-out macro-F1 by model and prompt\n""Marginal CIs do not test paired prompt differences")

ax.grid(axis="x", alpha=0.2)
ax.legend(title="Prompt", loc="lower right")

fig.tight_layout()

png = OUTPUT / "figure2_macro_f1.png"
pdf = OUTPUT / "figure2_macro_f1.pdf"

assert not png.exists() and not pdf.exists(), (
    "Figure 2 already exists"
)

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("FIGURE_2_CREATED_OK")
print("PNG:", png.resolve())
print("PDF:", pdf.resolve())
print("Nine model-prompt combinations included")
