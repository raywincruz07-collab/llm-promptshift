"""Figure 1: Prompt sensitivity with bootstrap confidence intervals."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

validation = pd.read_csv("data/main_flip_confidence.csv")
test = pd.read_csv("data/test_flip_confidence.csv")

models = ["gemma4-26b", "qwen3.6-36b", "llama-3.3-70b"]
labels = ["gemma4-26b", "qwen3.6-36b", "llama-3.3-70b"]

validation = validation.set_index("model").loc[models]
test = test.set_index("model").loc[models]

fig, ax = plt.subplots(figsize=(9, 4.8))
y = np.arange(len(models))

for df, shift, name, marker in [
    (validation, -0.13, "Validation (300 posts)", "o"),
    (test, 0.13, "Held-out test (860 posts)", "s"),
]:
    center = df["flip_rate"].to_numpy() * 100
    lower = (df["flip_rate"] - df["ci_low"]).to_numpy() * 100
    upper = (df["ci_high"] - df["flip_rate"]).to_numpy() * 100

    ax.errorbar(
        center,
        y + shift,
        xerr=np.vstack([lower, upper]),
        fmt=marker,
        markersize=7,
        capsize=4,
        linewidth=1.7,
        label=name,
    )

ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(0, 13)
ax.set_xlabel("Posts with differing labels across prompts (%)")
ax.set_title("Prompt disagreement across datasets\n""Offensive share: validation 34.7%, test 27.9%")
ax.grid(axis="x", alpha=0.2)
ax.legend(loc="lower right")

fig.tight_layout()

png = FIGURES / "figure1_prompt_sensitivity.png"
pdf = FIGURES / "figure1_prompt_sensitivity.pdf"

assert not png.exists() and not pdf.exists(), "Figure already exists"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("FIGURE_1_CREATED_OK")
print("PNG:", png.resolve())
print("PDF:", pdf.resolve())
print("95% bootstrap confidence intervals included")
