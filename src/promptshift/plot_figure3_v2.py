"""Figure 3: Held-out class-recall trade-off."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "figures"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(str(Path(__file__).resolve().parents[2] / "data" / "test_metrics.csv"))

models = {
    "gemma4-26b": "gemma4-26b",
    "qwen3.6-36b": "qwen3.6-36b",
    "llama-3.3-70b": "llama-3.3-70b",
}

prompts = ["P0", "P1", "P2"]
markers = {"P0": "o", "P1": "s", "P2": "^"}

# One consistent color per model.
colors = {
    "gemma4-26b": "#0072B2",
    "qwen3.6-36b": "#D55E00",
    "llama-3.3-70b": "#009E73",
}

assert len(df) == 9
assert not df.duplicated(
    ["model", "prompt_id"]
).any()

fig, ax = plt.subplots(figsize=(9.4, 5.8))

for model in models:
    for prompt in prompts:
        row = df[
            (df["model"] == model)
            & (df["prompt_id"] == prompt)
        ]

        assert len(row) == 1

        ax.scatter(
            row["nonoffensive_recall"].iloc[0],
            row["offensive_recall"].iloc[0],
            color=colors[model],
            marker=markers[prompt],
            s=125,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

ax.set_xlabel("Non-offensive recall (specificity)")
ax.set_ylabel("Offensive recall (sensitivity)")
ax.set_title(
    "Held-out class recall by model and prompt"
)

ax.set_xlim(0.81, 0.99)
ax.set_ylim(0.25, 0.86)
ax.grid(alpha=0.2)

# Separate model legend: color identifies model.
model_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        color=colors[model],
        markersize=9,
        label=name,
    )
    for model, name in models.items()
]

model_legend = ax.legend(
    handles=model_handles,
    title="Model (color)",
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    frameon=False,
)

ax.add_artist(model_legend)

# Separate prompt legend: shape identifies prompt.
prompt_handles = [
    Line2D(
        [0], [0],
        marker=markers[prompt],
        linestyle="None",
        color="#333333",
        markersize=9,
        label=prompt,
    )
    for prompt in prompts
]

ax.legend(
    handles=prompt_handles,
    title="Prompt (shape)",
    loc="upper left",
    bbox_to_anchor=(1.02, 0.57),
    frameon=False,
)

fig.subplots_adjust(right=0.72)

png = OUT / "figure3_recall_tradeoff_v2.png"
pdf = OUT / "figure3_recall_tradeoff_v2.pdf"

assert not png.exists() and not pdf.exists()

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("FIGURE_3_V2_CREATED_OK")
print("PNG:", png.resolve())
print("PDF:", pdf.resolve())
print("Model colors: consistent")
print("Prompt markers: consistent")
print("Overlapping labels: removed")
