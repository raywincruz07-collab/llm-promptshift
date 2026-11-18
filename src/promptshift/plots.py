"""Figures: uncertainty on every plot, one fixed colour per model.

Colours: validated categorical slots 1-3 (all-pairs colour-blind safe).
Model names are always printed next to the marks, so colour is never the
only cue; marker shape also differs by model, and fill encodes definition
(hollow = no definition, filled = definition).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

MODEL_COLOR = {"gemma4-26b": "#2a78d6", "qwen3.6-36b": "#eb6834", "llama-3.3-70b": "#1baf7a"}
MODEL_MARKER = {"gemma4-26b": "o", "qwen3.6-36b": "s", "llama-3.3-70b": "^"}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 11,
    "axes.titlecolor": INK, "axes.titleweight": "regular", "figure.dpi": 100,
})


def _style(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, out, name, w_pad=1.0):
    fig.tight_layout(w_pad=w_pad)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _errorbar(ax, x, y, low, high, model, filled=True, horizontal=False, size=7):
    color = MODEL_COLOR[model]
    kwargs = dict(fmt=MODEL_MARKER[model], color=color, ecolor=color, elinewidth=2,
                  capsize=0, markersize=size, markeredgewidth=1.5,
                  markerfacecolor=color if filled else "white")
    if horizontal:
        ax.errorbar(x, y, xerr=[[x - low], [high - x]], **kwargs)
    else:
        ax.errorbar(x, y, yerr=[[y - low], [high - y]], **kwargs)


def primary_forest(out, primary):
    metric_name = {"macro_f1": "macro-F1", "offensive_recall": "offensive recall"}[C.PRIMARY_H2]
    groups = [("H1", "H1: wording minus repeat disagreement\n(percentage points of posts)", 100),
              ("H2", f"H2: definition minus no definition\n({metric_name})", 1)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, (prefix, title, scale) in zip(axes, groups):
        tests = [t for t in primary if t["test"].startswith(prefix + "_")]
        for i, t in enumerate(tests):
            yy = len(tests) - 1 - i
            _errorbar(ax, t["estimate"] * scale, yy, t["ci_low"] * scale, t["ci_high"] * scale,
                      t["model"], horizontal=True)
            star = " *" if t["significant_holm"] else ""
            ax.annotate(f"{t['estimate'] * scale:+.2f}{star}", xy=(1.0, yy),
                        xycoords=("axes fraction", "data"), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK_2, fontsize=9)
        ax.set_yticks(range(len(tests)))
        ax.set_yticklabels([C.MODEL_SHORT[t["model"]] for t in reversed(tests)])
        ax.axvline(0, color=INK_2, linewidth=1)
        sesoi = tests[0]["sesoi"] * scale if tests else 0
        ax.axvline(sesoi, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        if tests and tests[0]["alternative"] == "two-sided":
            ax.axvline(-sesoi, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(title, loc="left")
        ax.set_ylim(-0.6, len(tests) - 0.4)
        _style(ax)
    fig.text(0.01, -0.02, "Lines: 95% bootstrap CI. Dashed: smallest effect of interest. * = significant after Holm.",
             color=INK_2, fontsize=8.5)
    _save(fig, out, "fig1_primary_tests", w_pad=5.0)


def disagreement(out, secondary_rows):
    measures = [
        ("wording disagreement, no definition (mean pairwise W0-W2)", "Wording\nno def.", False),
        ("repeat disagreement W0 vs R0", "Repeat\nno def.", False),
        ("wording disagreement, definition (mean pairwise D0-D2)", "Wording\ndefinition", True),
        ("repeat disagreement D0 vs RD0", "Repeat\ndefinition", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, model in zip(axes, C.MODELS):
        for x, (measure, _, filled) in enumerate(measures):
            row = next(r for r in secondary_rows if r["model"] == model and r["measure"] == measure)
            _errorbar(ax, x, row["estimate"] * 100, row["ci_low"] * 100, row["ci_high"] * 100, model, filled)
        ax.set_xticks(range(4))
        ax.set_xticklabels([m[1] for m in measures], fontsize=8.5)
        ax.set_title(C.MODEL_SHORT[model], loc="left")
        ax.set_xlim(-0.5, 3.5)
        _style(ax, "y")
    axes[0].set_ylabel("Posts with a different label (%)")
    axes[0].set_ylim(bottom=0)
    fig.text(0.01, -0.02, "Hollow: no definition. Filled: OLID definition. Lines: 95% bootstrap CI.",
             color=INK_2, fontsize=8.5)
    _save(fig, out, "fig2_wording_vs_repeat")


def prevalence(out, y, metrics):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    conditions = ["W0", "W1", "W2", "D0", "D1", "D2"]
    for m_index, model in enumerate(C.MODELS):
        for c_index, condition in enumerate(conditions):
            row = next(r for r in metrics if r["model"] == model and r["condition"] == condition)
            x = m_index * 7 + c_index + (0.5 if c_index >= 3 else 0)
            _errorbar(ax, x, row["prevalence"] * 100, row["prevalence_ci_low"] * 100,
                      row["prevalence_ci_high"] * 100, model, filled=c_index >= 3, size=6)
    gold = float(np.mean(y == 1)) * 100
    ax.axhline(gold, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(f"Human labels: {gold:.1f}%", (-0.6, gold), xytext=(0, 4), textcoords="offset points",
                ha="left", color=INK_2, fontsize=9)
    ax.set_xticks([m * 7 + c + (0.5 if c >= 3 else 0) for m in range(3) for c in range(6)])
    ax.set_xticklabels(conditions * 3, fontsize=8.5)
    for m_index, model in enumerate(C.MODELS):
        ax.text(m_index * 7 + 2.75, -0.16, C.MODEL_SHORT[model], transform=ax.get_xaxis_transform(),
                ha="center", color=INK, fontsize=10)
    ax.set_ylabel("Labelled offensive (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Predicted offensive prevalence by model and condition", loc="left")
    _style(ax, "y")
    fig.text(0.01, -0.06, "Hollow: no definition (W). Filled: OLID definition (D). Lines: 95% bootstrap CI.",
             color=INK_2, fontsize=8.5)
    _save(fig, out, "fig3_prevalence")


def recall_tradeoff(out, metrics):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for model in C.MODELS:
        rows = {r["condition"]: r for r in metrics if r["model"] == model}
        w = (np.mean([rows[c]["not_offensive_recall"] for c in ("W0", "W1", "W2")]),
             np.mean([rows[c]["offensive_recall"] for c in ("W0", "W1", "W2")]))
        d = (np.mean([rows[c]["not_offensive_recall"] for c in ("D0", "D1", "D2")]),
             np.mean([rows[c]["offensive_recall"] for c in ("D0", "D1", "D2")]))
        color = MODEL_COLOR[model]
        ax.annotate("", xy=d, xytext=w, arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
        ax.plot(*w, MODEL_MARKER[model], color=color, markerfacecolor="white", markersize=8, markeredgewidth=1.5)
        ax.plot(*d, MODEL_MARKER[model], color=color, markersize=8)
        ax.annotate(C.MODEL_SHORT[model], d, xytext=(8, -4), textcoords="offset points", color=INK, fontsize=10)
    ax.set_xlabel("Not-offensive recall (higher = fewer false alarms)")
    ax.set_ylabel("Offensive recall (higher = fewer misses)")
    ax.set_title("What the definition changes (mean of 3 wordings)", loc="left")
    _style(ax, "both")
    fig.text(0.01, -0.03, "Hollow: no definition. Filled: OLID definition. Arrow: direction of change.",
             color=INK_2, fontsize=8.5)
    _save(fig, out, "fig4_recall_tradeoff")


def make_all(out, y, metrics, primary, secondary_rows):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    primary_forest(out, primary)
    disagreement(out, secondary_rows)
    prevalence(out, y, metrics)
    recall_tradeoff(out, metrics)
