"""Test public reproduction without the original data directory."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPRO = ROOT / "reproducibility"
ARTIFACTS = ROOT / "artifacts" / "predictions"

ANALYSIS = [
    "analyze_pilot.py",
    "analyze_repeatability.py",
    "analyze_main.py",
    "analyze_test.py",
    "analyze_uncertainty.py",
    "analyze_test_uncertainty.py",
    "analyze_all_comparisons.py",
    "plot_figure1.py",
    "plot_figure2.py",
    "plot_figure3_v2.py",
]

PREDICTIONS = [
    "pilot_predictions.jsonl",
    "main_predictions.jsonl",
    "test_predictions.jsonl",
    "repeatability_predictions.jsonl",
]


def execute(script, directory):
    print(f"RUNNING: {script}", flush=True)

    subprocess.run(
        [sys.executable, str(directory / script)],
        cwd=directory,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    print(f"PASSED: {script}", flush=True)


def main():
    print("PUBLIC_ANALYSIS_TEST_START", flush=True)

    # First reconstruct datasets from the public source.
    subprocess.run(
        [
            sys.executable,
            str(REPRO / "bootstrap_public_data.py"),
        ],
        cwd=ROOT,
        check=True,
    )

    workspace = Path(
        tempfile.mkdtemp(
            prefix="is6170_public_reproduction_"
        )
    )

    data_dir = workspace / "data"
    data_dir.mkdir()

    # Copy only publicly reconstructable datasets.
    for source in (REPRO / "fresh_build").glob("*.parquet"):
        shutil.copy2(source, data_dir / source.name)

    # Copy the validated prediction artifacts.
    for filename in PREDICTIONS:
        source = ARTIFACTS / filename
        assert source.is_file(), f"Missing artifact: {filename}"
        shutil.copy2(source, data_dir / filename)

    # Copy code needed to reproduce numerical analysis.
    scripts = (
        ANALYSIS
        + ["prepare_repeatability.py", "prompts.py"]
    )

    for filename in scripts:
        shutil.copy2(
            ROOT / filename,
            workspace / filename,
        )

    # Reconstruct repeatability request metadata.
    execute("prepare_repeatability.py", workspace)

    # Run every numerical analysis.
    for filename in ANALYSIS:
        execute(filename, workspace)

    # Validate the independently reconstructed results.
    test_metrics = pd.read_csv(
        data_dir / "test_metrics.csv"
    )
    flips = pd.read_csv(
        data_dir / "test_flips.csv"
    )
    comparisons = pd.read_csv(
        data_dir / "test_all9_comparisons.csv"
    )

    assert len(test_metrics) == 9
    assert len(flips) == 3
    assert len(comparisons) == 9

    expected = {
        "gemma4-26b": 39,
        "qwen3.6-36b": 41,
        "llama-3.3-70b": 31,
    }

    actual = dict(
        zip(
            flips["model"],
            flips["posts_with_any_flip"],
        )
    )

    assert actual == expected, (
        f"Unexpected flip counts: {actual}"
    )

    assert not comparisons[
        "adjusted_ci_excludes_zero"
    ].any()

    print()
    figure_dir = workspace / "figures"
    expected_figures = [
        "figure1_prompt_sensitivity",
        "figure2_macro_f1",
        "figure3_recall_tradeoff_v2",
    ]

    for name in expected_figures:
        for extension in [".png", ".pdf"]:
            path = figure_dir / f"{name}{extension}"
            assert path.is_file(), f"Missing figure: {path}"
            assert path.stat().st_size > 1000

    print("PUBLIC_FIGURES_REPRODUCED_OK")
    print("Figures:", len(expected_figures))
    print("PNG/PDF files:", len(expected_figures) * 2)
    print("PUBLIC_ANALYSIS_REPRODUCTION_OK")
    print("Test model-prompt conditions:", len(test_metrics))
    print("Model flip summaries:", len(flips))
    print("Statistical comparisons:", len(comparisons))
    print("Original data directory used: NO")
    print("LLM inference calls: 0")
    # Export only derived tables and figures to a stable location.
    # Do not export raw posts, prompts or private experiment files.
    destination = ROOT / "reproduced_outputs"
    tables_dir = destination / "tables"
    figures_dir = destination / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables = sorted(data_dir.glob("*.csv"))

    assert len(tables) >= 10, "Expected analysis tables missing"

    for source in tables:
        columns = pd.read_csv(source, nrows=0).columns

        assert not {"text", "prompt_text"}.intersection(columns), (
            f"Unexpected text column in {source.name}"
        )

        shutil.copy2(source, tables_dir / source.name)

    figure_files = sorted(
        path for path in figure_dir.iterdir()
        if path.suffix in {".png", ".pdf"}
    )

    assert len(figure_files) == 6

    for source in figure_files:
        shutil.copy2(source, figures_dir / source.name)

    print("REPRODUCED_OUTPUTS_EXPORTED_OK")
    print("Tables exported:", len(tables))
    print("Figures exported:", len(figure_files))
    print("Results:", destination.resolve())

    shutil.rmtree(workspace)
    print("Temporary workspace removed: YES")


if __name__ == "__main__":
    main()
