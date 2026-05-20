#!/usr/bin/env python3
"""Cross-dataset Friedman test.

Loads ``results/<dataset>/all_results.json`` for each dataset that has been
run (e.g. ``twitter`` and ``phrasebank``) and runs the Friedman test across
the per-model accuracy and macro-F1 scores. This shows whether one model
consistently dominates across datasets — the strongest possible empirical
claim for a thesis.

Usage:
    # First run the experiment suite on each dataset:
    python scripts/run_all.py --dataset zeroshot/twitter-financial-news-sentiment
    python scripts/run_all.py --dataset warwickai/financial_phrasebank_mirror

    # Then aggregate:
    python scripts/cross_dataset_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas.config import RESULTS_DIR
from mas.evaluation import friedman_test, print_friedman


def load_dataset_results(name: str) -> dict[str, dict[str, float]]:
    path = RESULTS_DIR / name / "all_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    return {
        r["model_name"]: {
            "accuracy": r["accuracy"],
            "weighted_f1": r["weighted_f1"],
            "macro_f1": r["macro_f1"],
        }
        for r in rows
    }


def main() -> None:
    candidates = ["twitter", "phrasebank"]
    by_ds = {ds: load_dataset_results(ds) for ds in candidates}
    by_ds = {k: v for k, v in by_ds.items() if v}

    if len(by_ds) < 2:
        print("Need >=2 datasets with results to run Friedman test.")
        print(f"Found: {list(by_ds.keys())}")
        return

    common_models = set.intersection(*[set(d.keys()) for d in by_ds.values()])
    if not common_models:
        print("No models common to all datasets; cannot run Friedman test.")
        return

    print(f"Datasets: {list(by_ds.keys())}")
    print(f"Common models ({len(common_models)}):")
    for m in sorted(common_models):
        print(f"  - {m}")

    output: dict = {"datasets": list(by_ds.keys())}
    for metric in ("accuracy", "weighted_f1", "macro_f1"):
        scores_per_model: dict[str, list[float]] = {}
        for model in sorted(common_models):
            scores_per_model[model] = [by_ds[ds][model][metric] for ds in by_ds]
        result = friedman_test(scores_per_model)
        print(f"\n--- Metric: {metric} ---")
        print_friedman(result)
        output[metric] = result

        print("Per-dataset scores:")
        for model in sorted(common_models):
            scores = scores_per_model[model]
            print(f"  {model:<35} " + "  ".join(f"{s:.4f}" for s in scores))

    out_path = RESULTS_DIR / "cross_dataset_friedman.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved Friedman results to {out_path}")


if __name__ == "__main__":
    main()
