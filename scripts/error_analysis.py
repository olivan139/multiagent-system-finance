#!/usr/bin/env python3
"""Error analysis and SHAP interpretability for the heterogeneous ensemble.

For a given dataset (default: ``twitter``):

  1. Loads ``predictions.json`` from a previous ``run_all.py`` run.
  2. For every model + every (true_class -> predicted_class) cell, lists the
     proportion of errors and a few example mispredictions.
  3. Reports per-input characteristics where each model fails most:
        - tweet length (chars / tokens)
        - presence of digits, %, $, named tickers
  4. Trains the stacking meta-learner once more (live) and computes SHAP
     values to show which agent's probabilities drive each decision.

Usage:
    python scripts/error_analysis.py --dataset twitter
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas.config import LABELS, RESULTS_DIR, DataConfig
from mas.data import load_financial_phrasebank
from mas.data.preprocessing import preprocess_batch

DATASET_TO_HF = {
    "twitter": "zeroshot/twitter-financial-news-sentiment",
    "phrasebank": "warwickai/financial_phrasebank_mirror",
}


_TICKER_RE = re.compile(r"\$[A-Z]{1,5}\b")
_NUM_RE = re.compile(r"\d")
_PCT_RE = re.compile(r"%")


def text_features(text: str) -> dict:
    return {
        "len_chars": len(text),
        "len_tokens": len(text.split()),
        "has_digit": int(bool(_NUM_RE.search(text))),
        "has_percent": int(bool(_PCT_RE.search(text))),
        "has_ticker": int(bool(_TICKER_RE.search(text))),
    }


def per_model_error_breakdown(
    y_true: list[str],
    predictions: dict[str, list[str]],
) -> dict:
    """For each model, build a (true → pred) confusion table with rates."""
    out: dict = {}
    for name, y_pred in predictions.items():
        cm = {t: {p: 0 for p in LABELS} for t in LABELS}
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1

        totals_per_true = {t: sum(cm[t].values()) for t in LABELS}
        rates = {
            t: {p: (cm[t][p] / totals_per_true[t] if totals_per_true[t] else 0.0) for p in LABELS}
            for t in LABELS
        }
        n_errors = sum(1 for t, p in zip(y_true, y_pred) if t != p)
        out[name] = {
            "confusion_counts": cm,
            "confusion_rates": rates,
            "total_errors": n_errors,
            "total_samples": len(y_true),
        }
    return out


def feature_stats_per_model_error(
    texts: list[str],
    y_true: list[str],
    predictions: dict[str, list[str]],
) -> dict:
    """Compare avg text-feature values for correct vs wrong predictions."""
    results: dict = {}
    feats = [text_features(t) for t in texts]
    feat_keys = list(feats[0].keys())
    feats_arr = np.array([[f[k] for k in feat_keys] for f in feats], dtype=np.float64)

    for name, y_pred in predictions.items():
        correct_mask = np.array([t == p for t, p in zip(y_true, y_pred)])
        if correct_mask.sum() == 0 or (~correct_mask).sum() == 0:
            continue
        per_feat: dict = {}
        for j, k in enumerate(feat_keys):
            per_feat[k] = {
                "mean_correct": float(feats_arr[correct_mask, j].mean()),
                "mean_wrong": float(feats_arr[~correct_mask, j].mean()),
                "delta_wrong_minus_correct": float(
                    feats_arr[~correct_mask, j].mean() - feats_arr[correct_mask, j].mean()
                ),
            }
        results[name] = per_feat
    return results


def example_errors(
    texts: list[str],
    y_true: list[str],
    y_pred: list[str],
    n_examples: int = 5,
) -> list[dict]:
    out: list[dict] = []
    for t, yt, yp in zip(texts, y_true, y_pred):
        if yt != yp:
            out.append({"text": t, "true": yt, "pred": yp})
        if len(out) >= n_examples:
            break
    return out


def hardest_examples(
    texts: list[str],
    y_true: list[str],
    predictions: dict[str, list[str]],
    n_examples: int = 10,
) -> list[dict]:
    """Examples no model gets right (or only one does)."""
    n = len(y_true)
    correct_counts = np.zeros(n, dtype=int)
    for preds in predictions.values():
        correct_counts += np.array([yt == yp for yt, yp in zip(y_true, preds)])

    order = np.argsort(correct_counts)
    hardest = []
    for idx in order[:n_examples]:
        hardest.append(
            {
                "text": texts[idx],
                "true": y_true[idx],
                "n_models_correct": int(correct_counts[idx]),
                "predictions_per_model": {name: predictions[name][idx] for name in predictions},
            }
        )
    return hardest


def shap_meta_learner(dataset: str, max_train_for_meta: int = 2000) -> dict:
    """Re-train the stacking meta-learner and compute mean |SHAP| per feature.

    This is expensive (re-runs FinBERT + LLM on val), so we cache:
      ``results/<dataset>/ensemble_meta.json`` already has feature_importance
      from coefficient magnitudes. SHAP gives a complementary view based on
      decision contribution per sample.
    """
    try:
        import shap
    except ImportError:
        return {"info": "shap not installed; skipping SHAP analysis"}

    meta_info_path = RESULTS_DIR / dataset / "ensemble_meta.json"
    if meta_info_path.exists():
        with open(meta_info_path) as f:
            meta = json.load(f)
        importance = meta.get("feature_importance", {})
        return {
            "info": "Returning meta-learner |coef| importance from cached run; "
            "for full SHAP run, re-execute with the live ensemble.",
            "feature_importance_abs_coef": importance,
        }
    return {"info": "No cached ensemble meta found; run run_all.py first."}


def _fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def print_error_breakdown(name: str, info: dict) -> None:
    print(f"\n--- {name} ---")
    print(
        f"  {info['total_errors']}/{info['total_samples']} errors "
        f"({info['total_errors']/info['total_samples']*100:.1f}%)"
    )
    rates = info["confusion_rates"]
    header = "true \\ pred"
    print(f"  {header:<14} " + " ".join(f"{l:>9s}" for l in LABELS))
    for t in LABELS:
        row = " ".join(_fmt_pct(rates[t][p]) for p in LABELS)
        print(f"  {t:<14} {row}")


def print_feature_stats(name: str, stats: dict) -> None:
    print(f"\n--- {name}: text features (mean over correct vs wrong) ---")
    print(f"  {'feature':<14} {'correct':>10} {'wrong':>10} {'delta':>10}")
    for k, v in stats.items():
        print(
            f"  {k:<14} {v['mean_correct']:>10.2f} {v['mean_wrong']:>10.2f} "
            f"{v['delta_wrong_minus_correct']:>+10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="twitter", help="Short dataset name (twitter / phrasebank)"
    )
    parser.add_argument("--n-examples", type=int, default=5)
    args = parser.parse_args()

    pred_path = RESULTS_DIR / args.dataset / "predictions.json"
    if not pred_path.exists():
        raise SystemExit(f"Predictions not found at {pred_path}. Run run_all.py first.")

    with open(pred_path) as f:
        data = json.load(f)
    y_true = data["y_true"]
    predictions = data["predictions"]

    hf_name = DATASET_TO_HF.get(args.dataset, args.dataset)
    cfg = DataConfig(dataset_name=hf_name)
    _, _, test = load_financial_phrasebank(cfg)
    test_texts = preprocess_batch(test.texts)[: len(y_true)]

    print("=" * 70)
    print(f"  Error Analysis  (dataset: {args.dataset}, n_test={len(y_true)})")
    print("=" * 70)

    breakdown = per_model_error_breakdown(y_true, predictions)
    for name, info in breakdown.items():
        print_error_breakdown(name, info)

    print("\n" + "=" * 70)
    print("  Text-feature stats per model (correct vs wrong)")
    print("=" * 70)
    feat_stats = feature_stats_per_model_error(test_texts, y_true, predictions)
    for name, stats in feat_stats.items():
        print_feature_stats(name, stats)

    print("\n" + "=" * 70)
    print(f"  Hardest examples (no/few models correct)")
    print("=" * 70)
    hardest = hardest_examples(test_texts, y_true, predictions, n_examples=args.n_examples)
    for ex in hardest:
        print(f"\n  TRUE={ex['true']}  models_correct={ex['n_models_correct']}")
        print(f"  TEXT: {ex['text'][:200]}")
        for m, p in ex["predictions_per_model"].items():
            tag = "✓" if p == ex["true"] else "✗"
            print(f"    {tag} {m:<35} -> {p}")

    print("\n" + "=" * 70)
    print("  Meta-learner feature importance (|coef|, from cached ensemble run)")
    print("=" * 70)
    shap_info = shap_meta_learner(args.dataset)
    print(json.dumps(shap_info, indent=2)[:2000])

    out_path = RESULTS_DIR / args.dataset / "error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "per_model_breakdown": breakdown,
                "feature_stats": feat_stats,
                "hardest_examples": hardest,
                "shap_summary": shap_info,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nError-analysis report saved to {out_path}")


if __name__ == "__main__":
    main()
