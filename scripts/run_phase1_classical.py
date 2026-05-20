#!/usr/bin/env python3
"""Phase 1: Classical ML baseline (TF-IDF + Logistic Regression)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas.baselines.tfidf_logreg import TfidfLogRegBaseline
from mas.config import RESULTS_DIR, DataConfig
from mas.data import load_financial_phrasebank
from mas.data.preprocessing import preprocess_batch
from mas.evaluation import compute_metrics, print_metrics, save_results


def main() -> None:
    print("=" * 60)
    print("  Phase 1: TF-IDF + Logistic Regression Baseline")
    print("=" * 60)

    config = DataConfig()
    print(f"\nLoading {config.dataset_name} ({config.dataset_config})...")
    train, val, test = load_financial_phrasebank(config)
    print(f"  Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    train_texts = preprocess_batch(train.texts)
    val_texts = preprocess_batch(val.texts)
    test_texts = preprocess_batch(test.texts)

    print("\nTraining TF-IDF + LogReg...")
    model = TfidfLogRegBaseline()
    t0 = time.time()
    model.train(train_texts, train.labels)
    print(f"  Training took {time.time() - t0:.2f}s")

    t0 = time.time()
    test_preds = model.predict(test_texts)
    infer_ms = (time.time() - t0) * 1000

    test_result = compute_metrics(
        test.labels,
        test_preds,
        model_name="TF-IDF + LogReg",
        latency_ms=infer_ms,
    )
    print_metrics(test_result)

    val_preds = model.predict(val_texts)
    val_result = compute_metrics(
        val.labels,
        val_preds,
        model_name="TF-IDF + LogReg (val)",
    )
    print_metrics(val_result)

    all_texts = train_texts + val_texts + test_texts
    all_labels = train.labels + val.labels + test.labels
    print("\nRunning 5-fold cross-validation...")
    cv = model.cross_validate(all_texts, all_labels)
    print(f"  CV Weighted F1: {cv['mean_f1']:.4f} (+/- {cv['std_f1']:.4f})")

    save_results([test_result], RESULTS_DIR / "phase1_results.json")
    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
