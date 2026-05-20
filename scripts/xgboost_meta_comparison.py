#!/usr/bin/env python3
"""Re-fit the stacking meta-learner with XGBoost on the same 12-feature
(4 agents x 3 classes) input on both Twitter and PhraseBank, and report
deltas vs the production logistic-regression meta-learner.

Inputs:
    python/results/twitter/_cache_agents.npz
    python/results/twitter/_cache_finbert.npz
    python/results/phrasebank/_cache_agents.npz
    python/results/phrasebank/_cache_finbert.npz

Outputs:
    python/results/xgboost_meta_summary.json     (machine-readable)
    Console table for the thesis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
SUMMARY_PATH = RESULTS / "xgboost_meta_summary.json"

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}


def _enc(labels: np.ndarray) -> np.ndarray:
    return np.array([LABEL_TO_INT[str(s)] for s in labels], dtype=np.int64)


def _meta_features(agents_proba: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float32) for a in agents_proba], axis=1)


def load_dataset(name: str) -> dict:
    base = RESULTS / name
    ag = np.load(base / "_cache_agents.npz", allow_pickle=True)
    fb = np.load(base / "_cache_finbert.npz", allow_pickle=True)
    val_proba = _meta_features(
        [
            ag["tfidf_val_proba"],
            fb["val_proba"],
            ag["llm_val_proba"],
            ag["lex_val_proba"],
        ]
    )
    test_proba = _meta_features(
        [
            ag["tfidf_test_proba"],
            fb["test_proba"],
            ag["llm_test_proba"],
            ag["lex_test_proba"],
        ]
    )
    val_y = _enc(ag["val_labels"])
    test_y = _enc(ag["test_labels"])
    return {
        "name": name,
        "X_val": val_proba,
        "X_test": test_proba,
        "y_val": val_y,
        "y_test": test_y,
    }


def fit_logreg(X, y):
    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


def fit_xgb(X, y, n_classes: int = 3):
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=n_classes,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
        n_jobs=1,
    )
    clf.fit(X, y)
    return clf


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def main() -> int:
    summary: list[dict] = []
    print(
        f"{'Dataset':<11} {'Meta-learner':<22} "
        f"{'Acc':>7} {'MacroF1':>8} {'WeightedF1':>11} {'Kappa':>7}"
    )
    for name in ("twitter", "phrasebank"):
        ds = load_dataset(name)

        clf_lr = fit_logreg(ds["X_val"], ds["y_val"])
        m_lr = metrics(ds["y_test"], clf_lr.predict(ds["X_test"]))

        clf_xgb = fit_xgb(ds["X_val"], ds["y_val"])
        m_xgb = metrics(ds["y_test"], clf_xgb.predict(ds["X_test"]))

        for label, m in (
            ("LogReg (production)", m_lr),
            ("XGBoost (max_depth=3, n_est=200)", m_xgb),
        ):
            print(
                f"{name:<11} {label:<22} "
                f"{m['accuracy']:>7.4f} {m['macro_f1']:>8.4f} "
                f"{m['weighted_f1']:>11.4f} {m['cohens_kappa']:>7.4f}"
            )

        delta = {
            "accuracy_pp": (m_xgb["accuracy"] - m_lr["accuracy"]) * 100,
            "macro_f1_pp": (m_xgb["macro_f1"] - m_lr["macro_f1"]) * 100,
            "weighted_f1_pp": (m_xgb["weighted_f1"] - m_lr["weighted_f1"]) * 100,
            "kappa_pp": (m_xgb["cohens_kappa"] - m_lr["cohens_kappa"]) * 100,
        }
        print(
            f"{'':<11} {'delta XGB - LR (pp)':<22} "
            f"{delta['accuracy_pp']:>+7.2f} {delta['macro_f1_pp']:>+8.2f} "
            f"{delta['weighted_f1_pp']:>+11.2f} {delta['kappa_pp']:>+7.2f}"
        )

        summary.append(
            {
                "dataset": name,
                "n_val_samples": int(len(ds["y_val"])),
                "n_test_samples": int(len(ds["y_test"])),
                "logreg_test_metrics": m_lr,
                "xgboost_test_metrics": m_xgb,
                "delta_pp": delta,
                "xgb_hyperparams": {
                    "n_estimators": 200,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                },
            }
        )

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
