#!/usr/bin/env python3
"""Tuned XGBoost meta-learner on the same 12-feature stack used by the
production logistic regression.

Search protocol (per dataset):
  * Random search over an XGBoost hyper-parameter grid, N_TRIALS configs,
    seeded so the run is reproducible.
  * Each config is evaluated by 5-fold stratified CV on the validation
    split (no test-set leakage).
  * The config with the highest mean CV macro-F1 is refit on the full
    validation split and scored on the test split.
  * For context we also report:
        - the default-config XGBoost score (max_depth=3, n=200, lr=0.1)
        - the production logistic-regression score on the same features.

Outputs:
  python/results/xgboost_meta_tuned_summary.json
"""

from __future__ import annotations

import json
import random
import time
from itertools import islice
from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
SUMMARY_PATH = RESULTS / "xgboost_meta_tuned_summary.json"

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}

N_TRIALS = 80
N_FOLDS = 5
RANDOM_SEED = 42

GRID: dict[str, list] = {
    "n_estimators": [50, 100, 200, 300, 500, 800],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.1, 1.0, 3.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "gamma": [0.0, 0.1, 0.5],
}


def _enc(labels: np.ndarray) -> np.ndarray:
    return np.array([LABEL_TO_INT[str(s)] for s in labels], dtype=np.int64)


def _meta_features(arrs: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float32) for a in arrs], axis=1)


def load_dataset(name: str) -> dict:
    base = RESULTS / name
    ag = np.load(base / "_cache_agents.npz", allow_pickle=True)
    fb = np.load(base / "_cache_finbert.npz", allow_pickle=True)
    return {
        "name": name,
        "X_val": _meta_features(
            [ag["tfidf_val_proba"], fb["val_proba"], ag["llm_val_proba"], ag["lex_val_proba"]]
        ),
        "X_test": _meta_features(
            [ag["tfidf_test_proba"], fb["test_proba"], ag["llm_test_proba"], ag["lex_test_proba"]]
        ),
        "y_val": _enc(ag["val_labels"]),
        "y_test": _enc(ag["test_labels"]),
    }


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def make_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        verbosity=0,
        n_jobs=1,
        random_state=RANDOM_SEED,
        **params,
    )


def sample_configs(n: int, seed: int) -> Iterator[dict]:
    rng = random.Random(seed)
    seen: set[tuple] = set()
    while len(seen) < n:
        cfg = {k: rng.choice(v) for k, v in GRID.items()}
        key = tuple(cfg[k] for k in sorted(cfg))
        if key in seen:
            continue
        seen.add(key)
        yield cfg


def cv_score(X: np.ndarray, y: np.ndarray, params: dict) -> dict:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    acc, mf1 = [], []
    for tr, va in skf.split(X, y):
        clf = make_xgb(params)
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[va])
        acc.append(accuracy_score(y[va], p))
        mf1.append(f1_score(y[va], p, average="macro"))
    return {
        "cv_acc_mean": float(np.mean(acc)),
        "cv_acc_std": float(np.std(acc)),
        "cv_mf1_mean": float(np.mean(mf1)),
        "cv_mf1_std": float(np.std(mf1)),
    }


LR_GRID: dict[str, list] = {
    "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
    "penalty": ["l2"],
    "class_weight": [None, "balanced"],
}


def make_logreg(params: dict) -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_SEED,
        **params,
    )


def lr_cv_score(X: np.ndarray, y: np.ndarray, params: dict) -> dict:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    acc, mf1 = [], []
    for tr, va in skf.split(X, y):
        clf = make_logreg(params)
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[va])
        acc.append(accuracy_score(y[va], p))
        mf1.append(f1_score(y[va], p, average="macro"))
    return {
        "cv_acc_mean": float(np.mean(acc)),
        "cv_acc_std": float(np.std(acc)),
        "cv_mf1_mean": float(np.mean(mf1)),
        "cv_mf1_std": float(np.std(mf1)),
    }


def tune_logreg(X_val: np.ndarray, y_val: np.ndarray) -> tuple[dict, dict]:
    """Exhaustive grid over LR_GRID, scored by 5-fold CV on the val split."""
    from itertools import product

    keys = list(LR_GRID.keys())
    best = None
    best_params: dict | None = None
    for combo in product(*(LR_GRID[k] for k in keys)):
        params = dict(zip(keys, combo))
        s = lr_cv_score(X_val, y_val, params)
        if best is None or s["cv_mf1_mean"] > best["cv_mf1_mean"]:
            best = s
            best_params = params
    assert best_params is not None and best is not None
    return best_params, best


def evaluate_dataset(ds: dict) -> dict:
    print(f"\n=== {ds['name']} (val={len(ds['y_val'])}, test={len(ds['y_test'])}) ===")

    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_SEED)
    lr.fit(ds["X_val"], ds["y_val"])
    m_lr = metrics(ds["y_test"], lr.predict(ds["X_test"]))
    print(f"LogReg (production) test acc={m_lr['accuracy']:.4f}  " f"mf1={m_lr['macro_f1']:.4f}")

    lr_best_params, lr_cv = tune_logreg(ds["X_val"], ds["y_val"])
    clf_lr_tuned = make_logreg(lr_best_params)
    clf_lr_tuned.fit(ds["X_val"], ds["y_val"])
    m_lr_tuned = metrics(ds["y_test"], clf_lr_tuned.predict(ds["X_test"]))
    print(
        f"LogReg (tuned)      test acc={m_lr_tuned['accuracy']:.4f}  "
        f"mf1={m_lr_tuned['macro_f1']:.4f}  best={lr_best_params}"
    )

    default_params = dict(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
    )
    clf_def = make_xgb(default_params)
    clf_def.fit(ds["X_val"], ds["y_val"])
    m_def = metrics(ds["y_test"], clf_def.predict(ds["X_test"]))
    print(f"XGB default         test acc={m_def['accuracy']:.4f}  " f"mf1={m_def['macro_f1']:.4f}")

    t0 = time.time()
    trials: list[dict] = []
    for i, cfg in enumerate(sample_configs(N_TRIALS, seed=RANDOM_SEED + 1)):
        s = cv_score(ds["X_val"], ds["y_val"], cfg)
        trials.append({"config": cfg, **s})
        if (i + 1) % 10 == 0:
            best = max(trials, key=lambda t: t["cv_mf1_mean"])
            print(
                f"  [{i + 1:3d}/{N_TRIALS}] best CV mf1 so far: "
                f"{best['cv_mf1_mean']:.4f} ({best['cv_mf1_std']:.4f})"
            )
    dt = time.time() - t0
    best = max(trials, key=lambda t: t["cv_mf1_mean"])
    print(
        f"Search took {dt:.1f}s. Best CV mf1={best['cv_mf1_mean']:.4f} "
        f"(std {best['cv_mf1_std']:.4f}) acc={best['cv_acc_mean']:.4f}"
    )
    print(f"Best config: {best['config']}")

    clf_best = make_xgb(best["config"])
    clf_best.fit(ds["X_val"], ds["y_val"])
    m_best = metrics(ds["y_test"], clf_best.predict(ds["X_test"]))
    print(
        f"XGB tuned           test acc={m_best['accuracy']:.4f}  " f"mf1={m_best['macro_f1']:.4f}"
    )

    delta_default = {
        k + "_pp": (m_def[k] - m_lr[k]) * 100
        for k in ("accuracy", "macro_f1", "weighted_f1", "kappa")
    }
    delta_tuned = {
        k + "_pp": (m_best[k] - m_lr[k]) * 100
        for k in ("accuracy", "macro_f1", "weighted_f1", "kappa")
    }
    delta_tuned_vs_lrtuned = {
        k + "_pp": (m_best[k] - m_lr_tuned[k]) * 100
        for k in ("accuracy", "macro_f1", "weighted_f1", "kappa")
    }
    return {
        "dataset": ds["name"],
        "n_val": int(len(ds["y_val"])),
        "n_test": int(len(ds["y_test"])),
        "logreg_test_metrics": m_lr,
        "logreg_tuned_metrics": m_lr_tuned,
        "logreg_tuned_params": lr_best_params,
        "logreg_tuned_cv": lr_cv,
        "xgb_default_metrics": m_def,
        "xgb_default_params": default_params,
        "xgb_tuned_metrics": m_best,
        "xgb_tuned_params": best["config"],
        "xgb_tuned_cv": {
            k: best[k] for k in ("cv_acc_mean", "cv_acc_std", "cv_mf1_mean", "cv_mf1_std")
        },
        "delta_default_pp": delta_default,
        "delta_tuned_pp": delta_tuned,
        "delta_tuned_vs_lrtuned_pp": delta_tuned_vs_lrtuned,
        "search_seconds": dt,
        "n_trials": N_TRIALS,
        "n_folds": N_FOLDS,
        "random_seed": RANDOM_SEED,
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+", default=["twitter", "phrasebank", "semeval2017", "fiqa2018"]
    )
    args = p.parse_args()
    out: list[dict] = []
    for name in args.datasets:
        out.append(evaluate_dataset(load_dataset(name)))

    SUMMARY_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {SUMMARY_PATH}")

    print("\n=== Summary table ===")
    print(f"{'Dataset':<11} {'Meta':<16} " f"{'Acc':>8} {'MacroF1':>9} {'WF1':>8} {'Kappa':>8}")
    for r in out:
        for label, m in (
            ("LogReg (prod)", r["logreg_test_metrics"]),
            ("LogReg (tuned)", r["logreg_tuned_metrics"]),
            ("XGB (default)", r["xgb_default_metrics"]),
            ("XGB (tuned)", r["xgb_tuned_metrics"]),
        ):
            print(
                f"{r['dataset']:<11} {label:<16} "
                f"{m['accuracy']:>8.4f} {m['macro_f1']:>9.4f} "
                f"{m['weighted_f1']:>8.4f} {m['kappa']:>8.4f}"
            )
        d = r["delta_tuned_vs_lrtuned_pp"]
        print(
            f"{'':<11} {'Δ XGB-LR(tuned)':<16} "
            f"{d['accuracy_pp']:>+8.2f} {d['macro_f1_pp']:>+9.2f} "
            f"{d['weighted_f1_pp']:>+8.2f} {d['kappa_pp']:>+8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
