#!/usr/bin/env python3
"""Re-fit the stacking meta-learner with the FinTwitBERT-sentiment agent
added on top of the existing TF-IDF, FinBERT-FT, GPT-4o-mini and
Loughran-McDonald base agents. Reports test-split deltas vs the
production 4-agent stack on both Twitter and PhraseBank, and runs
McNemar's test on each pair of stack predictions.

Inputs:
    python/results/<dataset>/_cache_agents.npz
    python/results/<dataset>/_cache_finbert.npz
    python/results/<dataset>/_cache_fintwit.npz   # produced earlier

Outputs:
    python/results/five_agent_stack_summary.json
    Console table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
SUMMARY_PATH = RESULTS / "five_agent_stack_summary.json"

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}


def _enc(labels) -> np.ndarray:
    return np.array([LABEL_TO_INT[str(s)] for s in labels], dtype=np.int64)


def _meta_features(arrs: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float32) for a in arrs], axis=1)


def fit_lr_stack(X_val, y_val, X_test, y_test) -> dict:
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    clf.fit(X_val, y_val)
    pred = clf.predict(X_test)
    return {
        "predictions": pred,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "kappa": float(cohen_kappa_score(y_test, pred)),
        "coef_abs_sum": float(np.abs(clf.coef_).sum()),
        "coef_per_block": np.abs(clf.coef_).sum(axis=0),
        "intercepts": clf.intercept_.tolist(),
    }


def mcnemar_p(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    """Two-sided McNemar test on two systems' predictions."""
    a_right = a == y
    b_right = b == y
    n01 = int(np.sum(~a_right & b_right))
    n10 = int(np.sum(a_right & ~b_right))
    if n01 + n10 == 0:
        return 1.0
    table = [[0, n01], [n10, 0]]
    return float(mcnemar(table, exact=False, correction=True).pvalue)


def run_one(name: str) -> dict:
    print(f"\n========== {name} ==========")
    ag = np.load(RESULTS / name / "_cache_agents.npz", allow_pickle=True)
    fb = np.load(RESULTS / name / "_cache_finbert.npz", allow_pickle=True)
    ft = np.load(RESULTS / name / "_cache_fintwit.npz", allow_pickle=True)

    y_val = _enc(ag["val_labels"])
    y_test = _enc(ag["test_labels"])
    n_val, n_test = len(y_val), len(y_test)
    print(f"val={n_val}  test={n_test}")

    blocks_4 = [
        ("tfidf", ag["tfidf_val_proba"], ag["tfidf_test_proba"]),
        ("finbert", fb["val_proba"], fb["test_proba"]),
        ("llm", ag["llm_val_proba"], ag["llm_test_proba"]),
        ("lexicon", ag["lex_val_proba"], ag["lex_test_proba"]),
    ]
    blocks_5 = blocks_4 + [
        ("fintwit", ft["val_proba"], ft["test_proba"]),
    ]

    X4_val = _meta_features([b[1] for b in blocks_4])
    X4_test = _meta_features([b[2] for b in blocks_4])
    X5_val = _meta_features([b[1] for b in blocks_5])
    X5_test = _meta_features([b[2] for b in blocks_5])

    print(f"feature dims: 4-agent={X4_val.shape[1]}, 5-agent={X5_val.shape[1]}")

    m4 = fit_lr_stack(X4_val, y_val, X4_test, y_test)
    m5 = fit_lr_stack(X5_val, y_val, X5_test, y_test)
    p_mcnemar = mcnemar_p(m4["predictions"], m5["predictions"], y_test)

    def _print(label, m):
        print(
            f"  {label:<18} acc={m['accuracy']:.4f}  mf1={m['macro_f1']:.4f}  "
            f"wf1={m['weighted_f1']:.4f}  kappa={m['kappa']:.4f}"
        )

    _print("4-agent stack (LR)", m4)
    _print("5-agent stack (LR)", m5)
    print(
        f"  delta 5-vs-4 (pp): "
        f"acc={(m5['accuracy']-m4['accuracy'])*100:+.2f}  "
        f"mf1={(m5['macro_f1']-m4['macro_f1'])*100:+.2f}  "
        f"wf1={(m5['weighted_f1']-m4['weighted_f1'])*100:+.2f}  "
        f"kappa={(m5['kappa']-m4['kappa'])*100:+.2f}"
    )
    print(f"  McNemar p-value (4 vs 5): {p_mcnemar:.4g}")

    coef_names_4 = ["tfidf", "finbert", "llm", "lexicon"]
    coef_names_5 = coef_names_4 + ["fintwit"]
    coefs_5_per_block = {}
    for i, name_ in enumerate(coef_names_5):
        s = float(m5["coef_per_block"][i * 3 : (i + 1) * 3].sum())
        coefs_5_per_block[name_] = s
    total5 = sum(coefs_5_per_block.values())
    print(
        "  5-agent coef shares: "
        + ", ".join(f"{k}={v/total5*100:.1f}%" for k, v in coefs_5_per_block.items())
    )

    fintwit_standalone = {
        "accuracy": float(ft["test_accuracy"][0]),
        "macro_f1": float(ft["test_macro_f1"][0]),
    }
    finbert_standalone = {
        "accuracy": float(fb["test_accuracy"][0]),
    }

    return {
        "dataset": name,
        "n_val": int(n_val),
        "n_test": int(n_test),
        "fintwit_standalone": fintwit_standalone,
        "finbert_standalone": finbert_standalone,
        "stack_4_agent": {
            k: v for k, v in m4.items() if k not in ("predictions", "coef_per_block", "intercepts")
        },
        "stack_5_agent": {
            k: v for k, v in m5.items() if k not in ("predictions", "coef_per_block", "intercepts")
        },
        "delta_pp": {
            "accuracy": (m5["accuracy"] - m4["accuracy"]) * 100,
            "macro_f1": (m5["macro_f1"] - m4["macro_f1"]) * 100,
            "weighted_f1": (m5["weighted_f1"] - m4["weighted_f1"]) * 100,
            "kappa": (m5["kappa"] - m4["kappa"]) * 100,
        },
        "mcnemar_p_5_vs_4": p_mcnemar,
        "five_agent_coef_shares_pct": {k: v / total5 * 100 for k, v in coefs_5_per_block.items()},
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+", default=["twitter", "phrasebank", "semeval2017", "fiqa2018"]
    )
    args = p.parse_args()
    out = [run_one(name) for name in args.datasets]
    SUMMARY_PATH.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
