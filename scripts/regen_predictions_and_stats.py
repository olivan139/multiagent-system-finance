#!/usr/bin/env python3
"""Regenerate ``predictions.json`` and ``statistical_report.json`` for
twitter and phrasebank, using the **current** (post lexicon-stemming-fix)
``_cache_agents.npz`` + ``_cache_finbert.npz`` caches as the
authoritative source.

What this fixes
---------------
``refresh_all_results_from_cache.py`` only updated the *aggregate*
rows in ``all_results.json`` for lexicon / 4-agent-stack / 4 ablations.
It did NOT regenerate ``predictions.json`` and it did NOT refresh
the majority-vote / weighted-average rows.  As a result:

    * Majority + weighted-average rows in ``all_results.json`` still
      reflect the pre-stemming-fix lexicon votes.
    * Per-sample predictions in ``predictions.json`` (which the McNemar
      table, bootstrap CIs, agent-agreement heatmap and confusion-matrix
      figures all read from) are pre-stemming-fix.
    * Three different *snapshots of truth* are referenced in the thesis
      simultaneously: the very old McNemar table values, the May-14
      ``predictions.json``, and the May-17 ``all_results.json``.

This script rebuilds everything that depends on per-sample predictions
from the **post-fix** caches so the whole pipeline becomes internally
consistent.

What it does NOT touch
----------------------
* ``_cache_agents.npz`` (treated as authoritative input)
* ``_cache_finbert.npz`` (authoritative input)
* ``_cache_fintwit.npz`` (authoritative input for the 5-agent extension)
* ``ensemble_meta.json`` (will be refreshed)
* Per-agent baseline rows in ``all_results.json`` for systems whose
  predictions are not derived from cache (e.g. Multi-Agent Pipeline).
  Their predictions in ``predictions.json`` are *preserved* from the
  current file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mas.config import LABELS, RESULTS_DIR
from mas.evaluation.statistical import (
    pairwise_mcnemar,
    bootstrap_ci,
    save_statistical_report,
)

LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}
NEUTRAL_IDX = LABEL_TO_INT["neutral"]


def _argmax_labels(proba: np.ndarray) -> list[str]:
    return [LABELS[i] for i in np.argmax(proba, axis=1)]


def _majority_vote(per_agent_argmax: list[np.ndarray]) -> list[str]:
    """Reproduce ``HeterogeneousEnsemble._majority_vote`` exactly:
    argmax of vote counts, ties → ``neutral`` (when every vote is for a
    different class, counts.max() == 1).
    """
    n = len(per_agent_argmax[0])
    K = len(LABELS)
    out: list[str] = []
    for i in range(n):
        votes = np.bincount([arr[i] for arr in per_agent_argmax], minlength=K)
        top = int(np.argmax(votes))
        if votes[top] == 1:
            top = NEUTRAL_IDX
        out.append(LABELS[top])
    return out


def _weighted_average(
    per_agent_proba: list[np.ndarray], weights: list[float] | None = None
) -> list[str]:
    """Uniform-weight by default, matching ``HeterogeneousEnsemble`` init."""
    if weights is None:
        weights = [1.0] * len(per_agent_proba)
    total = sum(weights)
    avg = sum(w * p for w, p in zip(weights, per_agent_proba)) / total
    return _argmax_labels(avg)


def _stack_predict(X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray) -> list[str]:
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    clf.fit(X_val, y_val)
    pred_idx = clf.predict(X_test)
    return [LABELS[int(i)] for i in pred_idx]


def _metrics_row(
    name: str,
    y_true: list[str],
    y_pred: list[str],
    metadata: dict | None = None,
    latency_ms: float | None = None,
    cost_usd: float = 0.0,
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()
    rep = classification_report(
        y_true, y_pred, labels=list(LABELS), zero_division=0, output_dict=True
    )
    return {
        "model_name": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class_report": rep,
        "confusion_matrix": cm,
        "latency_ms_per_sample": latency_ms,
        "cost_usd_per_sample": float(cost_usd),
        "total_samples": int(len(y_true)),
        "metadata": metadata or {},
    }


def regen_dataset(name: str, dry_run: bool = False) -> dict:
    """Regenerate predictions / stats for one dataset.

    Returns a delta-report dict for printing.
    """
    rdir = RESULTS_DIR / name
    print(f"\n========== {name} ==========")

    cache = np.load(rdir / "_cache_agents.npz", allow_pickle=True)
    fb = np.load(rdir / "_cache_finbert.npz", allow_pickle=True)
    y_test_str: list[str] = [str(s) for s in cache["test_labels"]]
    y_val: np.ndarray = np.array([LABEL_TO_INT[str(s)] for s in cache["val_labels"]])

    preds_path = rdir / "predictions.json"
    pj_old = json.loads(preds_path.read_text())
    old_preds: dict[str, list[str]] = pj_old["predictions"]

    proba = {
        "tfidf": cache["tfidf_test_proba"],
        "finbert": fb["test_proba"],
        "llm": cache["llm_test_proba"],
        "lexicon": cache["lex_test_proba"],
    }
    proba_val = {
        "tfidf": cache["tfidf_val_proba"],
        "finbert": fb["val_proba"],
        "llm": cache["llm_val_proba"],
        "lexicon": cache["lex_val_proba"],
    }
    argmax_test = {k: np.argmax(v, axis=1) for k, v in proba.items()}

    new_preds: dict[str, list[str]] = dict(old_preds)
    new_results_rows: dict[str, dict] = {}

    new_preds["TF-IDF + LogReg"] = _argmax_labels(proba["tfidf"])
    new_preds["FinBERT (fine-tuned)"] = _argmax_labels(proba["finbert"])
    new_preds["Single LLM (zero_shot)"] = _argmax_labels(proba["llm"])
    new_preds["Loughran-McDonald Lexicon"] = _argmax_labels(proba["lexicon"])

    new_preds["Ensemble (majority)"] = _majority_vote(
        [argmax_test[a] for a in ("tfidf", "finbert", "llm", "lexicon")]
    )
    new_preds["Ensemble (weighted_average)"] = _weighted_average(
        [proba[a] for a in ("tfidf", "finbert", "llm", "lexicon")]
    )

    full = ("tfidf", "finbert", "llm", "lexicon")
    Xv = np.hstack([proba_val[a] for a in full])
    Xt = np.hstack([proba[a] for a in full])
    new_preds["Ensemble (stacking)"] = _stack_predict(Xv, y_val, Xt)

    for dropped in full:
        active = tuple(a for a in full if a != dropped)
        Xv_a = np.hstack([proba_val[a] for a in active])
        Xt_a = np.hstack([proba[a] for a in active])
        nm = f"Ablation: drop {dropped} (stacking)"
        new_preds[nm] = _stack_predict(Xv_a, y_val, Xt_a)

    rows_to_update = [
        ("Loughran-McDonald Lexicon", {"source": "pysentiment2 (stemmed)"}),
        ("Ensemble (majority)", {"strategy": "majority", "n_agents": 4}),
        (
            "Ensemble (weighted_average)",
            {
                "strategy": "weighted_average",
                "n_agents": 4,
                "weights": "uniform (1.0, 1.0, 1.0, 1.0)",
            },
        ),
        (
            "Ensemble (stacking)",
            {
                "strategy": "stacking",
                "n_agents": 4,
                "agents": list(full),
                "_note": "refit from updated cache after lexicon stem fix",
            },
        ),
    ] + [
        (
            f"Ablation: drop {d} (stacking)",
            {"strategy": "stacking", "dropped": d, "active_agents": [a for a in full if a != d]},
        )
        for d in full
    ]

    for nm, md in rows_to_update:
        new_results_rows[nm] = _metrics_row(nm, y_test_str, new_preds[nm], metadata=md)

    print(f"  {'system':45s} {'old_acc':>8s} {'new_acc':>8s} {'delta_pp':>9s}")
    deltas = {}
    for nm in new_preds:
        old_acc = accuracy_score(y_test_str, old_preds[nm]) if nm in old_preds else float("nan")
        new_acc = accuracy_score(y_test_str, new_preds[nm])
        delta = (new_acc - old_acc) * 100 if not np.isnan(old_acc) else 0.0
        deltas[nm] = (old_acc, new_acc, delta)
        flag = " *" if abs(delta) >= 0.5 else ""
        print(f"  {nm:45s} {old_acc:8.4f} {new_acc:8.4f} {delta:+8.2f}{flag}")

    if dry_run:
        print("  [dry-run: not writing]")
        return deltas

    pj_new = {
        "y_true": pj_old["y_true"],
        "predictions": new_preds,
    }

    for k in ("probabilities", "probabilities_note"):
        if k in pj_old:
            pj_new[k] = pj_old[k]
    preds_path.write_text(json.dumps(pj_new, indent=2))
    print(f"  wrote {preds_path}")

    ar_path = rdir / "all_results.json"
    ar = json.loads(ar_path.read_text())
    by_name = {r["model_name"]: r for r in ar}
    for nm, row in new_results_rows.items():
        if nm in by_name:
            by_name[nm].update(row)
        else:
            ar.append(row)
    ar_path.write_text(json.dumps(ar, indent=2))
    print(f"  wrote {ar_path}")

    mc = pairwise_mcnemar(y_test_str, new_preds)
    cis = []
    for nm, preds in new_preds.items():
        cis.append(bootstrap_ci(y_test_str, preds, metric="accuracy", model_name=nm))
        cis.append(bootstrap_ci(y_test_str, preds, metric="weighted_f1", model_name=nm))
        cis.append(bootstrap_ci(y_test_str, preds, metric="macro_f1", model_name=nm))
    save_statistical_report(mc, cis, None, rdir / "statistical_report.json")
    print(f"  wrote {rdir / 'statistical_report.json'}")

    return deltas


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Print delta report without writing.")
    p.add_argument("--datasets", nargs="+", default=["twitter", "phrasebank"])
    args = p.parse_args(argv)
    for ds in args.datasets:
        regen_dataset(ds, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
