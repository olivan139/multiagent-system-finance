#!/usr/bin/env python3
"""Update ``all_results.json`` for both datasets without re-running the
LLM. We re-use ``_cache_agents.npz`` and ``_cache_finbert.npz`` to:

  * refresh the Loughran-McDonald Lexicon row (now stemmed),
  * refit the 4-agent stacking meta-learner and refresh its row,
  * refresh the four single-agent ablations
    (drop tfidf / drop finbert / drop llm / drop lexicon),
  * leave every other row (per-agent baselines other than lexicon,
    Multi-Agent Pipeline, debate, etc.) unchanged.

Run on Twitter and PhraseBank.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
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

LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}


def _stacking_predict(X_val, y_val, X_test, classes):
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    clf.fit(X_val, y_val)
    return clf.predict(X_test)


def _metrics(name: str, y_true: list[str], y_pred: list[str], metadata: dict | None = None) -> dict:
    labels = list(LABELS)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    rep = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
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
        "latency_ms_per_sample": None,
        "cost_usd_per_sample": 0.0,
        "total_samples": int(len(y_true)),
        "metadata": metadata or {},
    }


def _to_labels(idx_arr, classes=LABELS) -> list[str]:
    return [classes[int(i)] for i in idx_arr]


def refresh_dataset(name: str) -> None:
    print(f"\n=== {name} ===")
    rdir = RESULTS_DIR / name
    cache = np.load(rdir / "_cache_agents.npz", allow_pickle=True)
    fb = np.load(rdir / "_cache_finbert.npz", allow_pickle=True)
    y_val_str = [str(s) for s in cache["val_labels"]]
    y_test_str = [str(s) for s in cache["test_labels"]]
    y_val = np.array([LABEL_TO_INT[s] for s in y_val_str])
    y_test = np.array([LABEL_TO_INT[s] for s in y_test_str])

    rows = json.loads((rdir / "all_results.json").read_text())
    by_name = {r["model_name"]: r for r in rows}

    new_lex = _metrics(
        "Loughran-McDonald Lexicon",
        y_test_str,
        _to_labels(np.argmax(cache["lex_test_proba"], axis=1)),
        metadata={"source": "pysentiment2 (stemmed)"},
    )
    if "Loughran-McDonald Lexicon" in by_name:
        by_name["Loughran-McDonald Lexicon"].update(new_lex)
    else:
        rows.append(new_lex)
    print(f"  lex: acc={new_lex['accuracy']:.4f} mf1={new_lex['macro_f1']:.4f}")

    AGENTS = [
        ("tfidf", cache["tfidf_val_proba"], cache["tfidf_test_proba"]),
        ("finbert", fb["val_proba"], fb["test_proba"]),
        ("llm", cache["llm_val_proba"], cache["llm_test_proba"]),
        ("lexicon", cache["lex_val_proba"], cache["lex_test_proba"]),
    ]

    def build(active):
        Xv = np.hstack([a[1] for a in AGENTS if a[0] in active])
        Xt = np.hstack([a[2] for a in AGENTS if a[0] in active])
        return Xv, Xt

    full = ("tfidf", "finbert", "llm", "lexicon")
    Xv, Xt = build(full)
    preds = _stacking_predict(Xv, y_val, Xt, LABELS)
    ens_pred_labels = _to_labels(preds)
    new_stack = _metrics(
        "Ensemble (stacking)",
        y_test_str,
        ens_pred_labels,
        metadata={
            "strategy": "stacking",
            "n_agents": 4,
            "agents": list(full),
            "_note": "refit from updated cache after lexicon stem fix",
        },
    )
    if "Ensemble (stacking)" in by_name:
        by_name["Ensemble (stacking)"].update(new_stack)
    else:
        rows.append(new_stack)
    print(f"  4-agent stack: acc={new_stack['accuracy']:.4f} " f"mf1={new_stack['macro_f1']:.4f}")

    for dropped in full:
        active = tuple(a for a in full if a != dropped)
        Xv, Xt = build(active)
        preds = _stacking_predict(Xv, y_val, Xt, LABELS)
        nm = f"Ablation: drop {dropped} (stacking)"
        new_abl = _metrics(
            nm,
            y_test_str,
            _to_labels(preds),
            metadata={"strategy": "stacking", "dropped": dropped, "active_agents": list(active)},
        )
        if nm in by_name:
            by_name[nm].update(new_abl)
        else:
            rows.append(new_abl)
        print(
            f"  ablation drop {dropped}: acc={new_abl['accuracy']:.4f} "
            f"mf1={new_abl['macro_f1']:.4f}"
        )

    (rdir / "all_results.json").write_text(json.dumps(rows, indent=2))
    print(f"  wrote {rdir/'all_results.json'}")


def main() -> int:
    refresh_dataset("twitter")
    refresh_dataset("phrasebank")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
