#!/usr/bin/env python3
"""Three additional evidence-strengthening experiments for the 5-agent stack
vs FinTwitBERT-alone comparison.

Per dataset, we compute:

  (A) Calibration: Brier score (sum-of-squared-error against one-hot),
      negative log-likelihood, and Expected Calibration Error with 15 bins.
      Lower is better on all three. Calibration is sample-size-robust in
      a way McNemar isn't.

  (B) Selective prediction (accuracy @ coverage): sort each system by its
      max-class predicted probability and report accuracy on the top-K
      most-confident predictions for K in {100%, 75%, 50%, 25%}. This is
      the "production deployment" story: if the system is allowed to
      abstain on its lowest-confidence cases, how good is it on the rest?

  (C) Disagreement-subset analysis: on items where the four non-FinTwit
      agents (TF-IDF, FinBERT, LLM, lexicon) majority-disagree with
      FinTwitBERT's prediction, who is right more often -- FinTwitBERT
      or the 5-stack? This isolates "what the other 4 agents contribute".

Writes one JSON summary at
``results/calibration_selective_disagreement.json``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mas.config import LABELS, RESULTS_DIR

LABEL_ORDER = list(LABELS)
LBL_TO_INT = {l: i for i, l in enumerate(LABEL_ORDER)}


def _enc(labels) -> np.ndarray:
    return np.asarray([LBL_TO_INT[str(x)] for x in labels])


def _brier(proba: np.ndarray, y_int: np.ndarray) -> float:
    """Multiclass Brier = mean over rows of sum_k (p_k - y_k)^2."""
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_int)), y_int] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _nll(proba: np.ndarray, y_int: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(proba[np.arange(len(y_int)), y_int], eps, 1.0)
    return float(-np.mean(np.log(p)))


def _ece(proba: np.ndarray, y_int: np.ndarray, n_bins: int = 15) -> float:
    """ECE: expected gap |accuracy(bin) - mean confidence(bin)| weighted by bin size."""
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    correct = (pred == y_int).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_int)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        bin_acc = float(correct[mask].mean())
        bin_conf = float(conf[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _selective(proba: np.ndarray, y_int: np.ndarray) -> dict:
    """Accuracy @ coverage curve at fixed coverages."""
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    n = len(y_int)
    order = np.argsort(-conf)
    pred_s = pred[order]
    y_s = y_int[order]
    out = {}
    for cov in (1.00, 0.75, 0.50, 0.25):
        k = max(1, int(round(cov * n)))
        out[f"acc@cov{int(cov*100):03d}"] = float(np.mean(pred_s[:k] == y_s[:k]))
    return out


def _fit_stack(
    blocks_val: list[np.ndarray], blocks_test: list[np.ndarray], y_val_int: np.ndarray
) -> np.ndarray:
    clf = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    clf.fit(np.hstack(blocks_val), y_val_int)
    return clf.predict_proba(np.hstack(blocks_test))


def _majority_pred(predmat: np.ndarray) -> np.ndarray:
    """Majority vote across rows of predmat (n_agents, n_rows)."""
    out = np.empty(predmat.shape[1], dtype=np.int64)
    for j in range(predmat.shape[1]):
        out[j] = Counter(predmat[:, j].tolist()).most_common(1)[0][0]
    return out


def _disagreement(other_preds: list[np.ndarray], ft_pred: np.ndarray) -> np.ndarray:
    """Boolean mask: True where >=2 of the 4 other agents disagree with FinTwit."""
    other = np.vstack(other_preds)
    disagree_count = np.sum(other != ft_pred[None, :], axis=0)
    return disagree_count >= 2


def evaluate(name: str) -> dict:
    ag = np.load(RESULTS_DIR / name / "_cache_agents.npz", allow_pickle=True)
    fb = np.load(RESULTS_DIR / name / "_cache_finbert.npz", allow_pickle=True)
    ft = np.load(RESULTS_DIR / name / "_cache_fintwit.npz", allow_pickle=True)

    yv = _enc(ag["val_labels"])
    yt = _enc(ag["test_labels"])

    b5v = [
        ag["tfidf_val_proba"],
        fb["val_proba"],
        ag["llm_val_proba"],
        ag["lex_val_proba"],
        ft["val_proba"],
    ]
    b5t = [
        ag["tfidf_test_proba"],
        fb["test_proba"],
        ag["llm_test_proba"],
        ag["lex_test_proba"],
        ft["test_proba"],
    ]

    p5 = _fit_stack(b5v, b5t, yv)
    pft = np.asarray(ft["test_proba"], dtype=np.float64)

    pft_pred = np.argmax(pft, axis=1)
    p5_pred = np.argmax(p5, axis=1)
    other_preds = [
        np.argmax(ag["tfidf_test_proba"], axis=1),
        np.argmax(fb["test_proba"], axis=1),
        np.argmax(ag["llm_test_proba"], axis=1),
        np.argmax(ag["lex_test_proba"], axis=1),
    ]

    calib = {
        "5stack": {
            "brier": _brier(p5, yt),
            "nll": _nll(p5, yt),
            "ece": _ece(p5, yt),
            "acc": float(np.mean(p5_pred == yt)),
        },
        "fintwit": {
            "brier": _brier(pft, yt),
            "nll": _nll(pft, yt),
            "ece": _ece(pft, yt),
            "acc": float(np.mean(pft_pred == yt)),
        },
    }

    selective = {
        "5stack": _selective(p5, yt),
        "fintwit": _selective(pft, yt),
    }

    mask = _disagreement(other_preds, pft_pred)
    n_dis = int(mask.sum())
    if n_dis > 0:
        ft_dis = float(np.mean(pft_pred[mask] == yt[mask]))
        s5_dis = float(np.mean(p5_pred[mask] == yt[mask]))
    else:
        ft_dis = s5_dis = float("nan")

    mask_a = ~mask
    n_agr = int(mask_a.sum())
    if n_agr > 0:
        ft_agr = float(np.mean(pft_pred[mask_a] == yt[mask_a]))
        s5_agr = float(np.mean(p5_pred[mask_a] == yt[mask_a]))
    else:
        ft_agr = s5_agr = float("nan")
    disagreement = {
        "n_test": int(len(yt)),
        "n_disagree": n_dis,
        "n_agree": n_agr,
        "fintwit_acc_disagree": ft_dis,
        "5stack_acc_disagree": s5_dis,
        "fintwit_acc_agree": ft_agr,
        "5stack_acc_agree": s5_agr,
        "delta_disagree_pp": None if np.isnan(ft_dis) else (s5_dis - ft_dis) * 100.0,
    }

    return {
        "dataset": name,
        "calibration": calib,
        "selective": selective,
        "disagreement": disagreement,
    }


def main() -> int:
    out: list[dict] = []
    for ds in ("twitter", "phrasebank", "semeval2017", "fiqa2018"):
        out.append(evaluate(ds))
    summary_path = RESULTS_DIR / "calibration_selective_disagreement.json"
    summary_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {summary_path}")

    print("\n" + "=" * 100)
    print("(A) Calibration  (lower Brier/NLL/ECE is better)")
    print("=" * 100)
    hdr = f"{'Dataset':<14}{'System':<11}{'Acc':>8}{'Brier':>10}{'NLL':>10}{'ECE':>10}"
    print(hdr)
    for r in out:
        for sys_name in ("5stack", "fintwit"):
            c = r["calibration"][sys_name]
            print(
                f"{r['dataset']:<14}{sys_name:<11}"
                f"{c['acc']:>8.4f}{c['brier']:>10.4f}{c['nll']:>10.4f}{c['ece']:>10.4f}"
            )

    print("\n" + "=" * 100)
    print("(B) Selective prediction  (accuracy @ coverage)")
    print("=" * 100)
    hdr = f"{'Dataset':<14}{'System':<11}{'@100%':>9}{'@75%':>9}{'@50%':>9}{'@25%':>9}"
    print(hdr)
    for r in out:
        for sys_name in ("5stack", "fintwit"):
            s = r["selective"][sys_name]
            print(
                f"{r['dataset']:<14}{sys_name:<11}"
                f"{s['acc@cov100']:>9.4f}{s['acc@cov075']:>9.4f}"
                f"{s['acc@cov050']:>9.4f}{s['acc@cov025']:>9.4f}"
            )

    print("\n" + "=" * 100)
    print("(C) Disagreement subset: items where >=2 of the 4 other agents")
    print("    disagree with FinTwitBERT's prediction")
    print("=" * 100)
    hdr = (
        f"{'Dataset':<14}{'n_dis':>7}{'n_agr':>7}{'FT_acc(dis)':>13}{'5S_acc(dis)':>13}{'Δ pp':>8}"
    )
    print(hdr)
    for r in out:
        d = r["disagreement"]
        delta = d["delta_disagree_pp"]
        delta_s = "n/a" if delta is None else f"{delta:+.1f}"
        print(
            f"{r['dataset']:<14}{d['n_disagree']:>7}{d['n_agree']:>7}"
            f"{d['fintwit_acc_disagree']:>13.4f}{d['5stack_acc_disagree']:>13.4f}{delta_s:>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
