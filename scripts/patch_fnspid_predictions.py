#!/usr/bin/env python3
"""Patch a saved FNSPID ``predictions.json`` in place with:
  * corrected ``lex_proba``/``lex_pred`` (stemmed lexicon, see review),
  * a re-fitted stacking meta-learner trained on the refreshed
    ``_cache_agents.npz`` from the meta-source dataset.

The meta-source defaults to PhraseBank (matching ``--meta-source phrasebank``
in ``downstream_signal_test.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mas.baselines.lexicon import LoughranMcDonaldAgent
from mas.config import LABELS, RESULTS_DIR
from mas.data.preprocessing import preprocess_batch
from sklearn.linear_model import LogisticRegression

LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}


def _fit_meta(meta_source: str, drop_llm: bool):
    cache = np.load(RESULTS_DIR / meta_source / "_cache_agents.npz", allow_pickle=True)
    fb = np.load(RESULTS_DIR / meta_source / "_cache_finbert.npz", allow_pickle=True)
    y_val = np.array([LABEL_TO_INT[str(s)] for s in cache["val_labels"]])
    if drop_llm:
        X_val = np.hstack([cache["tfidf_val_proba"], fb["val_proba"], cache["lex_val_proba"]])
    else:
        X_val = np.hstack(
            [
                cache["tfidf_val_proba"],
                fb["val_proba"],
                cache["llm_val_proba"],
                cache["lex_val_proba"],
            ]
        )
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    clf.fit(X_val, y_val)
    return clf


def patch(pred_path: Path, meta_source: str) -> None:
    print(f"\n=== {pred_path} ===")
    blob = json.loads(pred_path.read_text())
    rows = blob["rows"]
    drop_llm = "llm_proba" not in rows[0]
    print(f"  rows={len(rows)}  drop_llm={drop_llm}  meta_source={meta_source}")

    texts = [r["text"] for r in rows]
    pre = preprocess_batch(texts)

    lex = LoughranMcDonaldAgent()
    new_lex_proba = lex.predict_proba(pre, label_order=list(LABELS))
    new_lex_pred = [LABELS[int(np.argmax(p))] for p in new_lex_proba]

    print(f"  fitting meta from {meta_source} cache " f"({'3-agent' if drop_llm else '4-agent'})")
    meta = _fit_meta(meta_source=meta_source, drop_llm=drop_llm)

    tfidf = np.array([r["tfidf_proba"] for r in rows], dtype=np.float64)
    fb = np.array([r["finbert_proba"] for r in rows], dtype=np.float64)
    if drop_llm:
        X = np.hstack([tfidf, fb, new_lex_proba])
    else:
        llm = np.array([r["llm_proba"] for r in rows], dtype=np.float64)
        X = np.hstack([tfidf, fb, llm, new_lex_proba])
    ens_proba = meta.predict_proba(X)
    ens_pred = [LABELS[int(i)] for i in np.argmax(ens_proba, axis=1)]

    old_lex = [r["lex_pred"] for r in rows]
    old_ens = [r["ensemble_pred"] for r in rows]
    lex_chg = sum(1 for a, b in zip(old_lex, new_lex_pred) if a != b)
    ens_chg = sum(1 for a, b in zip(old_ens, ens_pred) if a != b)
    print(f"  lex_pred changed in {lex_chg}/{len(rows)} rows " f"({lex_chg/len(rows):.1%})")
    print(f"  ensemble_pred changed in {ens_chg}/{len(rows)} rows " f"({ens_chg/len(rows):.1%})")

    for i, r in enumerate(rows):
        r["lex_proba"] = new_lex_proba[i].tolist()
        r["lex_pred"] = new_lex_pred[i]
        r["ensemble_proba"] = ens_proba[i].tolist()
        r["ensemble_pred"] = ens_pred[i]
    blob["meta"]["_patched"] = {
        "lex_pred_changed": int(lex_chg),
        "ensemble_pred_changed": int(ens_chg),
        "meta_source": meta_source,
        "patch_reason": ("fixed lexicon stem mismatch; refit meta from updated cache"),
    }
    pred_path.write_text(json.dumps(blob, indent=2))
    print(f"  wrote {pred_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--meta-source", choices=["phrasebank", "twitter"], default="phrasebank")
    p.add_argument("--targets", nargs="*", default=["fnspid_optc_v2", "fnspid_optc_30k_noLLM"])
    args = p.parse_args()
    for name in args.targets:
        patch(RESULTS_DIR / name / "predictions.json", meta_source=args.meta_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
