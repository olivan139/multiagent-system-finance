#!/usr/bin/env python3
"""Regenerate the lexicon probability/label entries in
``_cache_agents.npz`` for both datasets, using the fixed lexicon
implementation (Porter-stemmed tokenisation that matches the
pysentiment2 LM dictionary).

We keep every other key in the cache file untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mas.baselines.lexicon import LoughranMcDonaldAgent
from mas.config import DataConfig, LABELS
from mas.data.loader import load_financial_phrasebank


def regen(dataset_name: str) -> None:
    print(f"\n=== {dataset_name} ===")
    cfg = DataConfig(
        dataset_name=(
            "zeroshot/twitter-financial-news-sentiment"
            if dataset_name == "twitter"
            else "warwickai/financial_phrasebank_mirror"
        )
    )
    _, val, test = load_financial_phrasebank(cfg)
    val_texts = [str(t) for t in val.texts]
    test_texts = [str(t) for t in test.texts]

    agent = LoughranMcDonaldAgent()
    print(
        f"lexicon source: {agent.source}, |pos|={len(agent.positive)} |neg|={len(agent.negative)}"
    )

    val_proba = agent.predict_proba(val_texts, label_order=list(LABELS)).astype(np.float32)
    test_proba = agent.predict_proba(test_texts, label_order=list(LABELS)).astype(np.float32)
    test_pred_labels = np.array(agent.predict(test_texts))

    cache_path = REPO_ROOT / "results" / dataset_name / "_cache_agents.npz"
    z = np.load(cache_path, allow_pickle=True)
    new = {k: z[k] for k in z.files}
    old_test_acc = float((np.array(z["lex_test_pred_labels"]) == np.array(z["test_labels"])).mean())
    new["lex_val_proba"] = val_proba
    new["lex_test_proba"] = test_proba
    new["lex_test_pred_labels"] = test_pred_labels
    np.savez(cache_path, **new)

    new_test_acc = float((test_pred_labels == np.array(z["test_labels"])).mean())
    print(
        f"lexicon test acc  : old={old_test_acc:.4f}  new={new_test_acc:.4f}  "
        f"delta={(new_test_acc - old_test_acc)*100:+.2f} pp"
    )
    print(f"wrote {cache_path}")


def main() -> int:
    for name in ("twitter", "phrasebank"):
        regen(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
