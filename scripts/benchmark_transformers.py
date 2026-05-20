#!/usr/bin/env python3
"""Benchmark several public pre-trained sentiment transformers on the
Twitter and PhraseBank test splits in zero-shot mode.

For each candidate we report:
  - accuracy and macro-F1 on the test split
  - agreement (Cohen's kappa) with each existing base agent on the test
    split, which lets us pick the candidate that adds the most diversity
    rather than the candidate with the highest standalone score.

Outputs:
  python/results/transformer_benchmark.json
  Console table.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging as hf_logging,
)

from mas.config import DataConfig
from mas.data.loader import load_financial_phrasebank

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

RESULTS = REPO_ROOT / "results"
SUMMARY_PATH = RESULTS / "transformer_benchmark.json"

LABEL_ORDER = ["negative", "neutral", "positive"]
LABEL_TO_INT = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}


CANDIDATES = [
    {
        "id": "prosus_zeroshot",
        "hf_name": "ProsusAI/finbert",
        "note": "FinBERT (Araci 2019), zero-shot. Production reference.",
        "label_map": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
    },
    {
        "id": "finbert_tone",
        "hf_name": "yiyanghkust/finbert-tone",
        "note": "FinBERT-Tone (Yang et al. 2020), Bloomberg corpus.",
        "label_map": {"Positive": "positive", "Negative": "negative", "Neutral": "neutral"},
    },
    {
        "id": "distilroberta_finnews",
        "hf_name": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
        "note": "DistilRoBERTa fine-tuned on financial-news sentiment.",
        "label_map": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
    },
    {
        "id": "financial_bert",
        "hf_name": "ahmedrachid/FinancialBERT-Sentiment-Analysis",
        "note": "FinancialBERT (Hazourli 2022) fine-tuned for sentiment.",
        "label_map": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
    },
    {
        "id": "twitter_roberta",
        "hf_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "note": "Cardiff Twitter-RoBERTa (Loureiro et al. 2022), 124M tweets.",
        "label_map": {"positive": "positive", "negative": "negative", "neutral": "neutral"},
    },
    {
        "id": "fintwitbert",
        "hf_name": "StephanAkkerman/FinTwitBERT-sentiment",
        "note": "FinTwitBERT-sentiment (Akkerman 2023), finance Twitter.",
        "label_map": {"BULLISH": "positive", "BEARISH": "negative", "NEUTRAL": "neutral"},
    },
]


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _enc(labels: list[str]) -> np.ndarray:
    return np.array([LABEL_TO_INT[str(s)] for s in labels], dtype=np.int64)


def predict_with_model(
    hf_name: str,
    texts: list[str],
    label_map: dict[str, str],
    device: str,
    batch_size: int = 16,
    max_length: int = 128,
) -> tuple[list[str], np.ndarray]:
    """Run zero-shot inference; map output labels back to {neg, neu, pos}."""
    tok = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name)
    model.to(device).eval()
    id2label_native = {int(k): v for k, v in model.config.id2label.items()}

    canonical_for_native: dict[int, str | None] = {}
    for nid, nlabel in id2label_native.items():
        canon = label_map.get(nlabel)
        if canon is None:

            for k, v in label_map.items():
                if str(k).lower() == str(nlabel).lower():
                    canon = v
                    break
        canonical_for_native[nid] = canon

    if any(v is None for v in canonical_for_native.values()):
        raise ValueError(
            f"{hf_name}: cannot map every native label. native={id2label_native}, "
            f"label_map={label_map}"
        )

    preds: list[str] = []
    all_probs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        ids = np.argmax(probs, axis=-1)
        preds.extend(canonical_for_native[int(p)] for p in ids)

    probs_native = np.vstack(all_probs)
    proba_canon = np.zeros((probs_native.shape[0], 3), dtype=np.float32)
    for nid, canon in canonical_for_native.items():
        col = LABEL_TO_INT[canon]
        proba_canon[:, col] += probs_native[:, nid]
    return preds, proba_canon


def load_test_text(name: str) -> tuple[list[str], list[str]]:
    cfg = DataConfig(
        dataset_name=(
            "zeroshot/twitter-financial-news-sentiment"
            if name == "twitter"
            else "warwickai/financial_phrasebank_mirror"
        )
    )
    train, val, test = load_financial_phrasebank(cfg)
    return test.texts, test.labels


def load_existing_agent_preds(name: str) -> dict[str, np.ndarray]:
    """Pull test predictions from the per-dataset prediction caches.

    The lexicon key is stored under different names depending on which
    pipeline ran first, so we load it from ``_cache_agents.npz`` for
    consistency across datasets.
    """
    with open(RESULTS / name / "predictions.json") as f:
        p = json.load(f)
    preds = p["predictions"]
    cache = np.load(RESULTS / name / "_cache_agents.npz", allow_pickle=True)
    return {
        "tfidf": np.array(preds["TF-IDF + LogReg"]),
        "finbert": np.array(preds["FinBERT (fine-tuned)"]),
        "llm": np.array(preds["Single LLM (zero_shot)"]),
        "lexicon": np.array([str(x) for x in cache["lex_test_pred_labels"]]),
        "y_true": np.array(p["y_true"]),
    }


def main() -> int:
    device = _device()
    print(f"Device: {device}")

    all_results: list[dict] = []
    for ds_name in ("twitter", "phrasebank"):
        print(f"\n========== {ds_name} ==========")
        texts, y_true = load_test_text(ds_name)
        existing = load_existing_agent_preds(ds_name)
        assert list(existing["y_true"]) == list(
            y_true
        ), f"label order mismatch for {ds_name}: cache vs.\\ fresh-load disagree."

        y_idx = _enc(y_true)
        ds_block = {"dataset": ds_name, "n_test": int(len(y_true)), "models": []}
        for cand in CANDIDATES:
            print(f"\n  [{cand['id']:<22}] {cand['hf_name']}")
            t0 = time.time()
            try:
                preds, _ = predict_with_model(
                    cand["hf_name"],
                    texts,
                    cand["label_map"],
                    device,
                )
            except Exception as exc:
                print(f"    SKIP: {exc}")
                ds_block["models"].append(
                    {
                        "id": cand["id"],
                        "hf_name": cand["hf_name"],
                        "error": str(exc),
                    }
                )
                continue
            dt = time.time() - t0
            p_idx = _enc(preds)
            acc = accuracy_score(y_idx, p_idx)
            mf1 = f1_score(y_idx, p_idx, average="macro")
            wf1 = f1_score(y_idx, p_idx, average="weighted")
            kap = cohen_kappa_score(y_idx, p_idx)
            agreement = {
                k: float(
                    cohen_kappa_score(
                        _enc(existing[k].tolist()),
                        p_idx,
                    )
                )
                for k in ("tfidf", "finbert", "llm", "lexicon")
            }
            print(
                f"    acc={acc:.4f}  mf1={mf1:.4f}  wf1={wf1:.4f}  kappa={kap:.4f}"
                f"   ({dt:.1f}s)"
            )
            print(
                f"    agreement (kappa) vs existing agents: "
                f"tfidf={agreement['tfidf']:.3f} "
                f"finbert={agreement['finbert']:.3f} "
                f"llm={agreement['llm']:.3f} "
                f"lex={agreement['lexicon']:.3f}"
            )
            ds_block["models"].append(
                {
                    "id": cand["id"],
                    "hf_name": cand["hf_name"],
                    "note": cand["note"],
                    "accuracy": float(acc),
                    "macro_f1": float(mf1),
                    "weighted_f1": float(wf1),
                    "kappa": float(kap),
                    "agreement_kappa_vs": agreement,
                    "infer_seconds": float(dt),
                }
            )
        all_results.append(ds_block)

    SUMMARY_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
