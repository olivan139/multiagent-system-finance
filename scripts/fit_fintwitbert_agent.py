#!/usr/bin/env python3
"""Fine-tune FinTwitBERT-sentiment (Akkerman 2023) as a 5th base agent
and cache its validation + test probability vectors aligned to the
canonical LABELS = [negative, neutral, positive] order.

The model's native id2label is {0: NEUTRAL, 1: BULLISH, 2: BEARISH}.
We map BULLISH -> positive, BEARISH -> negative, NEUTRAL -> neutral and
fine-tune for ``num_epochs`` epochs with the same hyper-parameters as
the production FinBERT agent so the comparison is apples-to-apples.

Outputs:
    python/results/<dataset>/_cache_fintwit.npz
        keys: val_proba (N, 3), test_proba (N, 3),
              val_labels, test_labels, test_pred_labels,
              test_accuracy, ft_wall_seconds
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)

from mas.config import DataConfig
from mas.data.loader import load_financial_phrasebank

hf_logging.set_verbosity_error()

HF_NAME = "StephanAkkerman/FinTwitBERT-sentiment"
LABEL_ORDER = ["negative", "neutral", "positive"]


NATIVE_TO_CANON = {
    "BULLISH": "positive",
    "BEARISH": "negative",
    "NEUTRAL": "neutral",
}


class _Ds(TorchDataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fit_and_cache(
    dataset_name: str,
    *,
    num_epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    learning_rate: float = 2e-5,
) -> dict:
    """Fine-tune FinTwitBERT on ``dataset_name`` and save the cache."""
    print(f"\n========== {dataset_name} ==========")
    _DATASET_HF_NAME = {
        "twitter": "zeroshot/twitter-financial-news-sentiment",
        "phrasebank": "warwickai/financial_phrasebank_mirror",
        "semeval2017": "semeval2017",
        "fiqa2018": "fiqa2018",
    }
    hf_name = _DATASET_HF_NAME.get(dataset_name)
    if hf_name is None:
        raise ValueError(
            f"Unknown dataset_name {dataset_name!r}; " f"expected one of {list(_DATASET_HF_NAME)}"
        )
    cfg = DataConfig(dataset_name=hf_name)
    train, val, test = load_financial_phrasebank(cfg)
    train_texts = [str(t) for t in train.texts]
    train_labels = [str(l) for l in train.labels]
    val_texts = [str(t) for t in val.texts]
    val_labels = [str(l) for l in val.labels]
    test_texts = [str(t) for t in test.texts]
    test_labels = [str(l) for l in test.labels]
    print(f"train={len(train_texts)}  val={len(val_texts)}  test={len(test_texts)}")

    tok = AutoTokenizer.from_pretrained(HF_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(HF_NAME)
    id2label_native = {int(k): str(v) for k, v in model.config.id2label.items()}
    label2id_native = {v: k for k, v in id2label_native.items()}
    print(f"native id2label: {id2label_native}")

    canon_to_native: dict[str, int] = {}
    for nid, nname in id2label_native.items():
        canon = NATIVE_TO_CANON.get(nname.upper())
        if canon is None:
            raise ValueError(f"Cannot map native label {nname} for {HF_NAME}")
        canon_to_native[canon] = nid

    train_enc = tok(
        train_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    val_enc = tok(
        val_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    test_enc = tok(
        test_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )

    train_lab = [canon_to_native[lab] for lab in train_labels]
    val_lab = [canon_to_native[lab] for lab in val_labels]
    test_lab = [canon_to_native[lab] for lab in test_labels]

    args = TrainingArguments(
        output_dir=f"./fintwit_output_{dataset_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=42,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_Ds(train_enc, train_lab),
        eval_dataset=_Ds(val_enc, val_lab),
    )

    t0 = time.time()
    trainer.train()
    ft_seconds = time.time() - t0
    print(f"fine-tune took {ft_seconds:.1f}s")

    model = trainer.model
    device = _device()
    model.to(device).eval()

    def predict_proba(texts: list[str]) -> np.ndarray:
        probs_native = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_native.append(p)
        return np.vstack(probs_native)

    val_proba_native = predict_proba(val_texts)
    test_proba_native = predict_proba(test_texts)

    col_idx = [canon_to_native[lbl] for lbl in LABEL_ORDER]
    val_proba = val_proba_native[:, col_idx].astype(np.float32)
    test_proba = test_proba_native[:, col_idx].astype(np.float32)

    test_pred_ids = np.argmax(test_proba, axis=1)
    test_pred_labels = np.array([LABEL_ORDER[i] for i in test_pred_ids])
    test_acc = accuracy_score(test_labels, test_pred_labels)
    test_mf1 = f1_score(test_labels, test_pred_labels, average="macro")
    print(f"FinTwitBERT fine-tuned: test_acc={test_acc:.4f}  test_mf1={test_mf1:.4f}")

    out = REPO_ROOT / "results" / dataset_name / "_cache_fintwit.npz"
    np.savez(
        out,
        val_proba=val_proba,
        test_proba=test_proba,
        val_labels=np.array(val_labels),
        test_labels=np.array(test_labels),
        test_pred_labels=test_pred_labels,
        test_accuracy=np.array([test_acc]),
        test_macro_f1=np.array([test_mf1]),
        ft_wall_seconds=np.array([ft_seconds]),
    )
    print(f"wrote {out}")
    return {
        "dataset": dataset_name,
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_mf1),
        "ft_seconds": float(ft_seconds),
        "cache_path": str(out),
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+", default=["twitter", "phrasebank", "semeval2017", "fiqa2018"]
    )
    args = p.parse_args()
    summary = []
    for name in args.datasets:
        summary.append(fit_and_cache(name))
    print("\n=== Summary ===")
    for r in summary:
        print(
            f"{r['dataset']:<11} acc={r['test_accuracy']:.4f}  "
            f"mf1={r['test_macro_f1']:.4f}  ft={r['ft_seconds']:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
