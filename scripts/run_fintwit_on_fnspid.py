#!/usr/bin/env python3
"""Run a PhraseBank-fine-tuned FinTwitBERT on every FNSPID headline, then
refit the 5-agent stacking meta-learner on the PhraseBank validation cache
(matching ``patch_fnspid_predictions.py``'s meta-source choice) and write
both ``fintwit_*`` fields and ``ensemble5_*`` fields to
``results/fnspid_optc_v2/predictions.json`` in place.

Why PhraseBank as the FinTwit fine-tuning source?
- FNSPID rows are news headlines, closer in style to PhraseBank's analyst
  sentences than to ticker-laden tweets.
- The existing 4-agent ensemble already meta-learns on PhraseBank's
  validation split, so using the same source keeps every downstream
  comparison apples-to-apples.

After running this script, re-run ``scripts/downstream_metrics.py`` on
``results/fnspid_optc_v2/predictions.json`` to recompute the IC, hit
rate and conditional-return numbers for the new ``fintwit`` and
``ensemble5`` systems alongside the existing four.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)
from torch.utils.data import Dataset as TorchDataset

from mas.config import DataConfig, LABELS, RESULTS_DIR
from mas.data.loader import load_financial_phrasebank

hf_logging.set_verbosity_error()

HF_NAME = "StephanAkkerman/FinTwitBERT-sentiment"
NATIVE_TO_CANON = {"BULLISH": "positive", "BEARISH": "negative", "NEUTRAL": "neutral"}
LBL_TO_INT = {l: i for i, l in enumerate(LABELS)}
META_SOURCE = "phrasebank"
FNSPID_PRED = RESULTS_DIR / "fnspid_optc_v2" / "predictions.json"

BATCH = 16
MAX_LEN = 128
NUM_EPOCHS = 3
LR = 2e-5


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


def main() -> int:
    if not FNSPID_PRED.exists():
        raise FileNotFoundError(FNSPID_PRED)
    blob = json.loads(FNSPID_PRED.read_text())
    rows = blob["rows"]
    n = len(rows)
    print(f"loaded {n} FNSPID rows")

    print("\n[1/3] Fine-tuning FinTwitBERT on PhraseBank")
    cfg = DataConfig(dataset_name="warwickai/financial_phrasebank_mirror")
    train, val, test = load_financial_phrasebank(cfg)
    train_texts = [str(t) for t in train.texts]
    train_labels = [str(l) for l in train.labels]
    val_texts = [str(t) for t in val.texts]
    val_labels = [str(l) for l in val.labels]
    print(f"  PhraseBank train={len(train_texts)} val={len(val_texts)}")

    tok = AutoTokenizer.from_pretrained(HF_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(HF_NAME)
    id2label_native = {int(k): str(v) for k, v in model.config.id2label.items()}
    canon_to_native = {}
    for nid, nname in id2label_native.items():
        canon = NATIVE_TO_CANON.get(nname.upper())
        if canon is None:
            raise ValueError(f"cannot map native label {nname}")
        canon_to_native[canon] = nid

    train_enc = tok(
        train_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
    )
    val_enc = tok(val_texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    train_lab = [canon_to_native[lab] for lab in train_labels]
    val_lab = [canon_to_native[lab] for lab in val_labels]

    args = TrainingArguments(
        output_dir="./_fintwit_fnspid_trainer",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        learning_rate=LR,
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
    print(f"  fine-tune took {time.time() - t0:.1f}s")
    model = trainer.model
    device = _device()
    model.to(device).eval()
    print(f"  device = {device}")

    print(f"\n[2/3] Running FinTwitBERT on {n} FNSPID headlines")
    fnspid_texts = [str(r["text"]) for r in rows]
    col_idx = [canon_to_native[lbl] for lbl in LABELS]
    out_proba = np.empty((n, 3), dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, BATCH):
        batch = fnspid_texts[i : i + BATCH]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            logits = model(**enc).logits
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        out_proba[i : i + len(batch)] = p[:, col_idx]
        if (i // BATCH) % 50 == 0:
            done = min(i + BATCH, n)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            print(f"    {done}/{n} ({rate:.1f} rows/s)")
    print(f"  inference took {time.time() - t0:.1f}s")

    fintwit_pred_labels = [LABELS[int(j)] for j in np.argmax(out_proba, axis=1)]
    for r, p, lab in zip(rows, out_proba, fintwit_pred_labels):
        r["fintwit_proba"] = [float(x) for x in p]
        r["fintwit_pred"] = lab

    print("\n[3/3] Refitting 5-agent meta-learner on PhraseBank val cache")
    ag = np.load(RESULTS_DIR / META_SOURCE / "_cache_agents.npz", allow_pickle=True)
    fb = np.load(RESULTS_DIR / META_SOURCE / "_cache_finbert.npz", allow_pickle=True)
    ft = np.load(RESULTS_DIR / META_SOURCE / "_cache_fintwit.npz", allow_pickle=True)
    y_val = np.array([LBL_TO_INT[str(s)] for s in ag["val_labels"]])
    X_val = np.hstack(
        [
            ag["tfidf_val_proba"],
            fb["val_proba"],
            ag["llm_val_proba"],
            ag["lex_val_proba"],
            ft["val_proba"],
        ]
    )
    meta5 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    meta5.fit(X_val, y_val)
    print(f"  5-agent meta fit on n_val={len(y_val)}; classes={meta5.classes_.tolist()}")

    tfidf = np.array([r["tfidf_proba"] for r in rows], dtype=np.float64)
    finbert = np.array([r["finbert_proba"] for r in rows], dtype=np.float64)
    llm = np.array([r["llm_proba"] for r in rows], dtype=np.float64)
    lex = np.array([r["lex_proba"] for r in rows], dtype=np.float64)
    X_fnspid = np.hstack([tfidf, finbert, llm, lex, out_proba.astype(np.float64)])
    ens5_proba = meta5.predict_proba(X_fnspid)
    ens5_pred = [LABELS[int(i)] for i in np.argmax(ens5_proba, axis=1)]
    for r, p, lab in zip(rows, ens5_proba, ens5_pred):
        r["ensemble5_proba"] = [float(x) for x in p]
        r["ensemble5_pred"] = lab

    meta = blob.setdefault("meta", {})
    meta["fintwit"] = {
        "hf_name": HF_NAME,
        "fine_tuned_on": META_SOURCE,
        "ft_seconds": float(time.time() - t0),
    }
    meta["ensemble5"] = {
        "meta_source": META_SOURCE,
        "n_val": int(len(y_val)),
        "agents": ["tfidf", "finbert", "llm", "lex", "fintwit"],
    }
    meta["_fintwit_added"] = True

    FNSPID_PRED.write_text(json.dumps(blob, indent=2))
    print(f"\n  patched {FNSPID_PRED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
