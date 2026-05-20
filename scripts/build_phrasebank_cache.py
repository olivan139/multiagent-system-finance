#!/usr/bin/env python3
"""Build a Financial PhraseBank agent-prob cache for the Section 5.X
downstream test (Option C, news-domain meta-learner).

Outputs
-------
results/phrasebank/_cache_agents.npz  (schema mirrors twitter cache):
    tfidf_val_proba, tfidf_test_proba    -- (N, 3) in LABELS order
    tfidf_test_pred_labels               -- (N_test,) strings
    lex_val_proba,   lex_test_proba      -- (N, 3) in LABELS order
    lex_test_pred_labels                 -- (N_test,) strings
    llm_val_labels                       -- (N_val,) strings
    llm_val_confidence                   -- (N_val,)
    llm_val_proba                        -- (N_val, 3)
    llm_test_labels                      -- (N_test,) strings
    llm_test_proba                       -- (N_test, 3)
    llm_val_cost_usd, llm_val_tokens     -- scalars (1,)
    val_labels, test_labels              -- ground truth strings

results/phrasebank/_cache_finbert.npz:
    val_proba, test_proba                -- (N, 3)
    val_labels, test_labels              -- (N,) strings
    test_pred_labels                     -- (N_test,) strings
    test_accuracy, ft_wall_seconds       -- (1,)

results/finbert_finetuned_phrasebank/    -- saved HF model + tokenizer

Run with:
    python scripts/build_phrasebank_cache.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import Trainer, TrainingArguments

from mas.agents.ensemble import _llm_to_pseudo_proba
from mas.agents.single import SingleLLMAgent
from mas.baselines.lexicon import LoughranMcDonaldAgent
from mas.baselines.tfidf_logreg import TfidfLogRegBaseline
from mas.baselines.transformer import (
    TransformerBaseline,
    _SentimentTorchDataset,
)
from mas.config import LABEL2ID, LABELS, RESULTS_DIR, DataConfig, TransformerConfig
from mas.data import load_financial_phrasebank
from mas.data.preprocessing import preprocess_batch

PB_DIR = RESULTS_DIR / "phrasebank"
PB_DIR.mkdir(parents=True, exist_ok=True)
FT_DIR = RESULTS_DIR / "finbert_finetuned_phrasebank"
FT_DIR.mkdir(parents=True, exist_ok=True)


def _tfidf_aligned(model: TfidfLogRegBaseline, texts: list[str]) -> np.ndarray:
    raw = model.predict_proba(texts)
    classes = list(model.pipeline.classes_)
    out = np.zeros((len(texts), len(LABELS)), dtype=np.float64)
    for j, lab in enumerate(LABELS):
        if lab in classes:
            out[:, j] = raw[:, classes.index(lab)]
    return out


def _run_llm(
    agent: SingleLLMAgent, texts: list[str], n_workers: int = 8, cache_path: Path | None = None
) -> tuple[list[str], np.ndarray]:
    n = len(texts)
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())
    todo = [i for i in range(n) if str(i) not in cache]
    if todo:
        print(f"    [LLM] {len(todo)} new requests x {n_workers} workers")
        t0 = time.time()
        done = 0

        def _one(i: int) -> tuple[int, str, float]:
            r = agent.analyze(texts[i])
            return i, r.sentiment, float(r.confidence)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for fut in ex.map(_one, todo):
                i, lab, conf = fut
                cache[str(i)] = {"label": lab, "confidence": conf}
                done += 1
                if done % 100 == 0:
                    rate = done / (time.time() - t0 + 1e-9)
                    eta = (len(todo) - done) / max(rate, 1e-9)
                    print(
                        f"      {done}/{len(todo)} done "
                        f"({rate:.1f} req/s, ETA {eta/60:.1f} min)"
                    )
                if cache_path and done % 200 == 0:
                    cache_path.write_text(json.dumps(cache))
        if cache_path:
            cache_path.write_text(json.dumps(cache))
    labels = [cache[str(i)]["label"] for i in range(n)]
    confs = np.array([cache[str(i)]["confidence"] for i in range(n)], dtype=np.float64)
    return labels, confs


def main() -> None:
    t_total = time.time()
    print("=" * 70)
    print("  Building Financial PhraseBank agent-prob cache (Option C)")
    print("=" * 70)

    cfg = DataConfig(dataset_name="warwickai/financial_phrasebank_mirror")
    train, val, test = load_financial_phrasebank(cfg)
    train_x = preprocess_batch(train.texts)
    val_x = preprocess_batch(val.texts)
    test_x = preprocess_batch(test.texts)
    train_y, val_y, test_y = train.labels, val.labels, test.labels
    print(f"  splits: train={len(train_x)}  val={len(val_x)}  test={len(test_x)}")

    print("\n  [1/4] TF-IDF + LogReg (fit on PhraseBank train)")
    tfidf = TfidfLogRegBaseline()
    tfidf.train(train_x, train_y)
    tfidf_val = _tfidf_aligned(tfidf, val_x)
    tfidf_test = _tfidf_aligned(tfidf, test_x)
    tfidf_pred = tfidf.predict(test_x)
    tfidf_acc = float(np.mean(np.array(tfidf_pred) == np.array(test_y)))
    print(f"    test accuracy = {tfidf_acc:.4f}")

    print("\n  [2/4] Loughran-McDonald lexicon")
    lex = LoughranMcDonaldAgent()
    lex_val = lex.predict_proba(val_x, label_order=LABELS)
    lex_test = lex.predict_proba(test_x, label_order=LABELS)
    lex_test_pred = [LABELS[int(i)] for i in np.argmax(lex_test, axis=1)]
    lex_acc = float(np.mean(np.array(lex_test_pred) == np.array(test_y)))
    print(f"    test accuracy = {lex_acc:.4f}")

    print("\n  [3/4] GPT-4o-mini zero-shot on PhraseBank val+test")
    llm = SingleLLMAgent(mode="zero_shot")
    llm_val_labels, llm_val_conf = _run_llm(
        llm, val_x, n_workers=8, cache_path=PB_DIR / "_llm_val_cache.json"
    )
    llm_val_proba = _llm_to_pseudo_proba(llm_val_labels, llm_val_conf)
    llm_val_cost = float(llm.total_cost_usd)
    llm_val_tokens = int(llm.total_tokens)
    print(f"    val cost ${llm_val_cost:.4f} ({llm_val_tokens} tokens)")

    llm2 = SingleLLMAgent(mode="zero_shot")
    llm_test_labels, llm_test_conf = _run_llm(
        llm2, test_x, n_workers=8, cache_path=PB_DIR / "_llm_test_cache.json"
    )
    llm_test_proba = _llm_to_pseudo_proba(llm_test_labels, llm_test_conf)
    llm_test_acc = float(np.mean(np.array(llm_test_labels) == np.array(test_y)))
    print(f"    test accuracy = {llm_test_acc:.4f}  " f"(extra cost ${llm2.total_cost_usd:.4f})")

    print("\n  [4/4] Fine-tuning FinBERT on PhraseBank")
    tcfg = TransformerConfig()
    fb = TransformerBaseline(tcfg)
    if torch.backends.mps.is_available():
        fb.device = "mps"
    elif torch.cuda.is_available():
        fb.device = "cuda"
    else:
        fb.device = "cpu"
    print(f"    device = {fb.device}")
    train_enc = fb._encode(train_x)
    val_enc = fb._encode(val_x)
    train_lab = [fb._label2id[l] for l in train_y]
    val_lab = [fb._label2id[l] for l in val_y]
    train_ds = _SentimentTorchDataset(train_enc, train_lab)
    val_ds = _SentimentTorchDataset(val_enc, val_lab)

    targs = TrainingArguments(
        output_dir=str(RESULTS_DIR / "_finbert_pb_trainer_scratch"),
        num_train_epochs=tcfg.num_epochs,
        per_device_train_batch_size=tcfg.batch_size,
        per_device_eval_batch_size=tcfg.batch_size,
        learning_rate=tcfg.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=42,
        report_to=[],
    )
    trainer = Trainer(model=fb.model, args=targs, train_dataset=train_ds, eval_dataset=val_ds)
    ft_t0 = time.time()
    trainer.train()
    ft_wall = time.time() - ft_t0
    fb.model = trainer.model
    fb.save(str(FT_DIR))
    print(f"    saved fine-tuned weights -> {FT_DIR}")

    print("    inference val + test ...")
    fb_val = fb.predict_proba(val_x, label_order=LABELS, show_progress=True)
    fb_test = fb.predict_proba(test_x, label_order=LABELS, show_progress=True)
    fb_test_pred = [LABELS[int(i)] for i in np.argmax(fb_test, axis=1)]
    fb_test_acc = float(np.mean(np.array(fb_test_pred) == np.array(test_y)))
    print(f"    test accuracy = {fb_test_acc:.4f}  fine-tune wall {ft_wall/60:.2f} min")

    np.savez_compressed(
        PB_DIR / "_cache_agents.npz",
        tfidf_val_proba=tfidf_val.astype(np.float32),
        tfidf_test_proba=tfidf_test.astype(np.float32),
        tfidf_test_pred_labels=np.array(tfidf_pred),
        lex_val_proba=lex_val.astype(np.float32),
        lex_test_proba=lex_test.astype(np.float32),
        lex_test_pred_labels=np.array(lex_test_pred),
        llm_val_labels=np.array(llm_val_labels),
        llm_val_confidence=llm_val_conf.astype(np.float32),
        llm_val_proba=llm_val_proba.astype(np.float32),
        llm_test_labels=np.array(llm_test_labels),
        llm_test_proba=llm_test_proba.astype(np.float32),
        llm_val_cost_usd=np.array([llm_val_cost]),
        llm_val_tokens=np.array([llm_val_tokens]),
        val_labels=np.array(val_y),
        test_labels=np.array(test_y),
    )
    np.savez_compressed(
        PB_DIR / "_cache_finbert.npz",
        val_proba=fb_val.astype(np.float32),
        test_proba=fb_test.astype(np.float32),
        val_labels=np.array(val_y),
        test_labels=np.array(test_y),
        test_pred_labels=np.array(fb_test_pred),
        test_accuracy=np.array([fb_test_acc]),
        ft_wall_seconds=np.array([ft_wall]),
    )
    print(f"\n  Wrote {PB_DIR/'_cache_agents.npz'}")
    print(f"  Wrote {PB_DIR/'_cache_finbert.npz'}")
    print(f"  Total wall-clock: {(time.time()-t_total)/60:.2f} min")


if __name__ == "__main__":
    main()
