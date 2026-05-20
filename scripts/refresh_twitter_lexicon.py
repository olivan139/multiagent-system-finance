#!/usr/bin/env python3
"""Re-run the Twitter Financial-News Sentiment ensemble with 4 agents.

The lexicon agent was added in Wave 1 and is now auto-instantiated by
``HeterogeneousEnsemble``. This refresh script rebuilds:

    * all_results.json
    * predictions.json
    * ensemble_meta.json   (12-feature, 4-agent feature importance)
    * statistical_report.json

Cost-saving strategy ("stitched cached"):

    * TF-IDF + LogReg              -> re-fit from scratch (~seconds)
    * FinBERT                      -> *zero-shot* inference on val+test
                                       (no fine-tune, ~minutes on MPS).
                                       The cached fine-tuned baseline row
                                       is preserved as a standalone result
                                       but does NOT feed the ensemble.
    * Single LLM (zero_shot)       -> cached test predictions reused;
                                       run live on val only (~$0.05).
    * Loughran-McDonald lexicon    -> live on val+test (free, fast).

We then train a fresh meta-learner on the val features (3 classes x
4 agents = 12 features) and apply it to the test features. Majority,
weighted-average, stacking, and 4 ablations are produced.

Multi-Agent Pipeline + FinBERT (fine-tuned) + FinBERT (zero-shot) +
Single LLM (zero_shot) rows are kept from the cached all_results.json
so the saved artefact remains a full comparison table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mas.config import LABEL2ID, LABELS, RESULTS_DIR, DataConfig
from mas.data import load_financial_phrasebank
from mas.data.preprocessing import preprocess_batch
from mas.evaluation import (
    ExperimentResult,
    bootstrap_ci,
    comparison_table,
    compute_metrics,
    pairwise_mcnemar,
    print_bootstrap_cis,
    print_mcnemar_results,
    print_metrics,
    save_results,
    save_statistical_report,
)
from mas.baselines.tfidf_logreg import TfidfLogRegBaseline
from mas.baselines.transformer import TransformerBaseline
from mas.baselines.lexicon import LoughranMcDonaldAgent
from mas.agents.single import SingleLLMAgent
from mas.agents.ensemble import (
    AGENT_ORDER,
    EnsembleAgentPredictions,
    HeterogeneousEnsemble,
    _llm_to_pseudo_proba,
)


def labels_to_pseudo_proba(labels: list[str], confidence: float = 0.85) -> np.ndarray:
    """Cached-label -> soft probability matrix (chosen class = ``confidence``)."""
    n = len(labels)
    proba = np.zeros((n, len(LABELS)), dtype=np.float64)
    rest = (1.0 - confidence) / (len(LABELS) - 1)
    for i, lab in enumerate(labels):
        chosen = LABEL2ID.get(lab, LABEL2ID["neutral"])
        proba[i, :] = rest
        proba[i, chosen] = confidence
    return proba


def tfidf_predict_proba_aligned(tfidf: TfidfLogRegBaseline, texts: list[str]) -> np.ndarray:
    """TF-IDF probas re-indexed to canonical LABELS order."""
    raw = tfidf.predict_proba(texts)
    classes = list(tfidf.pipeline.classes_)
    n = len(texts)
    out = np.zeros((n, len(LABELS)), dtype=np.float64)
    for j, lab in enumerate(LABELS):
        if lab in classes:
            out[:, j] = raw[:, classes.index(lab)]
    return out


def record(name, y_true, y_pred, latency_ms, cost_usd=0.0, metadata=None):
    r = compute_metrics(
        y_true,
        y_pred,
        name,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        metadata=metadata or {},
    )
    print_metrics(r)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="zeroshot/twitter-financial-news-sentiment")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    wall_t0 = time.time()
    out_dir = RESULTS_DIR / "twitter"
    out_dir.mkdir(parents=True, exist_ok=True)

    cached_predictions_path = out_dir / "predictions.json"
    cached_results_path = out_dir / "all_results.json"
    cached_predictions = json.loads(cached_predictions_path.read_text())
    cached_results = json.loads(cached_results_path.read_text())
    cached_results_by_name = {r["model_name"]: r for r in cached_results}

    print("=" * 70)
    print("  Twitter refresh: 4-agent ensemble (lexicon added)")
    print("=" * 70)

    data_cfg = DataConfig(dataset_name=args.dataset)
    train, val, test = load_financial_phrasebank(data_cfg)
    train_texts = preprocess_batch(train.texts)
    val_texts = preprocess_batch(val.texts)
    test_texts = preprocess_batch(test.texts)
    train_labels = train.labels
    val_labels = val.labels
    test_labels = test.labels
    if args.max_samples:
        test_texts = test_texts[: args.max_samples]
        test_labels = test_labels[: args.max_samples]
        val_texts = val_texts[: args.max_samples]
        val_labels = val_labels[: args.max_samples]
    n_test = len(test_texts)
    print(f"  train={len(train_texts)}  val={len(val_texts)}  test={n_test}")
    assert (
        cached_predictions["y_true"][:n_test] == test_labels
    ), "Cached y_true does not match fresh split; cannot reuse cached LLM labels."

    new_results: list[ExperimentResult] = []
    new_predictions: dict[str, list[str]] = {}

    print("\n[1/6] TF-IDF + LogReg (live re-fit)")
    tfidf = TfidfLogRegBaseline()
    tfidf.train(train_texts, train_labels)
    t0 = time.time()
    tfidf_test_preds = tfidf.predict(test_texts)
    tfidf_latency = (time.time() - t0) * 1000
    new_predictions["TF-IDF + LogReg"] = tfidf_test_preds
    new_results.append(
        record(
            "TF-IDF + LogReg",
            test_labels,
            tfidf_test_preds,
            tfidf_latency,
        )
    )
    tfidf_val_proba = tfidf_predict_proba_aligned(tfidf, val_texts)
    tfidf_test_proba = tfidf_predict_proba_aligned(tfidf, test_texts)

    print("\n[2/6] FinBERT zero-shot (live inference on val + test)")
    fb = TransformerBaseline()
    t0 = time.time()
    finbert_zs_test_preds = fb.predict_pretrained(test_texts)
    finbert_zs_test_latency = (time.time() - t0) * 1000
    new_predictions["FinBERT (zero-shot)"] = finbert_zs_test_preds
    new_results.append(
        record(
            "FinBERT (zero-shot)",
            test_labels,
            finbert_zs_test_preds,
            finbert_zs_test_latency,
        )
    )
    finbert_val_proba = fb.predict_proba(val_texts, label_order=LABELS)
    finbert_test_proba = fb.predict_proba(test_texts, label_order=LABELS)

    if "FinBERT (fine-tuned)" in cached_results_by_name:
        new_results.append(ExperimentResult(**cached_results_by_name["FinBERT (fine-tuned)"]))
        new_predictions["FinBERT (fine-tuned)"] = cached_predictions["predictions"][
            "FinBERT (fine-tuned)"
        ][:n_test]
        print("  [keep cached] FinBERT (fine-tuned) row preserved from previous run")

    print("\n[3/6] Single LLM zero-shot — running on VAL only")
    llm_agent = SingleLLMAgent(mode="zero_shot")
    t0 = time.time()
    llm_val_results = llm_agent.analyze_batch(val_texts, show_progress=True)
    llm_val_latency_ms = (time.time() - t0) * 1000
    llm_val_labels = [r.sentiment for r in llm_val_results]
    llm_val_conf = np.array([r.confidence for r in llm_val_results], dtype=np.float64)
    llm_val_proba = _llm_to_pseudo_proba(llm_val_labels, llm_val_conf)
    llm_val_cost = float(llm_agent.total_cost_usd)
    llm_val_tokens = int(llm_agent.total_tokens)
    print(f"  Val LLM cost: ${llm_val_cost:.4f}  tokens: {llm_val_tokens}")

    cached_llm_test_labels = cached_predictions["predictions"]["Single LLM (zero_shot)"][:n_test]
    new_predictions["Single LLM (zero_shot)"] = cached_llm_test_labels
    if "Single LLM (zero_shot)" in cached_results_by_name:
        new_results.append(ExperimentResult(**cached_results_by_name["Single LLM (zero_shot)"]))
        print("  [keep cached] Single LLM (zero_shot) test metrics preserved")

    llm_test_proba = labels_to_pseudo_proba(cached_llm_test_labels, confidence=0.85)

    llm_test_conf = np.full(n_test, 0.85)

    if "Multi-Agent Pipeline" in cached_results_by_name:
        new_results.append(ExperimentResult(**cached_results_by_name["Multi-Agent Pipeline"]))
        new_predictions["Multi-Agent Pipeline"] = cached_predictions["predictions"][
            "Multi-Agent Pipeline"
        ][:n_test]
        print("  [keep cached] Multi-Agent Pipeline row preserved from previous run")

    print("\n[4/6] Loughran-McDonald lexicon (live)")
    lex = LoughranMcDonaldAgent()
    lex_val_proba = lex.predict_proba(val_texts, label_order=LABELS)
    t0 = time.time()
    lex_test_proba = lex.predict_proba(test_texts, label_order=LABELS)
    lex_test_latency = (time.time() - t0) * 1000
    lex_test_preds = [LABELS[int(i)] for i in np.argmax(lex_test_proba, axis=1)]
    new_predictions["Loughran-McDonald Lexicon"] = lex_test_preds
    new_results.append(
        record(
            "Loughran-McDonald Lexicon",
            test_labels,
            lex_test_preds,
            lex_test_latency,
            metadata={"source": getattr(lex, "source", "unknown")},
        )
    )

    val_agent_preds = EnsembleAgentPredictions(
        tfidf_proba=tfidf_val_proba,
        finbert_proba=finbert_val_proba,
        llm_labels=llm_val_labels,
        llm_confidence=llm_val_conf,
        llm_proba=llm_val_proba,
        lexicon_proba=lex_val_proba,
    )
    test_agent_preds = EnsembleAgentPredictions(
        tfidf_proba=tfidf_test_proba,
        finbert_proba=finbert_test_proba,
        llm_labels=cached_llm_test_labels,
        llm_confidence=llm_test_conf,
        llm_proba=llm_test_proba,
        lexicon_proba=lex_test_proba,
    )

    print("\n[5/6] 4-agent ensemble strategies")
    ensemble_meta_info: dict = {}
    for strategy in ("majority", "weighted_average", "stacking"):
        print(f"  -- strategy: {strategy}")

        ens = HeterogeneousEnsemble(
            tfidf_baseline=tfidf,
            finbert_baseline=fb,
            llm_agent=llm_agent,
            strategy=strategy,
        )
        if strategy == "stacking":
            info = ens.fit_meta_learner(val_texts, val_labels, precomputed=val_agent_preds)
            info["feature_importance"] = ens.feature_importance()
            ensemble_meta_info = info
        t0 = time.time()
        preds = ens.predict_labels(test_texts, show_progress=False, precomputed=test_agent_preds)
        lat = (time.time() - t0) * 1000
        new_predictions[f"Ensemble ({strategy})"] = preds
        new_results.append(
            record(
                f"Ensemble ({strategy})",
                test_labels,
                preds,
                lat,
                metadata={"strategy": strategy, "n_agents": 4, "agents": list(AGENT_ORDER)},
            )
        )

    print("\n[6/6] Ablation study (stacking, drop one agent at a time)")
    for dropped in AGENT_ORDER:
        active = tuple(a for a in AGENT_ORDER if a != dropped)
        label = f"Ablation: drop {dropped} (stacking)"
        print(f"  -- {label}")
        ens = HeterogeneousEnsemble(
            tfidf_baseline=tfidf,
            finbert_baseline=fb,
            llm_agent=llm_agent,
            strategy="stacking",
            active_agents=active,
        )
        ens.fit_meta_learner(val_texts, val_labels, precomputed=val_agent_preds)
        t0 = time.time()
        preds = ens.predict_labels(test_texts, show_progress=False, precomputed=test_agent_preds)
        lat = (time.time() - t0) * 1000
        new_predictions[label] = preds
        new_results.append(
            record(
                label,
                test_labels,
                preds,
                lat,
                metadata={
                    "strategy": "stacking",
                    "dropped": dropped,
                    "active_agents": list(active),
                },
            )
        )

    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print(comparison_table(new_results))

    save_results(new_results, out_dir / "all_results.json")
    print(f"  -> {out_dir/'all_results.json'}")

    with open(out_dir / "predictions.json", "w") as f:
        json.dump({"y_true": test_labels, "predictions": new_predictions}, f, indent=2)
    print(f"  -> {out_dir/'predictions.json'}")

    if ensemble_meta_info:
        with open(out_dir / "ensemble_meta.json", "w") as f:
            json.dump(ensemble_meta_info, f, indent=2)
        print(
            f"  -> {out_dir/'ensemble_meta.json'}  "
            f"({ensemble_meta_info['n_features']} features)"
        )

    print("\n" + "=" * 70)
    print("  Statistical significance tests")
    print("=" * 70)
    mcnemar_results = pairwise_mcnemar(test_labels, new_predictions)
    print_mcnemar_results(mcnemar_results)
    cis = []
    for name, preds in new_predictions.items():
        for metric in ("accuracy", "weighted_f1"):
            cis.append(bootstrap_ci(test_labels, preds, metric=metric, model_name=name))
    print_bootstrap_cis(cis)
    save_statistical_report(
        mcnemar_results,
        cis,
        friedman_result=None,
        path=out_dir / "statistical_report.json",
    )
    print(f"  -> {out_dir/'statistical_report.json'}")

    elapsed = time.time() - wall_t0
    print(f"\nWall-clock: {elapsed/60:.2f} min  |  LLM cost: ${llm_val_cost:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
