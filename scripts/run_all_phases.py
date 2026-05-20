#!/usr/bin/env python3
"""Run the full experiment suite.

Phases:
    1. Classical baseline (TF-IDF + LogReg)
    2. Transformer baselines (FinBERT zero-shot + fine-tuned)
    3. Single LLM agent
    4. LLM-only multi-agent systems (pipeline + optional debate)
    5. Heterogeneous ensemble (majority, weighted average, stacking)
    6. Ablation: stacking ensemble with each agent dropped
    7. Statistical significance tests (McNemar, bootstrap CIs)

Per-model predictions are saved to ``results/<dataset>/predictions.json`` so
that downstream analyses (cross-dataset Friedman test, error analysis,
SHAP) can reuse them without re-running expensive models.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas.config import RESULTS_DIR, DataConfig
from mas.data import load_financial_phrasebank
from mas.data.preprocessing import preprocess_batch
from mas.evaluation import (
    ExperimentResult,
    bootstrap_ci,
    comparison_table,
    compute_metrics,
    pairwise_mcnemar,
    plot_comparison_bar,
    plot_confusion_matrices,
    print_bootstrap_cis,
    print_metrics,
    print_mcnemar_results,
    save_results,
    save_statistical_report,
)

DATASET_SHORT = {
    "zeroshot/twitter-financial-news-sentiment": "twitter",
    "warwickai/financial_phrasebank_mirror": "phrasebank",
    "financial_phrasebank": "phrasebank",
}


def _record(
    name: str,
    y_true: list[str],
    y_pred: list[str],
    latency_ms: float,
    cost_usd: float = 0.0,
    metadata: dict | None = None,
    all_results: list[ExperimentResult] | None = None,
    predictions: dict[str, list[str]] | None = None,
) -> ExperimentResult:
    r = compute_metrics(
        y_true, y_pred, name, latency_ms=latency_ms, cost_usd=cost_usd, metadata=metadata or {}
    )
    print_metrics(r)
    if all_results is not None:
        all_results.append(r)
    if predictions is not None:
        predictions[name] = list(y_pred)
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all thesis experiments")
    parser.add_argument(
        "--dataset",
        default="zeroshot/twitter-financial-news-sentiment",
        help="HuggingFace dataset id (twitter or phrasebank)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit test set size for quick validation"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Limit training set size (speeds up FinBERT fine-tune)",
    )
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-multi-agent", action="store_true")
    parser.add_argument(
        "--include-debate",
        action="store_true",
        help="Include legacy multi-agent debate (deprecated; replaced by ensemble)",
    )
    parser.add_argument(
        "--include-few-shot",
        action="store_true",
        help="Include the few-shot LLM variant (~2x LLM cost)",
    )
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    args = parser.parse_args()

    dataset_short = DATASET_SHORT.get(args.dataset, args.dataset.replace("/", "_"))
    out_dir = RESULTS_DIR / dataset_short
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Multi-Agent System for Social Data Analysis")
    print(f"  Dataset: {args.dataset}  →  results in {out_dir}")
    print("=" * 70)

    data_cfg = DataConfig(dataset_name=args.dataset)
    train, val, test = load_financial_phrasebank(data_cfg)
    train_texts = preprocess_batch(train.texts)
    val_texts = preprocess_batch(val.texts)
    test_texts = preprocess_batch(test.texts)
    test_labels = test.labels
    train_labels = train.labels
    val_labels = val.labels

    if args.max_train and len(train_texts) > args.max_train:
        train_texts = train_texts[: args.max_train]
        train_labels = train_labels[: args.max_train]
        print(f"  Capped training set at {len(train_texts)} samples")

    if args.max_samples:
        test_texts = test_texts[: args.max_samples]
        test_labels = test_labels[: args.max_samples]
        val_texts = val_texts[: args.max_samples]
        val_labels = val_labels[: args.max_samples]

    print(f"  train={len(train_texts)}  val={len(val_texts)}  test={len(test_texts)}")

    all_results: list[ExperimentResult] = []
    predictions: dict[str, list[str]] = {}

    print("\n\n" + "=" * 70)
    print("  PHASE 1: Classical Baseline")
    print("=" * 70)
    from mas.baselines.tfidf_logreg import TfidfLogRegBaseline

    tfidf = TfidfLogRegBaseline()
    tfidf.train(train_texts, train_labels)
    t0 = time.time()
    preds = tfidf.predict(test_texts)
    lat = (time.time() - t0) * 1000
    _record(
        "TF-IDF + LogReg", test_labels, preds, lat, all_results=all_results, predictions=predictions
    )

    print("\n\n" + "=" * 70)
    print("  PHASE 2: Transformer Baselines")
    print("=" * 70)
    fb = None
    fb2 = None
    try:
        from mas.baselines.transformer import TransformerBaseline

        fb = TransformerBaseline()
        t0 = time.time()
        preds = fb.predict_pretrained(test_texts)
        lat = (time.time() - t0) * 1000
        _record(
            "FinBERT (zero-shot)",
            test_labels,
            preds,
            lat,
            all_results=all_results,
            predictions=predictions,
        )

        if not args.skip_finetune:
            fb2 = TransformerBaseline()
            fb2.fine_tune(train_texts, train_labels, val_texts, val_labels)
            t0 = time.time()
            preds = fb2.predict(test_texts)
            lat = (time.time() - t0) * 1000
            _record(
                "FinBERT (fine-tuned)",
                test_labels,
                preds,
                lat,
                all_results=all_results,
                predictions=predictions,
            )
    except ImportError:
        print("  [SKIP] transformers/torch not installed")

    llm_agent = None
    if not args.skip_llm:
        print("\n\n" + "=" * 70)
        print("  PHASE 3: Single LLM Baseline")
        print("=" * 70)
        try:
            from mas.agents.single import SingleLLMAgent

            modes = ["zero_shot", "few_shot"] if args.include_few_shot else ["zero_shot"]
            for mode in modes:
                agent = SingleLLMAgent(mode=mode)
                t0 = time.time()
                preds = agent.predict_labels(test_texts)
                lat = (time.time() - t0) * 1000
                _record(
                    f"Single LLM ({mode})",
                    test_labels,
                    preds,
                    lat,
                    cost_usd=agent.total_cost_usd,
                    metadata={"tokens": agent.total_tokens},
                    all_results=all_results,
                    predictions=predictions,
                )
                if mode == "zero_shot":
                    llm_agent = agent
        except ImportError:
            print("  [SKIP] openai not installed")

    if not args.skip_multi_agent:
        print("\n\n" + "=" * 70)
        print("  PHASE 4: Multi-Agent Pipeline (LLM-only)")
        print("=" * 70)
        try:
            from mas.agents.multi import MultiAgentSystem

            agent = MultiAgentSystem()
            t0 = time.time()
            preds = agent.predict_labels(test_texts)
            lat = (time.time() - t0) * 1000
            _record(
                "Multi-Agent Pipeline",
                test_labels,
                preds,
                lat,
                cost_usd=agent.total_cost_usd,
                metadata={"tokens": agent.total_tokens},
                all_results=all_results,
                predictions=predictions,
            )
        except ImportError:
            print("  [SKIP] langgraph/langchain not installed")

    if args.include_debate:
        try:
            from mas.agents.debate import DebateMultiAgentSystem

            agent = DebateMultiAgentSystem(confidence_threshold=0.85)
            t0 = time.time()
            preds = agent.predict_labels(test_texts)
            lat = (time.time() - t0) * 1000
            _record(
                "Multi-Agent Debate",
                test_labels,
                preds,
                lat,
                cost_usd=agent.total_cost_usd,
                metadata={"tokens": agent.total_tokens, "routing": agent.routing_stats},
                all_results=all_results,
                predictions=predictions,
            )
        except ImportError:
            print("  [SKIP] langgraph/langchain not installed")

    ensemble_meta_info: dict = {}
    if not args.skip_ensemble:
        print("\n\n" + "=" * 70)
        print("  PHASE 5: Heterogeneous Multi-Agent Ensemble")
        print("=" * 70)

        if fb2 is None and fb is not None:
            print("  Fine-tuned FinBERT not available (use --no skip-finetune to enable);")
            print("  ensemble will use the zero-shot FinBERT instead.")
        finbert_for_ensemble = fb2 or fb

        if llm_agent is None:
            print("  [SKIP] Ensemble needs the single LLM agent (got --skip-llm).")
        elif finbert_for_ensemble is None:
            print("  [SKIP] Ensemble needs FinBERT (transformers not installed).")
        else:
            from mas.agents.ensemble import HeterogeneousEnsemble

            base_ens = HeterogeneousEnsemble(
                tfidf_baseline=tfidf,
                finbert_baseline=finbert_for_ensemble,
                llm_agent=llm_agent,
                strategy="stacking",
            )
            print("  Pre-collecting per-agent predictions on val + test (one-shot)")
            val_agent_preds = base_ens.collect_agent_predictions(val_texts, show_progress=True)
            test_agent_preds = base_ens.collect_agent_predictions(test_texts, show_progress=True)

            for strategy in ("majority", "weighted_average", "stacking"):
                print(f"\n  -- Ensemble strategy: {strategy} --")
                ens = HeterogeneousEnsemble(
                    tfidf_baseline=tfidf,
                    finbert_baseline=finbert_for_ensemble,
                    llm_agent=llm_agent,
                    strategy=strategy,
                )
                if strategy == "stacking":
                    info = ens.fit_meta_learner(val_texts, val_labels, precomputed=val_agent_preds)
                    importance = ens.feature_importance()
                    info["feature_importance"] = importance
                    ensemble_meta_info = info

                t0 = time.time()
                preds = ens.predict_labels(
                    test_texts, show_progress=False, precomputed=test_agent_preds
                )
                lat = (time.time() - t0) * 1000
                _record(
                    f"Ensemble ({strategy})",
                    test_labels,
                    preds,
                    lat,
                    cost_usd=0.0,
                    metadata={"strategy": strategy},
                    all_results=all_results,
                    predictions=predictions,
                )

    ablation_results: list[ExperimentResult] = []
    if (
        not args.skip_ablation
        and not args.skip_ensemble
        and llm_agent is not None
        and (fb2 is not None or fb is not None)
    ):
        print("\n\n" + "=" * 70)
        print("  PHASE 6: Ablation Study (stacking ensemble)")
        print("=" * 70)
        from mas.agents.ensemble import HeterogeneousEnsemble

        finbert_for_ensemble = fb2 or fb
        ablations = [
            ("tfidf", "finbert"),
            ("tfidf", "llm"),
            ("finbert", "llm"),
        ]
        for active in ablations:
            dropped = {"tfidf", "finbert", "llm"} - set(active)
            label = f"Ablation: drop {next(iter(dropped))} (stacking)"
            print(f"\n  -- {label} --")
            ens = HeterogeneousEnsemble(
                tfidf_baseline=tfidf,
                finbert_baseline=finbert_for_ensemble,
                llm_agent=llm_agent,
                strategy="stacking",
                active_agents=active,
            )
            ens.fit_meta_learner(val_texts, val_labels, precomputed=val_agent_preds)
            t0 = time.time()
            preds = ens.predict_labels(
                test_texts, show_progress=False, precomputed=test_agent_preds
            )
            lat = (time.time() - t0) * 1000
            r = _record(
                label, test_labels, preds, lat, all_results=all_results, predictions=predictions
            )
            ablation_results.append(r)

    print("\n\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print(comparison_table(all_results))

    save_results(all_results, out_dir / "all_results.json")
    plot_comparison_bar(all_results, out_dir / "comparison_bar.png")
    plot_confusion_matrices(all_results, out_dir / "confusion_matrices.png")

    with open(out_dir / "predictions.json", "w") as f:
        json.dump(
            {
                "y_true": test_labels,
                "predictions": predictions,
            },
            f,
            indent=2,
        )
    print(f"Predictions saved to {out_dir / 'predictions.json'}")

    if ensemble_meta_info:
        with open(out_dir / "ensemble_meta.json", "w") as f:
            json.dump(ensemble_meta_info, f, indent=2)
        print(f"Ensemble meta-learner info saved to {out_dir / 'ensemble_meta.json'}")

    if not args.skip_stats and len(predictions) >= 2:
        print("\n\n" + "=" * 70)
        print("  PHASE 7: Statistical Significance Tests")
        print("=" * 70)
        mcnemar_results = pairwise_mcnemar(test_labels, predictions)
        print_mcnemar_results(mcnemar_results)

        cis = []
        for name, preds in predictions.items():
            for metric in ("accuracy", "weighted_f1"):
                cis.append(bootstrap_ci(test_labels, preds, metric=metric, model_name=name))
        print_bootstrap_cis(cis)

        save_statistical_report(
            mcnemar_results,
            cis,
            friedman_result=None,
            path=out_dir / "statistical_report.json",
        )

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
