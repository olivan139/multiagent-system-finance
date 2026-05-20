#!/usr/bin/env python3
"""Phase 2: Transformer baselines + single LLM agents."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas.config import RESULTS_DIR, DataConfig
from mas.data import load_financial_phrasebank
from mas.data.preprocessing import preprocess_batch
from mas.evaluation import (
    ExperimentResult,
    compute_metrics,
    load_results,
    print_metrics,
    save_results,
)


def run_finbert_zero_shot(test_texts: list[str], test_labels: list[str]) -> ExperimentResult:
    from mas.baselines.transformer import TransformerBaseline

    print("\n--- FinBERT Zero-Shot ---")
    model = TransformerBaseline()
    t0 = time.time()
    preds = model.predict_pretrained(test_texts)
    latency = (time.time() - t0) * 1000
    return compute_metrics(test_labels, preds, model_name="FinBERT (zero-shot)", latency_ms=latency)


def run_finbert_finetuned(
    train_texts: list[str],
    train_labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
) -> ExperimentResult:
    from mas.baselines.transformer import TransformerBaseline

    print("\n--- FinBERT Fine-tuned ---")
    model = TransformerBaseline()
    model.fine_tune(train_texts, train_labels, val_texts, val_labels)
    t0 = time.time()
    preds = model.predict(test_texts)
    latency = (time.time() - t0) * 1000
    return compute_metrics(
        test_labels, preds, model_name="FinBERT (fine-tuned)", latency_ms=latency
    )


def run_single_llm(
    test_texts: list[str], test_labels: list[str], mode: str = "zero_shot"
) -> ExperimentResult:
    from mas.agents.single import SingleLLMAgent

    print(f"\n--- Single LLM Agent ({mode}) ---")
    agent = SingleLLMAgent(mode=mode)
    t0 = time.time()
    preds = agent.predict_labels(test_texts)
    latency = (time.time() - t0) * 1000
    return compute_metrics(
        test_labels,
        preds,
        model_name=f"Single LLM ({mode})",
        latency_ms=latency,
        cost_usd=agent.total_cost_usd,
        metadata={"total_tokens": agent.total_tokens},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-finetune", action="store_true", help="Skip FinBERT fine-tuning")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM agent experiments")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test samples")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 2: Transformer + LLM Baselines")
    print("=" * 60)

    config = DataConfig()
    train, val, test = load_financial_phrasebank(config)

    train_texts = preprocess_batch(train.texts)
    val_texts = preprocess_batch(val.texts)
    test_texts = preprocess_batch(test.texts)
    test_labels = test.labels

    if args.max_samples:
        test_texts = test_texts[: args.max_samples]
        test_labels = test_labels[: args.max_samples]
        print(f"  (limited to {args.max_samples} test samples)")

    results: list[ExperimentResult] = []

    zs_result = run_finbert_zero_shot(test_texts, test_labels)
    print_metrics(zs_result)
    results.append(zs_result)

    if not args.skip_finetune:
        ft_result = run_finbert_finetuned(
            train_texts, train.labels, val_texts, val.labels, test_texts, test_labels
        )
        print_metrics(ft_result)
        results.append(ft_result)

    if not args.skip_llm:
        for mode in ["zero_shot", "few_shot"]:
            llm_result = run_single_llm(test_texts, test_labels, mode=mode)
            print_metrics(llm_result)
            results.append(llm_result)

    phase1_path = RESULTS_DIR / "phase1_results.json"
    if phase1_path.exists():
        results = load_results(phase1_path) + results

    save_results(results, RESULTS_DIR / "phase2_results.json")
    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
