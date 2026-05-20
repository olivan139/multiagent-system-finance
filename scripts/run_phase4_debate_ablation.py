#!/usr/bin/env python3
"""Phase 4: Multi-agent debate with confidence routing + ablation studies."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas.agents.debate import DebateMultiAgentSystem
from mas.config import RESULTS_DIR, DataConfig
from mas.data import load_financial_phrasebank
from mas.data.preprocessing import preprocess_batch
from mas.evaluation import (
    ExperimentResult,
    comparison_table,
    compute_metrics,
    load_results,
    print_metrics,
    save_results,
)


def run_ablation(
    test_texts: list[str],
    test_labels: list[str],
    thresholds: list[float],
) -> list[ExperimentResult]:
    """Ablation: vary the confidence threshold for debate routing."""
    results: list[ExperimentResult] = []
    for threshold in thresholds:
        print(f"\n--- Debate (threshold={threshold}) ---")
        agent = DebateMultiAgentSystem(confidence_threshold=threshold)
        t0 = time.time()
        preds = agent.predict_labels(test_texts)
        latency = (time.time() - t0) * 1000

        result = compute_metrics(
            test_labels,
            preds,
            model_name=f"Debate (t={threshold})",
            latency_ms=latency,
            cost_usd=agent.total_cost_usd,
            metadata={
                "threshold": threshold,
                "total_tokens": agent.total_tokens,
                "routing": agent.routing_stats,
            },
        )
        print_metrics(result)
        stats = agent.routing_stats
        print(
            f"  Routing: {stats['fast_pct']:.1f}% fast path, "
            f"{stats['debate']} debates out of {stats['total']} samples"
        )
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 4: Multi-Agent Debate + Ablation")
    print("=" * 60)

    config = DataConfig()
    _, _, test = load_financial_phrasebank(config)
    test_texts = preprocess_batch(test.texts)
    test_labels = test.labels

    if args.max_samples:
        test_texts = test_texts[: args.max_samples]
        test_labels = test_labels[: args.max_samples]

    agent = DebateMultiAgentSystem(confidence_threshold=0.85)
    t0 = time.time()
    preds = agent.predict_labels(test_texts)
    latency = (time.time() - t0) * 1000

    main_result = compute_metrics(
        test_labels,
        preds,
        model_name="Multi-Agent Debate",
        latency_ms=latency,
        cost_usd=agent.total_cost_usd,
        metadata={
            "total_tokens": agent.total_tokens,
            "routing": agent.routing_stats,
        },
    )
    print_metrics(main_result)
    stats = agent.routing_stats
    print(
        f"  Routing: {stats['fast_pct']:.1f}% fast path, "
        f"{stats['debate']} debates out of {stats['total']} samples"
    )

    results = [main_result]

    if not args.skip_ablation:
        print("\n\n--- ABLATION: Confidence Thresholds ---")
        ablation = run_ablation(test_texts, test_labels, thresholds=[0.5, 0.7, 0.85, 0.95, 1.0])
        results.extend(ablation)

    prev_path = RESULTS_DIR / "phase3_results.json"
    prev = load_results(prev_path) if prev_path.exists() else []
    all_results = prev + results
    save_results(all_results, RESULTS_DIR / "phase4_results.json")

    print("\n\n" + "=" * 80)
    print("  FULL COMPARISON")
    print("=" * 80)
    print(comparison_table(all_results))

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
