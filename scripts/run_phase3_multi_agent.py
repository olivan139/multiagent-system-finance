#!/usr/bin/env python3
"""Phase 3: Multi-agent pipeline (Analyst + FactChecker + Aggregator)."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mas.agents.multi import MultiAgentSystem
from mas.config import RESULTS_DIR, DataConfig
from mas.data import load_financial_phrasebank
from mas.data.preprocessing import preprocess_batch
from mas.evaluation import (
    compute_metrics,
    load_results,
    print_metrics,
    save_results,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 3: Multi-Agent Pipeline")
    print("=" * 60)

    config = DataConfig()
    _, _, test = load_financial_phrasebank(config)
    test_texts = preprocess_batch(test.texts)
    test_labels = test.labels

    if args.max_samples:
        test_texts = test_texts[: args.max_samples]
        test_labels = test_labels[: args.max_samples]
        print(f"  (limited to {args.max_samples} test samples)")

    agent = MultiAgentSystem()
    t0 = time.time()
    preds = agent.predict_labels(test_texts)
    latency = (time.time() - t0) * 1000

    result = compute_metrics(
        test_labels,
        preds,
        model_name="Multi-Agent Pipeline",
        latency_ms=latency,
        cost_usd=agent.total_cost_usd,
        metadata={"total_tokens": agent.total_tokens},
    )
    print_metrics(result)

    prev_path = RESULTS_DIR / "phase2_results.json"
    prev = load_results(prev_path) if prev_path.exists() else []
    save_results(prev + [result], RESULTS_DIR / "phase3_results.json")
    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
