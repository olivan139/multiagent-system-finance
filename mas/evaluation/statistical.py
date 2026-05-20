"""Statistical significance tests for comparing classifiers.

Implements three complementary tests used widely in the ML literature:
    - McNemar's test (pairwise; tests whether two classifiers have
      significantly different error rates on the same test set)
    - Bootstrap confidence intervals (95% CI for accuracy / F1)
    - Friedman test + Nemenyi post-hoc (compares many classifiers across
      multiple datasets or folds)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar


@dataclass
class McNemarResult:
    model_a: str
    model_b: str
    n_a_only_correct: int
    n_b_only_correct: int
    n_both_correct: int
    n_both_wrong: int
    statistic: float
    p_value: float
    significant_at_0_05: bool


@dataclass
class BootstrapCI:
    model_name: str
    metric: str
    point_estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_bootstrap: int


def mcnemar_test(
    y_true: list[str],
    y_pred_a: list[str],
    y_pred_b: list[str],
    model_a: str = "A",
    model_b: str = "B",
) -> McNemarResult:
    """McNemar's test on the 2x2 contingency table of correctness.

    Uses the exact binomial test when off-diagonal counts are small (<25),
    otherwise the chi-squared approximation with continuity correction.
    """
    y_true_a = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)
    correct_a = y_pred_a == y_true_a
    correct_b = y_pred_b == y_true_a

    n_both_correct = int(np.sum(correct_a & correct_b))
    n_a_only = int(np.sum(correct_a & ~correct_b))
    n_b_only = int(np.sum(~correct_a & correct_b))
    n_both_wrong = int(np.sum(~correct_a & ~correct_b))

    table = [[n_both_correct, n_a_only], [n_b_only, n_both_wrong]]
    use_exact = (n_a_only + n_b_only) < 25
    res = mcnemar(table, exact=use_exact, correction=not use_exact)

    return McNemarResult(
        model_a=model_a,
        model_b=model_b,
        n_a_only_correct=n_a_only,
        n_b_only_correct=n_b_only,
        n_both_correct=n_both_correct,
        n_both_wrong=n_both_wrong,
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        significant_at_0_05=bool(res.pvalue < 0.05),
    )


def pairwise_mcnemar(
    y_true: list[str],
    predictions: dict[str, list[str]],
) -> list[McNemarResult]:
    """Run McNemar's test on all pairs of models in ``predictions``."""
    out: list[McNemarResult] = []
    for (name_a, preds_a), (name_b, preds_b) in combinations(predictions.items(), 2):
        out.append(mcnemar_test(y_true, preds_a, preds_b, model_a=name_a, model_b=name_b))
    return out


def bootstrap_ci(
    y_true: list[str],
    y_pred: list[str],
    metric: str = "accuracy",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
    model_name: str = "",
) -> BootstrapCI:
    """Compute bootstrap percentile CI for ``metric``.

    Supported metrics: 'accuracy', 'weighted_f1', 'macro_f1'.
    """
    rng = np.random.default_rng(random_seed)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n = len(y_true_arr)

    if metric == "accuracy":
        score_fn = lambda yt, yp: accuracy_score(yt, yp)
    elif metric == "weighted_f1":
        score_fn = lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0)
    elif metric == "macro_f1":
        score_fn = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    point = float(score_fn(y_true_arr, y_pred_arr))

    scores = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = score_fn(y_true_arr[idx], y_pred_arr[idx])

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(scores, alpha))
    hi = float(np.quantile(scores, 1.0 - alpha))

    return BootstrapCI(
        model_name=model_name,
        metric=metric,
        point_estimate=point,
        ci_low=lo,
        ci_high=hi,
        confidence_level=confidence,
        n_bootstrap=n_bootstrap,
    )


def friedman_test(
    scores_per_model: dict[str, list[float]],
) -> dict:
    """Friedman test across multiple models on multiple datasets/folds.

    ``scores_per_model`` maps model name to a list of per-dataset scores.
    All lists must have the same length (one score per dataset/fold).
    Returns the chi-squared statistic, p-value, and average ranks.
    """
    names = list(scores_per_model.keys())
    matrix = np.array([scores_per_model[n] for n in names])
    if matrix.shape[1] < 2:
        return {
            "info": "Friedman test requires >=2 datasets/folds; skipping",
            "n_datasets": matrix.shape[1],
        }

    stat, p = stats.friedmanchisquare(*matrix)

    ranks = np.empty_like(matrix, dtype=np.float64)
    for j in range(matrix.shape[1]):
        ranks[:, j] = stats.rankdata(-matrix[:, j])
    avg_ranks = ranks.mean(axis=1)

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant_at_0_05": bool(p < 0.05),
        "models": names,
        "average_ranks": {n: float(r) for n, r in zip(names, avg_ranks)},
        "n_datasets": int(matrix.shape[1]),
    }


def print_mcnemar_results(results: list[McNemarResult]) -> None:
    print("\n" + "=" * 70)
    print("  McNemar's Test (pairwise)")
    print("=" * 70)
    print(
        f"{'Model A':<25} {'Model B':<25} {'a-only':>7} {'b-only':>7} " f"{'p-value':>9} {'sig':>4}"
    )
    print("-" * 70)
    for r in results:
        sig = (
            "***"
            if r.p_value < 0.001
            else ("**" if r.p_value < 0.01 else ("*" if r.p_value < 0.05 else ""))
        )
        print(
            f"{r.model_a[:24]:<25} {r.model_b[:24]:<25} {r.n_a_only_correct:>7} "
            f"{r.n_b_only_correct:>7} {r.p_value:>9.4f} {sig:>4}"
        )
    print("=" * 70)
    print("  Significance: * p<0.05, ** p<0.01, *** p<0.001")


def print_bootstrap_cis(cis: list[BootstrapCI]) -> None:
    print("\n" + "=" * 70)
    print(f"  Bootstrap Confidence Intervals")
    print("=" * 70)
    print(f"{'Model':<35} {'Metric':<15} {'Point':>8} {'CI Low':>8} {'CI High':>8}")
    print("-" * 70)
    for ci in cis:
        print(
            f"{ci.model_name[:34]:<35} {ci.metric:<15} "
            f"{ci.point_estimate:>8.4f} {ci.ci_low:>8.4f} {ci.ci_high:>8.4f}"
        )
    print("=" * 70)


def print_friedman(result: dict) -> None:
    print("\n" + "=" * 70)
    print("  Friedman Test (multiple classifiers across datasets/folds)")
    print("=" * 70)
    if "info" in result:
        print(f"  {result['info']}")
        return
    print(f"  Chi-squared:      {result['statistic']:.4f}")
    print(f"  p-value:          {result['p_value']:.4f}")
    print(f"  Significant 0.05: {result['significant_at_0_05']}")
    print(f"  Datasets:         {result['n_datasets']}")
    print(f"  Average Ranks (lower = better):")
    sorted_ranks = sorted(result["average_ranks"].items(), key=lambda kv: kv[1])
    for name, rank in sorted_ranks:
        print(f"    {name:<35} {rank:.3f}")
    print("=" * 70)


def save_statistical_report(
    mcnemar_results: list[McNemarResult],
    bootstrap_cis: list[BootstrapCI],
    friedman_result: dict | None,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mcnemar": [asdict(r) for r in mcnemar_results],
        "bootstrap_ci": [asdict(r) for r in bootstrap_cis],
        "friedman": friedman_result or {},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nStatistical report saved to {path}")
