from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ..config import LABELS


@dataclass
class ExperimentResult:
    model_name: str
    accuracy: float
    weighted_f1: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    cohens_kappa: float
    per_class_report: dict
    confusion_matrix: list[list[int]]
    latency_ms_per_sample: float = 0.0
    cost_usd_per_sample: float = 0.0
    total_samples: int = 0
    metadata: dict = field(default_factory=dict)


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    model_name: str = "",
    latency_ms: float = 0.0,
    cost_usd: float = 0.0,
    metadata: dict | None = None,
) -> ExperimentResult:
    labels = LABELS
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = len(y_true)

    return ExperimentResult(
        model_name=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        weighted_f1=f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        macro_f1=f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        macro_precision=precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        macro_recall=recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        cohens_kappa=cohen_kappa_score(y_true, y_pred),
        per_class_report={k: v for k, v in report.items() if k in labels},
        confusion_matrix=cm.tolist(),
        latency_ms_per_sample=latency_ms / n if n > 0 else 0,
        cost_usd_per_sample=cost_usd / n if n > 0 else 0,
        total_samples=n,
        metadata=metadata or {},
    )


def print_metrics(result: ExperimentResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {result.model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:        {result.accuracy:.4f}")
    print(f"  Weighted F1:     {result.weighted_f1:.4f}")
    print(f"  Macro F1:        {result.macro_f1:.4f}")
    print(f"  Macro Precision: {result.macro_precision:.4f}")
    print(f"  Macro Recall:    {result.macro_recall:.4f}")
    print(f"  Cohen's Kappa:   {result.cohens_kappa:.4f}")
    if result.latency_ms_per_sample > 0:
        print(f"  Latency:         {result.latency_ms_per_sample:.1f} ms/sample")
    if result.cost_usd_per_sample > 0:
        print(f"  Cost:            ${result.cost_usd_per_sample:.6f}/sample")
    print(f"  Samples:         {result.total_samples}")

    for label in LABELS:
        if label in result.per_class_report:
            r = result.per_class_report[label]
            print(
                f"  {label:>10s}:  P={r['precision']:.3f}  "
                f"R={r['recall']:.3f}  F1={r['f1-score']:.3f}  "
                f"n={int(r['support'])}"
            )
    print(f"{'=' * 60}")


def save_results(results: list[ExperimentResult], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to {path}")


def load_results(path: Path) -> list[ExperimentResult]:
    with open(path) as f:
        data = json.load(f)
    results = []
    for d in data:
        results.append(ExperimentResult(**d))
    return results


def comparison_table(results: list[ExperimentResult]) -> str:
    header = (
        f"{'Model':<35} {'Acc':>6} {'W-F1':>6} {'M-F1':>6} " f"{'Kappa':>6} {'ms/s':>7} {'$/s':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lat = f"{r.latency_ms_per_sample:.1f}" if r.latency_ms_per_sample > 0 else "-"
        cost = f"${r.cost_usd_per_sample:.5f}" if r.cost_usd_per_sample > 0 else "-"
        lines.append(
            f"{r.model_name:<35} {r.accuracy:>6.4f} {r.weighted_f1:>6.4f} "
            f"{r.macro_f1:>6.4f} {r.cohens_kappa:>6.4f} {lat:>7s} {cost:>9s}"
        )
    return "\n".join(lines)


def plot_confusion_matrices(results: list[ExperimentResult], save_path: Path | None = None) -> None:
    n = len(results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, r in enumerate(results):
        cm = np.array(r.confusion_matrix)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABELS,
            yticklabels=LABELS,
            ax=axes[i],
        )
        axes[i].set_title(r.model_name, fontsize=10)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()


def plot_comparison_bar(results: list[ExperimentResult], save_path: Path | None = None) -> None:
    metrics = ["accuracy", "weighted_f1", "macro_f1", "cohens_kappa"]
    names = [r.model_name for r in results]
    x = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2), 6))
    for i, metric in enumerate(metrics):
        values = [getattr(r, metric) for r in results]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()
