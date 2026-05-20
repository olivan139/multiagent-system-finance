#!/usr/bin/env python3
"""Streamlit dashboard for thesis experiment results.

Run with:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_DIR = Path(__file__).parent / "results"
LABELS = ["negative", "neutral", "positive"]


@st.cache_data
def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def results_to_df(data: list[dict]) -> pd.DataFrame:
    rows = []
    for r in data:
        rows.append(
            {
                "Model": r["model_name"],
                "Accuracy": r["accuracy"],
                "Weighted F1": r["weighted_f1"],
                "Macro F1": r["macro_f1"],
                "Macro Precision": r["macro_precision"],
                "Macro Recall": r["macro_recall"],
                "Cohen's Kappa": r["cohens_kappa"],
                "Latency (ms/s)": r.get("latency_ms_per_sample", 0),
                "Cost ($/s)": r.get("cost_usd_per_sample", 0),
                "Samples": r.get("total_samples", 0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="MAS Sentiment Analysis — Results",
        layout="wide",
    )
    st.title("Multi-Agent System for Social Data Analysis")
    st.markdown("### Experiment Results Dashboard")

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if not result_files:
        st.warning(
            "No results found. Run experiments first:\n\n" "```bash\npython scripts/run_all.py\n```"
        )
        return

    selected = st.selectbox(
        "Result file",
        result_files,
        format_func=lambda p: p.name,
        index=len(result_files) - 1,
    )

    data = load_results(selected)
    if not data:
        st.info("Selected file is empty.")
        return

    df = results_to_df(data)

    st.subheader("Comparison Table")
    fmt = {
        "Accuracy": "{:.4f}",
        "Weighted F1": "{:.4f}",
        "Macro F1": "{:.4f}",
        "Macro Precision": "{:.4f}",
        "Macro Recall": "{:.4f}",
        "Cohen's Kappa": "{:.4f}",
        "Latency (ms/s)": "{:.1f}",
        "Cost ($/s)": "{:.6f}",
    }
    st.dataframe(
        df.style.format(fmt).highlight_max(
            subset=["Accuracy", "Weighted F1", "Macro F1", "Cohen's Kappa"], color="#d4edda"
        ),
        use_container_width=True,
    )

    st.subheader("Metric Comparison")
    metrics_to_plot = st.multiselect(
        "Metrics",
        ["Accuracy", "Weighted F1", "Macro F1", "Cohen's Kappa"],
        default=["Accuracy", "Weighted F1", "Macro F1"],
    )
    if metrics_to_plot:
        chart_df = df.set_index("Model")[metrics_to_plot]
        st.bar_chart(chart_df)

    st.subheader("Per-Class Metrics")
    model_names = df["Model"].tolist()
    selected_model = st.selectbox("Model", model_names)
    model_data = next(r for r in data if r["model_name"] == selected_model)

    per_class = model_data.get("per_class_report", {})
    if per_class:
        pc_rows = []
        for label in LABELS:
            if label in per_class:
                pc_rows.append(
                    {
                        "Class": label,
                        "Precision": per_class[label]["precision"],
                        "Recall": per_class[label]["recall"],
                        "F1": per_class[label]["f1-score"],
                        "Support": int(per_class[label]["support"]),
                    }
                )
        st.dataframe(
            pd.DataFrame(pc_rows).style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
            ),
            use_container_width=True,
        )

    cm = model_data.get("confusion_matrix")
    if cm:
        st.subheader(f"Confusion Matrix — {selected_model}")
        cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
        st.dataframe(cm_df, use_container_width=True)

    if any(r.get("cost_usd_per_sample", 0) > 0 for r in data):
        st.subheader("Cost Analysis")
        cost_df = df[df["Cost ($/s)"] > 0][["Model", "Cost ($/s)", "Latency (ms/s)"]]
        st.dataframe(cost_df, use_container_width=True)

    for img_name in ["comparison_bar.png", "confusion_matrices.png"]:
        img_path = RESULTS_DIR / img_name
        if img_path.exists():
            st.image(str(img_path), caption=img_name.replace("_", " ").replace(".png", ""))


if __name__ == "__main__":
    main()
