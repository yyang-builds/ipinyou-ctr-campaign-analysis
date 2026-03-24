from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, output_path: Path | str | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def _prettify_model_name(raw: str) -> str:
    mapping = {
        "logistic_regression": "Logistic regression",
        "hist_gradient_boosting": "Hist. gradient boosting",
    }
    return mapping.get(raw, raw.replace("_", " ").title())


def plot_model_comparison_charts(metrics: pd.DataFrame, output_path: Path | str | None = None) -> plt.Figure:
    """Bar chart grid for ROC AUC, average precision, log loss, and Brier score (validation set)."""
    if metrics.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Model comparison")
        ax.text(0.5, 0.5, "No model metrics available", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, output_path)
        return fig

    m = metrics.copy()
    m["_ord"] = m["model"].map(lambda x: 0 if "logistic" in str(x).lower() else 1)
    m = m.sort_values("_ord").drop(columns=["_ord"])
    names = m["model"].map(_prettify_model_name).tolist()
    n = len(m)
    colors = (["#2563eb", "#7c3aed", "#0d9488", "#c026d3"] * ((n // 4) + 1))[:n]
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle("CTR models — validation metrics", fontsize=14, fontweight="semibold")

    specs = [
        (axes[0, 0], "roc_auc", "ROC AUC", True),
        (axes[0, 1], "average_precision", "Average precision", True),
        (axes[1, 0], "log_loss", "Log loss", False),
        (axes[1, 1], "brier_score", "Brier score", False),
    ]
    for ax, col, label, higher_better in specs:
        vals = pd.to_numeric(m[col], errors="coerce").to_numpy(dtype=float)
        ax.bar(x, vals, color=colors, width=0.62, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=0, fontsize=10)
        suffix = " — higher is better" if higher_better else " — lower is better"
        ax.set_title(label + suffix, fontsize=11)
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    _save(fig, output_path)
    return fig


def plot_model_comparison_table(metrics: pd.DataFrame, output_path: Path | str | None = None) -> plt.Figure:
    """Rendered table of all metrics returned by ``evaluate_models``."""
    fig, ax = plt.subplots(figsize=(14, 2.8 + 0.35 * max(1, len(metrics))))
    ax.axis("off")

    if metrics.empty:
        ax.set_title("Model comparison")
        ax.text(0.5, 0.5, "No model metrics available", ha="center", va="center", transform=ax.transAxes)
        _save(fig, output_path)
        return fig

    display = metrics.copy()
    display["model"] = display["model"].map(_prettify_model_name)

    column_defs = [
        ("model", "Model"),
        ("roc_auc", "ROC AUC"),
        ("average_precision", "Average precision"),
        ("log_loss", "Log loss"),
        ("brier_score", "Brier score"),
        ("precision_at_mid_pr_threshold", "Precision @ mid PR threshold"),
        ("recall_at_mid_pr_threshold", "Recall @ mid PR threshold"),
    ]
    present = [(k, lab) for k, lab in column_defs if k in display.columns]

    rows: list[list[str]] = []
    for _, row in display.iterrows():
        line: list[str] = []
        for key, _ in present:
            val = row[key]
            if key == "model":
                line.append(str(val))
            elif pd.isna(val):
                line.append("—")
            elif key in ("roc_auc", "average_precision", "precision_at_mid_pr_threshold", "recall_at_mid_pr_threshold"):
                line.append(f"{float(val):.4f}")
            else:
                line.append(f"{float(val):.4f}")
        rows.append(line)

    col_labels = [lab for _, lab in present]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.85)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="semibold")
            cell.set_facecolor("#e2e8f0")
    ax.set_title("CTR models — full metric table (validation set)", fontsize=12, fontweight="semibold", pad=12)

    _save(fig, output_path)
    return fig


def plot_campaign_ecpc(summary: pd.DataFrame, output_path: Path | str | None = None, top_n: int = 15) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    ordered = summary.sort_values("ecpc", ascending=False).head(top_n).copy()
    # Seaborn treats numeric `y` as a continuous axis; advertiser IDs must be categorical strings.
    if "creative" in ordered.columns and ordered["advertiser"].duplicated().any():
        ordered["advertiser_label"] = (
            ordered["advertiser"].astype(str).str.cat(ordered["creative"].astype(str), sep=" · ")
        )
    else:
        ordered["advertiser_label"] = ordered["advertiser"].astype(str)
    order = ordered.sort_values("ecpc", ascending=True)["advertiser_label"].tolist()
    sns.barplot(
        data=ordered,
        x="ecpc",
        y="advertiser_label",
        order=order,
        hue="advertiser_label",
        dodge=False,
        legend=False,
        ax=ax,
        palette="magma",
    )
    ax.set_title(f"Top {top_n} Advertisers by eCPC")
    ax.set_xlabel("Effective CPC")
    ax.set_ylabel("Advertiser")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.2f}")
    _save(fig, output_path)
    return fig


def plot_bid_vs_payprice(df: pd.DataFrame, output_path: Path | str | None = None, sample_size: int = 5000) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    valid = df[["bidprice", "payprice"]].dropna()
    if valid.empty:
        ax.set_title("Bid Price vs Pay Price")
        ax.text(0.5, 0.5, "No valid bid/payprice rows available", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, output_path)
        return fig
    sample = valid.sample(n=min(sample_size, len(valid)), random_state=42)
    sns.scatterplot(data=sample, x="bidprice", y="payprice", alpha=0.4, s=40, ax=ax, color="#0f766e")
    ax.set_title("Bid Price vs Pay Price")
    ax.set_xlabel("Bid Price")
    ax.set_ylabel("Pay Price")
    _save(fig, output_path)
    return fig


def plot_win_rate_by_exchange(summary: pd.DataFrame, output_path: Path | str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    frame = summary.copy()
    frame["adexchange"] = frame["adexchange"].astype(str)
    sns.barplot(data=frame, x="adexchange", y="win_rate", hue="adexchange", dodge=False, legend=False, ax=ax, palette="viridis")
    ax.set_title("Win Rate by Ad Exchange")
    ax.set_xlabel("Ad Exchange")
    ax.set_ylabel("Win Rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.2%}")
    _save(fig, output_path)
    return fig
