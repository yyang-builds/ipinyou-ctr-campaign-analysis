from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, output_path: Path | str | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def plot_ctr_by_hour(summary: pd.DataFrame, output_path: Path | str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=summary, x="hour", y="ctr", marker="o", ax=ax, color="#2563eb")
    ax.set_title("CTR by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Click-Through Rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.2%}")
    _save(fig, output_path)
    return fig


def plot_top_regions(summary: pd.DataFrame, output_path: Path | str | None = None, top_n: int = 15) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    ordered = summary.sort_values("clicks", ascending=False).head(top_n).copy()
    ordered["region"] = ordered["region"].astype(str)
    sns.barplot(data=ordered, x="ctr", y="region", hue="region", dodge=False, legend=False, ax=ax, palette="Blues_r")
    ax.set_title(f"Top {top_n} Regions by CTR")
    ax.set_xlabel("CTR")
    ax.set_ylabel("Region")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.2%}")
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
