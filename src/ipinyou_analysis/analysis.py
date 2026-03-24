from __future__ import annotations

from typing import Sequence

import pandas as pd


def add_auction_metrics(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "bid_request" not in frame.columns:
        frame["bid_request"] = 1
    if "impression" not in frame.columns:
        frame["impression"] = 1

    if "click" not in frame.columns:
        frame["click"] = 0

    if "payprice" in frame.columns:
        payprice = pd.to_numeric(frame["payprice"], errors="coerce").fillna(0)
        frame["spend"] = payprice / 1000.0
        if "won_auction" not in frame.columns:
            frame["won_auction"] = (payprice > 0).astype(int)
    else:
        frame["spend"] = 0.0
        frame["won_auction"] = 0

    if "bidprice" in frame.columns:
        bidprice = pd.to_numeric(frame["bidprice"], errors="coerce").fillna(0)
        frame["bid_cost_proxy"] = bidprice / 1000.0
    else:
        frame["bid_cost_proxy"] = 0.0

    return frame


def _aggregate(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    summary = (
        frame.groupby(list(group_cols), dropna=False)
        .agg(
            bid_requests=("bid_request", "sum"),
            impressions=("impression", "sum"),
            clicks=("click", "sum"),
            spend=("spend", "sum"),
            bid_cost_proxy=("bid_cost_proxy", "sum"),
            wins=("won_auction", "sum"),
            avg_bidprice=("bidprice", "mean"),
            avg_payprice=("payprice", "mean"),
        )
        .reset_index()
    )
    summary["ctr"] = summary["clicks"] / summary["impressions"].clip(lower=1)
    summary["ecpc"] = summary["spend"] / summary["clicks"].clip(lower=1)
    summary["cpm"] = (summary["spend"] / summary["impressions"].clip(lower=1)) * 1000
    summary["win_rate"] = summary["wins"] / summary["bid_requests"].clip(lower=1)
    summary["pay_to_bid_ratio"] = summary["avg_payprice"] / summary["avg_bidprice"].clip(lower=1e-6)
    return summary.sort_values(["clicks", "ctr"], ascending=[False, False]).reset_index(drop=True)


def campaign_performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    frame = add_auction_metrics(df)
    group_cols = [column for column in ["advertiser", "creative", "adexchange"] if column in frame.columns]
    if not group_cols:
        raise ValueError("No campaign-level grouping columns were found in the dataset.")
    return _aggregate(frame, group_cols)


def segment_performance_summary(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    frame = add_auction_metrics(df)
    valid_cols = [column for column in group_cols if column in frame.columns]
    if not valid_cols:
        raise ValueError("None of the requested segment columns were found in the dataset.")
    return _aggregate(frame, valid_cols)
