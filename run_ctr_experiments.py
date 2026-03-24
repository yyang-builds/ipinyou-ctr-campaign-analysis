#!/usr/bin/env python3
"""CTR evaluation experiments for portfolio reporting.

1. Cross-season: train on season 2 (``training2nd``), test on season 3 (``training3rd``).
   This measures generalization across time and regime change. It does **not** reproduce the
   official withheld iPinYou leaderboard test.

2. Within-season temporal: train on early ``log_date`` days in ``training2nd``, test on later
   days in the same season. This complements cross-season results when distributions are closer.

Default row caps keep runs feasible on a laptop; increase with care.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd

from ipinyou_analysis.data import load_ipinyou_logs
from ipinyou_analysis.features import build_modeling_frame
from ipinyou_analysis.modeling import (
    evaluate_models,
    evaluate_models_on_frame,
    fit_ctr_models_full_data,
    train_ctr_models,
)


def temporal_split_by_log_date(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if "log_date" not in df.columns:
        raise ValueError("Column log_date is required for a temporal split.")
    dates = sorted({str(d) for d in df["log_date"].dropna().unique()})
    if len(dates) < 2:
        raise ValueError("Need at least two distinct log_date values for a temporal split.")

    n_train = max(1, int(len(dates) * train_frac))
    if n_train >= len(dates):
        n_train = len(dates) - 1
    train_dates = dates[:n_train]
    test_dates = dates[n_train:]
    train_set, test_set = set(train_dates), set(test_dates)
    ld = df["log_date"].astype(str)
    train_df = df[ld.isin(train_set)].copy()
    test_df = df[ld.isin(test_set)].copy()
    return train_df, test_df, train_dates, test_dates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S2→S3 and within-season CTR experiments.")
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument(
        "--max-days",
        type=int,
        default=2,
        help="Days per season to load (per bid file). Use 0 for all days.",
    )
    p.add_argument(
        "--max-rows-per-file",
        type=int,
        default=50_000,
        help="Row cap per raw bid file. Use 0 for no cap.",
    )
    p.add_argument(
        "--temporal-train-frac",
        type=float,
        default=0.7,
        help="Fraction of distinct log_date values in training2nd used for training in the temporal split.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/experiments"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir if args.data_dir.is_absolute() else PROJECT_ROOT / args.data_dir
    out_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    max_days = None if args.max_days == 0 else args.max_days
    max_rows = None if args.max_rows_per_file == 0 else args.max_rows_per_file

    report: dict = {
        "framing": (
            "Cross-season (S2 train, S3 test) tests robustness across seasons and time; "
            "within-season temporal split tests holdout on later calendar days in the same season. "
            "Neither reproduces the official withheld iPinYou test set."
        ),
        "load_settings": {
            "max_days": max_days,
            "max_rows_per_file": max_rows,
        },
    }

    # --- Cross-season: train on S2, evaluate on S3 ---
    df_s2 = load_ipinyou_logs(
        data_dir,
        season_folders=["training2nd"],
        max_days=max_days,
        max_rows_per_file=max_rows,
    )
    df_s3 = load_ipinyou_logs(
        data_dir,
        season_folders=["training3rd"],
        max_days=max_days,
        max_rows_per_file=max_rows,
    )
    df_s2_m = build_modeling_frame(df_s2)
    df_s3_m = build_modeling_frame(df_s3)

    models_cross = fit_ctr_models_full_data(df_s2_m)
    cross_metrics = evaluate_models_on_frame(models_cross, df_s3_m)
    cross_metrics.insert(0, "experiment", "cross_season_s2_train_s3_test")
    cross_metrics.to_csv(out_dir / "cross_season_s2_train_s3_test.csv", index=False)

    report["cross_season_s2_train_s3_test"] = {
        "train_season": "training2nd",
        "train_rows": int(len(df_s2)),
        "train_log_dates": sorted({str(x) for x in df_s2["log_date"].dropna().unique()}),
        "test_season": "training3rd",
        "test_rows": int(len(df_s3)),
        "test_log_dates": sorted({str(x) for x in df_s3["log_date"].dropna().unique()}),
    }

    # --- Within-season temporal on training2nd only ---
    train_df, test_df, train_dates, test_dates = temporal_split_by_log_date(df_s2, args.temporal_train_frac)
    train_m = build_modeling_frame(train_df)
    test_m = build_modeling_frame(test_df)
    models_temp = fit_ctr_models_full_data(train_m)
    temporal_metrics = evaluate_models_on_frame(models_temp, test_m)
    temporal_metrics.insert(0, "experiment", "within_season_temporal_training2nd")
    temporal_metrics.to_csv(out_dir / "within_season_temporal_training2nd.csv", index=False)

    report["within_season_temporal_training2nd"] = {
        "season": "training2nd",
        "temporal_train_frac": args.temporal_train_frac,
        "train_rows": int(len(train_df)),
        "train_log_dates": train_dates,
        "test_rows": int(len(test_df)),
        "test_log_dates": test_dates,
    }

    # --- Optional baseline: random split inside S2 (same as original pipeline style) ---
    artifacts = train_ctr_models(df_s2_m)
    bench = evaluate_models(artifacts)
    bench.insert(0, "experiment", "within_s2_train_val_test_split_baseline")
    bench.to_csv(out_dir / "within_s2_random_split_validation.csv", index=False)

    (out_dir / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote CSVs and experiment_report.json under {out_dir}")


if __name__ == "__main__":
    main()
