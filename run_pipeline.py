from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
CACHE_PATH = PROJECT_ROOT / ".cache" / "matplotlib"
CACHE_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_PATH))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd

from ipinyou_analysis.analysis import campaign_performance_summary, segment_performance_summary
from ipinyou_analysis.data import iter_ipinyou_training_days, load_ipinyou_logs, profile_dataset
from ipinyou_analysis.features import build_modeling_frame
from ipinyou_analysis.modeling import evaluate_models, train_ctr_models
from ipinyou_analysis.plotting import (
    plot_bid_vs_payprice,
    plot_campaign_ecpc,
    plot_ctr_by_hour,
    plot_top_regions,
    plot_win_rate_by_exchange,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the iPinYou analytics pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing `ipinyou.contest.dataset` or formatted logs.",
    )
    parser.add_argument(
        "--season-folders",
        nargs="+",
        default=["training2nd", "training3rd"],
        help="Raw iPinYou season folders to load.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=1,
        help="Number of bid/imp/clk/conv dates to load per season. Use 0 for all days.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=200000,
        help="Optional row cap per raw log file for faster iteration. Use 0 for all rows.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Process one calendar day at a time: save each chunk as Parquet under "
            "data/processed/incremental/chunks/, refresh CSV summaries and figures after each day, "
            "and write checkpoint.json so you can stop anytime and keep completed work."
        ),
    )
    parser.add_argument(
        "--model-each-checkpoint",
        action="store_true",
        help=(
            "With --incremental, retrain CTR models after every day (slow). "
            "Default is to train once after all days complete."
        ),
    )
    return parser.parse_args()


def write_summaries_and_plots(df: pd.DataFrame, processed_dir: Path, figure_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    profile_dataset(df).to_csv(processed_dir / "dataset_profile.csv", index=False)

    campaign_summary = campaign_performance_summary(df)
    hour_summary = segment_performance_summary(df, ["hour"])
    region_summary = segment_performance_summary(df, ["region"])
    exchange_summary = segment_performance_summary(df, ["adexchange"])

    campaign_summary.to_csv(processed_dir / "campaign_summary.csv", index=False)
    hour_summary.to_csv(processed_dir / "hour_summary.csv", index=False)
    region_summary.to_csv(processed_dir / "region_summary.csv", index=False)
    exchange_summary.to_csv(processed_dir / "exchange_summary.csv", index=False)

    plot_ctr_by_hour(hour_summary, figure_dir / "ctr_by_hour.png")
    plot_top_regions(region_summary, figure_dir / "top_regions_ctr.png")
    plot_campaign_ecpc(campaign_summary, figure_dir / "campaign_ecpc.png")
    plot_bid_vs_payprice(df, figure_dir / "bid_vs_payprice.png")
    plot_win_rate_by_exchange(exchange_summary, figure_dir / "win_rate_by_exchange.png")


def write_model_comparison(df: pd.DataFrame, processed_dir: Path) -> None:
    modeling_df = build_modeling_frame(df)
    artifacts = train_ctr_models(modeling_df)
    evaluate_models(artifacts).to_csv(processed_dir / "model_comparison.csv", index=False)


def run_incremental(
    data_dir: Path,
    season_folders: list[str],
    max_days: int | None,
    max_rows_per_file: int | None,
    processed_dir: Path,
    figure_dir: Path,
    model_each_checkpoint: bool,
) -> None:
    incremental_root = processed_dir / "incremental"
    chunks_dir = incremental_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    cumulative_frames: list[pd.DataFrame] = []
    completed: list[dict[str, str | int]] = []

    iterator = iter_ipinyou_training_days(
        data_dir=data_dir,
        season_folders=season_folders,
        max_days=max_days,
        max_rows_per_file=max_rows_per_file,
    )

    for season_folder, date_token, source_name, df_day in iterator:
        chunk_path = chunks_dir / f"{season_folder}_{date_token}.parquet"
        df_day.to_parquet(chunk_path, index=False, engine="pyarrow")

        cumulative_frames.append(df_day)
        df_cumulative = pd.concat(cumulative_frames, ignore_index=True, sort=False)

        completed.append(
            {
                "season_folder": season_folder,
                "date_token": date_token,
                "source_file": source_name,
                "rows": int(len(df_day)),
            }
        )

        write_summaries_and_plots(df_cumulative, processed_dir, figure_dir)

        checkpoint = {
            "completed": completed,
            "cumulative_rows": int(len(df_cumulative)),
            "last_chunk": completed[-1],
        }
        (incremental_root / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

        if model_each_checkpoint:
            write_model_comparison(df_cumulative, processed_dir)

    if not cumulative_frames:
        raise RuntimeError("Incremental run produced no data.")

    df_final = pd.concat(cumulative_frames, ignore_index=True, sort=False)

    if not model_each_checkpoint:
        write_model_comparison(df_final, processed_dir)


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    figure_dir = project_root / "outputs" / "figures"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir if args.data_dir.is_absolute() else project_root / args.data_dir

    max_days = None if args.max_days == 0 else args.max_days
    max_rows = None if args.max_rows_per_file == 0 else args.max_rows_per_file

    if args.incremental:
        run_incremental(
            data_dir=data_dir,
            season_folders=list(args.season_folders),
            max_days=max_days,
            max_rows_per_file=max_rows,
            processed_dir=processed_dir,
            figure_dir=figure_dir,
            model_each_checkpoint=bool(args.model_each_checkpoint),
        )
        return

    df = load_ipinyou_logs(
        data_dir=data_dir,
        season_folders=args.season_folders,
        max_days=max_days,
        max_rows_per_file=max_rows,
    )
    write_summaries_and_plots(df, processed_dir, figure_dir)
    write_model_comparison(df, processed_dir)


if __name__ == "__main__":
    main()
