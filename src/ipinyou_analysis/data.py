from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import bz2

IPINYOU_COLUMNS = [
    "bidid",
    "timestamp",
    "logtype",
    "ipinyouid",
    "useragent",
    "ip",
    "region",
    "city",
    "adexchange",
    "domain",
    "url",
    "urlid",
    "slotid",
    "slotwidth",
    "slotheight",
    "slotvisibility",
    "slotformat",
    "slotprice",
    "creative",
    "bidprice",
    "payprice",
    "keypage",
    "advertiser",
    "usertag",
]

BID_COLUMNS = [
    "bidid",
    "timestamp",
    "ipinyouid",
    "useragent",
    "ip",
    "region",
    "city",
    "adexchange",
    "domain",
    "url",
    "urlid",
    "slotid",
    "slotwidth",
    "slotheight",
    "slotvisibility",
    "slotformat",
    "slotprice",
    "creative",
    "bidprice",
    "advertiser",
    "usertag",
]

EVENT_COLUMNS = [
    "bidid",
    "timestamp",
    "logtype",
    "ipinyouid",
    "useragent",
    "ip",
    "region",
    "city",
    "adexchange",
    "domain",
    "url",
    "urlid",
    "slotid",
    "slotwidth",
    "slotheight",
    "slotvisibility",
    "slotformat",
    "slotprice",
    "creative",
    "bidprice",
    "payprice",
    "keypage",
    "advertiser",
    "usertag",
]

NUMERIC_COLUMNS = [
    "click",
    "conversion",
    "bid_request",
    "impression",
    "won_auction",
    "weekday",
    "hour",
    "region",
    "city",
    "adexchange",
    "slotwidth",
    "slotheight",
    "slotvisibility",
    "slotformat",
    "slotprice",
    "bidprice",
    "payprice",
    "advertiser",
]

DEFAULT_GLOB_PATTERNS = ("*.txt", "*.log", "*.csv", "*.tsv", "*.bz2")
FORMATTED_LOG_COLUMNS = ["click", "weekday", "hour", *IPINYOU_COLUMNS]
DEFAULT_SEASON_FOLDERS = ("training2nd", "training3rd")


def _normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]
    if "user_tag" in df.columns and "usertag" not in df.columns:
        df = df.rename(columns={"user_tag": "usertag"})
    return df


def _resolve_dataset_root(data_dir: Path | str) -> Path:
    data_dir = Path(data_dir)
    nested_root = data_dir / "ipinyou.contest.dataset"
    if nested_root.exists():
        return nested_root
    return data_dir


def _first_line_has_header(path: Path, delimiter: str) -> bool:
    if path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip().lower()
    else:
        first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip().lower()
    tokens = [token.strip() for token in first_line.split(delimiter)]
    known_columns = {column.lower() for column in IPINYOU_COLUMNS} | {"click", "weekday", "hour"}
    return any(token in known_columns for token in tokens)


def _assign_default_columns(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.columns) == len(IPINYOU_COLUMNS):
        df.columns = [column.lower() for column in IPINYOU_COLUMNS]
    elif len(df.columns) == len(FORMATTED_LOG_COLUMNS):
        df.columns = [column.lower() for column in FORMATTED_LOG_COLUMNS]
    return df


def _read_delimited_file(path: Path) -> pd.DataFrame:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    has_header = _first_line_has_header(path, delimiter)
    frame = pd.read_csv(path, sep=delimiter, header=0 if has_header else None)
    if not has_header:
        frame = _assign_default_columns(frame)
    return frame


def discover_log_files(data_dir: Path | str, patterns: Iterable[str] = DEFAULT_GLOB_PATTERNS) -> list[Path]:
    data_dir = Path(data_dir)
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(data_dir.rglob(pattern)))
    return [
        path
        for path in matches
        if path.is_file() and path.name.lower() not in {"schema.txt", "readme", "readme.md"}
    ]


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if column in NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _read_raw_log(path: Path, column_names: Sequence[str], nrows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        names=list(column_names),
        header=None,
        compression="bz2" if path.suffix == ".bz2" else "infer",
        low_memory=False,
        nrows=nrows,
    )
    return _normalise_column_names(frame)


def _read_filtered_raw_log(
    path: Path,
    column_names: Sequence[str],
    bidids: set[str],
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    if not path.exists() or not bidids:
        return pd.DataFrame(columns=column_names)

    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        sep="\t",
        names=list(column_names),
        header=None,
        compression="bz2" if path.suffix == ".bz2" else "infer",
        low_memory=False,
        chunksize=chunk_size,
    )
    for chunk in reader:
        filtered = chunk[chunk["bidid"].isin(bidids)]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame(columns=column_names)
    return _normalise_column_names(pd.concat(chunks, ignore_index=True))


def _load_one_training_day(
    season_path: Path,
    bid_path: Path,
    season_folder: str,
    max_rows_per_file: int | None,
) -> pd.DataFrame:
    """Load and join bid/imp/clk/conv for a single bid log file."""
    date_token = bid_path.name.split(".")[1]
    imp_path = season_path / f"imp.{date_token}.txt.bz2"
    clk_path = season_path / f"clk.{date_token}.txt.bz2"
    conv_path = season_path / f"conv.{date_token}.txt.bz2"

    bids = _read_raw_log(bid_path, BID_COLUMNS, nrows=max_rows_per_file)
    sampled_bidids = set(bids["bidid"].astype(str))
    impressions = _read_filtered_raw_log(imp_path, EVENT_COLUMNS, sampled_bidids)
    clicks = _read_filtered_raw_log(clk_path, EVENT_COLUMNS, sampled_bidids)
    conversions = _read_filtered_raw_log(conv_path, EVENT_COLUMNS, sampled_bidids)

    impression_lookup = (
        impressions[["bidid", "payprice", "keypage", "usertag"]]
        .drop_duplicates(subset="bidid")
        .rename(columns={"usertag": "impression_usertag"})
    )
    frame = bids.merge(impression_lookup, on="bidid", how="left")
    frame["usertag"] = frame["impression_usertag"].combine_first(frame.get("usertag"))
    frame = frame.drop(columns=["impression_usertag"], errors="ignore")

    frame["bid_request"] = 1
    frame["impression"] = frame["bidid"].isin(impressions["bidid"]).astype(int)
    frame["click"] = frame["bidid"].isin(clicks["bidid"]).astype(int)
    frame["conversion"] = frame["bidid"].isin(conversions["bidid"]).astype(int)
    frame["dataset_split"] = "train"
    frame["season_folder"] = season_folder
    frame["log_date"] = date_token
    frame["source_file"] = bid_path.name
    return enrich_time_columns(_coerce_numeric_columns(frame))


def iter_ipinyou_training_days(
    data_dir: Path | str,
    season_folders: Sequence[str] = DEFAULT_SEASON_FOLDERS,
    max_days: int | None = 1,
    max_rows_per_file: int | None = 200_000,
):
    """Yield one day's joined training frame at a time for incremental pipelines.

    Yields tuples of (season_folder, date_token, source_file_name, dataframe).
    """
    dataset_root = _resolve_dataset_root(data_dir)
    yielded_any = False

    for season_folder in season_folders:
        season_path = dataset_root / season_folder
        if not season_path.exists():
            continue

        bid_files = sorted(season_path.glob("bid.*.txt.bz2"))
        if max_days is not None:
            bid_files = bid_files[:max_days]

        for bid_path in bid_files:
            date_token = bid_path.name.split(".")[1]
            frame = _load_one_training_day(season_path, bid_path, season_folder, max_rows_per_file)
            yielded_any = True
            yield season_folder, date_token, bid_path.name, frame

    if not yielded_any:
        raise FileNotFoundError(
            f"No raw iPinYou training files were found under {dataset_root}. "
            "Expected folders such as training2nd and training3rd."
        )


def _load_raw_training_dataset(
    data_dir: Path | str,
    season_folders: Sequence[str] = DEFAULT_SEASON_FOLDERS,
    max_days: int | None = 1,
    max_rows_per_file: int | None = 200_000,
) -> pd.DataFrame:
    dataset_root = _resolve_dataset_root(data_dir)
    frames: list[pd.DataFrame] = []

    for season_folder in season_folders:
        season_path = dataset_root / season_folder
        if not season_path.exists():
            continue

        bid_files = sorted(season_path.glob("bid.*.txt.bz2"))
        if max_days is not None:
            bid_files = bid_files[:max_days]

        for bid_path in bid_files:
            frames.append(
                _load_one_training_day(season_path, bid_path, season_folder, max_rows_per_file)
            )

    if not frames:
        raise FileNotFoundError(
            f"No raw iPinYou training files were found under {dataset_root}. "
            "Expected folders such as training2nd and training3rd."
        )

    return pd.concat(frames, ignore_index=True, sort=False)


def _load_generic_logs(data_dir: Path | str, max_files: int | None = None) -> pd.DataFrame:
    files = discover_log_files(data_dir)
    if max_files is not None:
        files = files[:max_files]

    if not files:
        raise FileNotFoundError(
            f"No candidate data files found under {data_dir}. "
            "Place iPinYou raw logs or formalised campaign logs in data/raw."
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        frame = _read_delimited_file(path)
        frame = _normalise_column_names(frame)

        if "click" not in frame.columns and "logtype" in frame.columns:
            frame["click"] = (pd.to_numeric(frame["logtype"], errors="coerce") == 2).astype(int)

        frame["source_file"] = path.name
        frames.append(_coerce_numeric_columns(frame))

    return enrich_time_columns(pd.concat(frames, ignore_index=True, sort=False))


def load_ipinyou_logs(
    data_dir: Path | str,
    max_files: int | None = None,
    season_folders: Sequence[str] = DEFAULT_SEASON_FOLDERS,
    max_days: int | None = 1,
    max_rows_per_file: int | None = 200_000,
) -> pd.DataFrame:
    """Load iPinYou data from raw season logs or formatted files.

    By default this loads the real raw training logs from seasons 2 and 3,
    which include advertiser IDs and user tags and are the best fit for
    campaign analysis and CTR modeling.
    """
    dataset_root = _resolve_dataset_root(data_dir)
    raw_training_dirs = [dataset_root / folder for folder in season_folders]
    if any(path.exists() for path in raw_training_dirs):
        return _load_raw_training_dataset(
            data_dir=dataset_root,
            season_folders=season_folders,
            max_days=max_days,
            max_rows_per_file=max_rows_per_file,
        )
    return _load_generic_logs(data_dir=data_dir, max_files=max_files)


def enrich_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" in df.columns:
        timestamp_str = df["timestamp"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
        parsed = pd.to_datetime(timestamp_str, format="%Y%m%d%H%M%S%f", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(timestamp_str, format="%Y%m%d%H%M%S", errors="coerce")
        if parsed.notna().any():
            df["event_timestamp"] = parsed
            df["hour"] = df.get("hour", parsed.dt.hour).fillna(parsed.dt.hour)
            df["weekday"] = parsed.dt.weekday
            df["weekday_name"] = parsed.dt.day_name().fillna("Unknown")
            df["date"] = parsed.dt.date.astype("string")

    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")

    if "weekday" in df.columns:
        weekday_map = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
        df["weekday_name"] = df["weekday"].map(weekday_map).fillna(df.get("weekday_name", "Unknown"))

    return df


def profile_dataset(df: pd.DataFrame) -> pd.DataFrame:
    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null_count": [df[column].notna().sum() for column in df.columns],
            "null_pct": [round(df[column].isna().mean() * 100, 2) for column in df.columns],
            "n_unique": [df[column].nunique(dropna=True) for column in df.columns],
        }
    )
    return profile.sort_values(["null_pct", "n_unique"], ascending=[True, False]).reset_index(drop=True)
