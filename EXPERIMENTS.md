# CTR evaluation experiments

This document explains how **generalization** is measured in this repo, and what it is **not**.

## What these experiments are not

- They do **not** reproduce the official iPinYou competition leaderboard score on the withheld test set.
- They are **research-style** splits you control (seasons, calendar days, row caps), so numbers depend on how much data you load.

## Cross-season: Season 2 train → Season 3 test

**Intent:** Measure robustness when training and testing data come from **different seasons** (time and regime change).

**Procedure:** Fit logistic regression and histogram gradient boosting on all rows from `training2nd` after feature engineering, then evaluate on the **same feature set** built from `training3rd`.

**Outputs:** `data/processed/experiments/cross_season_s2_train_s3_test.csv` plus row counts and `log_date` tokens in `experiment_report.json`.

## Within-season temporal split (training2nd)

**Intent:** Show that evaluation is **not only** cross-season. Here, train and test both come from **the same season**, split by **early vs. later calendar days** (`log_date`).

**Procedure:** Sort distinct `log_date` values, use the first fraction (default 70%) of days for training and the remaining days for testing. Fit on the training slice, evaluate on the held-out days.

**Outputs:** `data/processed/experiments/within_season_temporal_training2nd.csv` and the same metadata in `experiment_report.json`.

## Baseline: random split inside Season 2

The script also writes `within_s2_random_split_validation.csv`, which mirrors the main pipeline’s **train/validation split** on a single season’s modeling frame. It is useful for comparison but does not measure temporal or cross-season shift.

## How to run

From the repository root (with the dataset under `data/raw/` as in the main README):

```bash
python run_ctr_experiments.py
```

Defaults cap load per season (`--max-days`, `--max-rows-per-file`) so a first run stays feasible on a laptop. Use `0` for those flags only when you want full volume and have the resources.

```bash
python run_ctr_experiments.py --max-days 5 --max-rows-per-file 200000
```

See `python run_ctr_experiments.py --help` for all options.
