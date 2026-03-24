# iPinYou CTR & Campaign Performance Analysis

This project is a portfolio-ready Python workflow for analyzing RTB campaign performance with the public iPinYou dataset. It combines business-facing performance analysis with CTR modeling so you can answer questions about which advertisers, creatives, exchanges, regions, and time windows perform best.

## Project Goals

- Measure campaign performance with metrics such as impressions, clicks, CTR, spend proxy, eCPC, CPM, and win rate.
- Analyze campaign results by hour, weekday, region, city, exchange, advertiser, and creative.
- Study auction efficiency using `bidprice`, `payprice`, and floor-price fields.
- Build and compare CTR prediction models using a logistic regression baseline and a stronger tree-based model.

## Dataset

This repository is designed for the public iPinYou RTB dataset described in the iPinYou competition paper and the public schema released by the benchmark authors.

Expected schema includes fields such as:

- `bidid`
- `timestamp`
- `logtype`
- `ipinyouid`
- `useragent`
- `IP`
- `region`
- `city`
- `adexchange`
- `domain`
- `url`
- `urlid`
- `slotid`
- `slotwidth`
- `slotheight`
- `slotvisibility`
- `slotformat`
- `slotprice`
- `creative`
- `bidprice`
- `payprice`
- `keypage`
- `advertiser`
- `usertag`

The project now works directly with the downloaded raw archive layout:

- `data/raw/ipinyou.contest.dataset/training2nd`
- `data/raw/ipinyou.contest.dataset/training3rd`

These seasons are the default focus because they include `advertiser` IDs and `usertag` fields, which are the most useful for campaign analysis and CTR modeling.

## Business Questions

- Which campaigns and creatives drive the highest CTR?
- Which hours and regions consistently outperform or underperform?
- Which exchanges show better win rates or lower effective CPC?
- How does `bidprice` compare with `payprice`, and where might spend be inefficient?
- Which features are most associated with higher click probability?

## Project Structure

```text
ipinyou-ctr-campaign-analysis/
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── outputs/
│   └── figures/
├── src/
│   └── ipinyou_analysis/
├── .gitignore
├── EXPERIMENTS.md
├── pyproject.toml
├── README.md
├── requirements.txt
├── run_ctr_experiments.py
└── run_pipeline.py
```

## Methodology

### 1. Data loading and schema profiling

The loader reads raw `bid`, `imp`, `clk`, and `conv` `.txt.bz2` files, joins them by `bidid`, and builds an auction-level modeling table with:

- bid requests
- impressions
- clicks
- conversions
- bid price
- pay price
- campaign and creative metadata

It also profiles the resulting dataset for missingness and cardinality.

### 2. Performance analysis

The analysis layer computes:

- `impressions`
- `clicks`
- `ctr`
- `spend` as `payprice / 1000`
- `eCPC`
- `CPM`
- `win_rate` based on impressions divided by bid requests
- `pay_to_bid_ratio`

### 3. Feature engineering

Modeling features include:

- time fields such as `hour` and `weekday`
- geography such as `region` and `city`
- placement and exchange fields
- auction features such as `bidprice`, `payprice`, `bid_gap`, and `floor_gap`
- derived creative and user context features such as slot area, browser, device type, and user tag count

### 4. CTR modeling

Two models are included:

- Logistic Regression baseline
- Histogram-based Gradient Boosting classifier

Evaluation metrics include:

- ROC AUC
- Average precision
- Log loss
- Brier score

### 5. Generalization experiments (optional)

For portfolio-style reporting, use `run_ctr_experiments.py` to run:

- **Season 2 train / Season 3 test** — cross-season robustness (not the official leaderboard test).
- **Within-season temporal split** on `training2nd` — early calendar days vs. later days in the same season.

Framing, file outputs, and CLI flags are documented in [EXPERIMENTS.md](EXPERIMENTS.md).

## Visualizations

The pipeline is set up to create charts for:

- CTR by hour
- Region CTR comparison
- eCPC by advertiser
- Bid price vs pay price
- Win rate by ad exchange

Generated figures are saved to `outputs/figures/`.

## Setup

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Add the dataset

Place your iPinYou files under:

```text
data/raw/ipinyou.contest.dataset/
```

The pipeline expects the original season folders such as `training2nd/` and `training3rd/` to still be inside that directory.

### 3. Run the pipeline

```bash
python run_pipeline.py
```

### 4. CTR generalization experiments (optional)

```bash
python run_ctr_experiments.py
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for how cross-season and within-season splits are defined and what gets written under `data/processed/experiments/`.

### Incremental mode (checkpoint per day)

If a long run might be interrupted, use `--incremental`. After **each** calendar day (each `bid.*.txt.bz2` file), the pipeline:

- Appends a Parquet chunk under `data/processed/incremental/chunks/`
- Refreshes `dataset_profile.csv`, summary CSVs, and figures under `data/processed/` and `outputs/figures/`
- Writes `data/processed/incremental/checkpoint.json` with completed days and row counts

CTR models are trained **once at the end** by default (faster). Use `--model-each-checkpoint` to retrain after every day.

```bash
python run_pipeline.py --incremental --max-days 2 --max-rows-per-file 0
```

Outputs will be written to:

- `data/processed/`
- `outputs/figures/`

By default, the script loads:

- `training2nd`
- `training3rd`
- `1` day per season
- up to `200000` rows per raw file

This keeps the first run manageable. To process more data:

```bash
python run_pipeline.py --max-days 3 --max-rows-per-file 0
```

To run all available days from the selected seasons:

```bash
python run_pipeline.py --max-days 0 --max-rows-per-file 0
```

## Notebooks

- `notebooks/01_eda_campaign_analysis.ipynb`
- `notebooks/02_ctr_modeling.ipynb`

These notebooks are intended for exploration, storytelling, and portfolio presentation. They use the reusable code from `src/ipinyou_analysis/`.

## GitHub Readiness

This repository is structured so you can upload it to GitHub cleanly:

- large raw data is excluded with `.gitignore`
- generated figures and processed data are excluded by default
- source code and notebooks remain lightweight and shareable
- the repository explains both business value and technical implementation

## Next Improvements

- Add region and city name mapping files for more readable charts
- Add calibration plots for CTR probabilities
- Add SHAP or permutation importance for model interpretation
- Add campaign-specific deep dives if you work with individual advertiser folders
