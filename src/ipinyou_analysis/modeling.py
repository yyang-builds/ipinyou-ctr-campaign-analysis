from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


DEFAULT_FEATURES = [
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
    "creative",
    "advertiser",
    "bidprice",
    "payprice",
    "won_auction",
    "slot_area",
    "slot_aspect_ratio",
    "bid_gap",
    "bid_to_pay_ratio",
    "floor_gap",
    "device_type",
    "browser",
    "user_tag_count",
    "keypage_prefix",
]


@dataclass
class CTRModelArtifacts:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    models: dict[str, Pipeline]


def _pick_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in DEFAULT_FEATURES if column in df.columns]


def _normalise_categorical_types(feature_frame: pd.DataFrame) -> pd.DataFrame:
    frame = feature_frame.copy()
    numeric_cols = frame.select_dtypes(include=["number", "Int64", "Float64"]).columns.tolist()
    categorical_cols = [column for column in frame.columns if column not in numeric_cols]
    for column in categorical_cols:
        frame[column] = frame[column].astype("string")
    return frame


def _build_preprocessors(feature_frame: pd.DataFrame) -> tuple[ColumnTransformer, ColumnTransformer]:
    numeric_cols = feature_frame.select_dtypes(include=["number", "Int64", "Float64"]).columns.tolist()
    categorical_cols = [column for column in feature_frame.columns if column not in numeric_cols]

    linear_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )
    return linear_preprocessor, tree_preprocessor


def train_ctr_models(df: pd.DataFrame, target_col: str = "click") -> CTRModelArtifacts:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    feature_cols = _pick_feature_columns(df)
    if not feature_cols:
        raise ValueError("No supported modeling features were found.")

    modeling_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    X = _normalise_categorical_types(modeling_df[feature_cols])
    y = modeling_df[target_col].astype(int)

    def _can_stratify(labels: pd.Series) -> bool:
        counts = labels.value_counts()
        return len(counts) > 1 and int(counts.min()) >= 2

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y if _can_stratify(y) else None,
        random_state=42,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp if _can_stratify(y_temp) else None,
        random_state=42,
    )

    linear_preprocessor, tree_preprocessor = _build_preprocessors(X_train)

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", linear_preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    n_jobs=None,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    gradient_boosting = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_depth=8,
                    max_iter=250,
                    min_samples_leaf=100,
                    random_state=42,
                ),
            ),
        ]
    )

    models = {
        "logistic_regression": logistic_regression.fit(X_train, y_train),
        "hist_gradient_boosting": gradient_boosting.fit(X_train, y_train),
    }

    return CTRModelArtifacts(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        models=models,
    )


def fit_ctr_models_full_data(df: pd.DataFrame, target_col: str = "click") -> dict[str, Pipeline]:
    """Fit both CTR models on all rows in ``df`` (for external train/test protocols)."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    feature_cols = _pick_feature_columns(df)
    if not feature_cols:
        raise ValueError("No supported modeling features were found.")

    modeling_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    X = _normalise_categorical_types(modeling_df[feature_cols])
    y = modeling_df[target_col].astype(int)

    linear_preprocessor, tree_preprocessor = _build_preprocessors(X)

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", linear_preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    n_jobs=None,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    gradient_boosting = Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_depth=8,
                    max_iter=250,
                    min_samples_leaf=100,
                    random_state=42,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": logistic_regression.fit(X, y),
        "hist_gradient_boosting": gradient_boosting.fit(X, y),
    }


def evaluate_models_on_frame(models: dict[str, Pipeline], df: pd.DataFrame, target_col: str = "click") -> pd.DataFrame:
    """Evaluate fitted models on a single labeled frame (e.g. season-holdout or temporal test)."""
    feature_cols = _pick_feature_columns(df)
    if not feature_cols:
        raise ValueError("No supported modeling features were found.")

    modeling_df = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
    X = _normalise_categorical_types(modeling_df[feature_cols])
    y = modeling_df[target_col].astype(int)

    rows = []
    for name, model in models.items():
        scores = model.predict_proba(X)[:, 1]
        if y.nunique() < 2:
            rows.append(
                {
                    "model": name,
                    "n_rows": len(y),
                    "n_positive": int(y.sum()),
                    "roc_auc": float("nan"),
                    "average_precision": float("nan"),
                    "log_loss": float("nan"),
                    "brier_score": float("nan"),
                    "precision_at_mid_pr_threshold": float("nan"),
                    "recall_at_mid_pr_threshold": float("nan"),
                }
            )
            continue

        roc_auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)
        precision, recall, thresholds = precision_recall_curve(y, scores)
        threshold = thresholds[min(len(thresholds) - 1, max(len(thresholds) // 2, 0))] if len(thresholds) else 0.5
        rows.append(
            {
                "model": name,
                "n_rows": len(y),
                "n_positive": int(y.sum()),
                "roc_auc": roc_auc,
                "average_precision": ap,
                "log_loss": log_loss(y, scores, labels=[0, 1]),
                "brier_score": brier_score_loss(y, scores),
                "precision_at_mid_pr_threshold": _precision_at_threshold(y, scores, threshold),
                "recall_at_mid_pr_threshold": float(recall[min(len(recall) - 1, max(len(recall) // 2, 0))]),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False, na_position="last").reset_index(drop=True)


def _precision_at_threshold(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> float:
    predicted = (y_score >= threshold).astype(int)
    positives = predicted.sum()
    if positives == 0:
        return 0.0
    return float(((predicted == 1) & (y_true.to_numpy() == 1)).sum() / positives)


def evaluate_models(artifacts: CTRModelArtifacts) -> pd.DataFrame:
    rows = []
    for name, model in artifacts.models.items():
        scores = model.predict_proba(artifacts.X_valid)[:, 1]
        precision, recall, thresholds = precision_recall_curve(artifacts.y_valid, scores)
        threshold = thresholds[min(len(thresholds) - 1, max(len(thresholds) // 2, 0))] if len(thresholds) else 0.5
        rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(artifacts.y_valid, scores),
                "average_precision": average_precision_score(artifacts.y_valid, scores),
                "log_loss": log_loss(artifacts.y_valid, scores, labels=[0, 1]),
                "brier_score": brier_score_loss(artifacts.y_valid, scores),
                "precision_at_mid_pr_threshold": _precision_at_threshold(artifacts.y_valid, scores, threshold),
                "recall_at_mid_pr_threshold": float(recall[min(len(recall) - 1, max(len(recall) // 2, 0))]),
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
