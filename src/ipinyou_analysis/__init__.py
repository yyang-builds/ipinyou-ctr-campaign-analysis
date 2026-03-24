"""Utilities for iPinYou CTR and campaign performance analysis."""

from .analysis import (
    add_auction_metrics,
    campaign_performance_summary,
    segment_performance_summary,
)
from .data import (
    IPINYOU_COLUMNS,
    iter_ipinyou_training_days,
    load_ipinyou_logs,
    profile_dataset,
)
from .features import build_modeling_frame
from .modeling import evaluate_models, evaluate_models_on_frame, fit_ctr_models_full_data, train_ctr_models

__all__ = [
    "IPINYOU_COLUMNS",
    "add_auction_metrics",
    "build_modeling_frame",
    "campaign_performance_summary",
    "evaluate_models",
    "evaluate_models_on_frame",
    "fit_ctr_models_full_data",
    "iter_ipinyou_training_days",
    "load_ipinyou_logs",
    "profile_dataset",
    "segment_performance_summary",
    "train_ctr_models",
]
