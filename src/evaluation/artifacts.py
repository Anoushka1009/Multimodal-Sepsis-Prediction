from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from .metrics import compute_binary_classification_metrics


def build_curve_tables(y_true, y_prob) -> Dict[str, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true, y_prob)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    roc_df = pd.DataFrame({
        'fpr': roc_fpr,
        'tpr': roc_tpr,
        'threshold': np.append(roc_thresholds, np.nan)[: len(roc_fpr)],
    })
    pr_df = pd.DataFrame({
        'precision': pr_precision,
        'recall': pr_recall,
        'threshold': np.append(pr_thresholds, np.nan)[: len(pr_precision)],
    })
    return {'roc_curve': roc_df, 'pr_curve': pr_df}


def build_calibration_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
    return pd.DataFrame({'mean_predicted_probability': prob_pred, 'fraction_positive': prob_true})


def summarize_predictions(predictions_df: pd.DataFrame, threshold: float = 0.5, calibration_bins: int = 10) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:
    metrics = compute_binary_classification_metrics(predictions_df['y_true'], predictions_df['y_prob'], threshold=threshold)
    curves = build_curve_tables(predictions_df['y_true'], predictions_df['y_prob'])
    curves['calibration'] = build_calibration_table(predictions_df['y_true'], predictions_df['y_prob'], n_bins=calibration_bins)
    return metrics, curves


def build_lead_time_table(predictions_df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    positives = predictions_df.loc[predictions_df['y_true'] == 1].copy()
    if positives.empty:
        return pd.DataFrame(columns=['horizon_hours', 'detected_positive_count', 'median_lead_time_hours'])
    return pd.DataFrame([
        {
            'horizon_hours': int(horizon_hours),
            'detected_positive_count': int(len(positives)),
            'median_lead_time_hours': float(horizon_hours),
        }
    ])


def collect_prediction_files(directory: str | pd.PathLike) -> Iterable[pd.PathLike]:
    directory = pd.PathLike(directory) if False else directory
    return []
