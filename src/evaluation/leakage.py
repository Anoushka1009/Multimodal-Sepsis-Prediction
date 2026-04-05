from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.metrics import compute_binary_classification_metrics
from src.models.baselines import build_baseline_models
from src.training.tabular_multimodal import (
    FEATURE_METADATA_COLUMNS,
    ID_COLUMNS,
    build_structured_augmented_tabular_dataset,
)


SUSPICIOUS_FEATURE_PATTERNS = (
    "sepsis",
    "label",
    "onset",
    "prediction_time",
    "outcome",
    "shock",
)

DEFAULT_TEXT_KEYWORDS = (
    "sepsis",
    "septic",
    "septic shock",
)


def prepare_structured_stay_table(structured_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    cfg = config.get("tabular_multimodal", {})
    return build_structured_augmented_tabular_dataset(
        structured_df,
        aggregations=cfg.get(
            "structured_aggregations",
            config.get("baselines", {}).get("tabular_aggregations", ["mean", "min", "max", "last"]),
        ),
        include_missingness=bool(cfg.get("include_missingness", True)),
        include_static_categoricals=bool(cfg.get("include_static_categoricals", True)),
    )


def summarize_patient_split_overlap(structured_df: pd.DataFrame) -> pd.DataFrame:
    stay_index = structured_df[ID_COLUMNS + ["split"]].drop_duplicates(subset=ID_COLUMNS).copy()

    split_subjects = {
        split_name: set(stay_index.loc[stay_index["split"] == split_name, "SUBJECT_ID"].dropna().astype(int).tolist())
        for split_name in ["train", "val", "test"]
    }
    split_stays = {
        split_name: {
            tuple(row)
            for row in stay_index.loc[stay_index["split"] == split_name, ID_COLUMNS].itertuples(index=False, name=None)
        }
        for split_name in ["train", "val", "test"]
    }

    rows: list[dict] = []
    for left, right in combinations(["train", "val", "test"], 2):
        subject_overlap = split_subjects[left].intersection(split_subjects[right])
        stay_overlap = split_stays[left].intersection(split_stays[right])
        rows.append(
            {
                "left_split": left,
                "right_split": right,
                "subject_overlap_count": int(len(subject_overlap)),
                "stay_overlap_count": int(len(stay_overlap)),
                "status": "pass" if not subject_overlap and not stay_overlap else "fail",
            }
        )
    return pd.DataFrame(rows)


def summarize_structured_time_leakage(
    structured_df: pd.DataFrame,
    *,
    history_window_hours: int,
) -> pd.DataFrame:
    frame = structured_df.copy()
    for column in ["hour", "prediction_time", "sepsis_onset_time", "INTIME"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    rows: list[dict] = []

    if {"hour", "prediction_time"} <= set(frame.columns):
        rows.append(
            {
                "check": "structured_rows_after_prediction_time",
                "violation_count": int((frame["hour"] > frame["prediction_time"]).fillna(False).sum()),
            }
        )
        lower_bound = frame["prediction_time"] - pd.to_timedelta(int(history_window_hours), unit="h")
        rows.append(
            {
                "check": "structured_rows_before_history_window",
                "violation_count": int((frame["hour"] < lower_bound).fillna(False).sum()),
            }
        )

    if {"sepsis3_label", "prediction_time", "sepsis_onset_time"} <= set(frame.columns):
        positive_mask = pd.to_numeric(frame["sepsis3_label"], errors="coerce").fillna(0).astype(int) == 1
        onset_violation = positive_mask & (frame["prediction_time"] >= frame["sepsis_onset_time"])
        rows.append(
            {
                "check": "positive_prediction_time_not_before_onset",
                "violation_count": int(onset_violation.fillna(False).sum()),
            }
        )

    if {"prediction_time", "sepsis_onset_time", "prediction_horizon_hours", "sepsis3_label"} <= set(frame.columns):
        positive_rows = frame.loc[pd.to_numeric(frame["sepsis3_label"], errors="coerce").fillna(0).astype(int) == 1].copy()
        if not positive_rows.empty:
            actual_horizon = (
                (positive_rows["sepsis_onset_time"] - positive_rows["prediction_time"]).dt.total_seconds() / 3600.0
            )
            configured_horizon = pd.to_numeric(positive_rows["prediction_horizon_hours"], errors="coerce")
            mismatch = (actual_horizon - configured_horizon).abs() > 1e-6
            rows.append(
                {
                    "check": "positive_prediction_horizon_mismatch",
                    "violation_count": int(mismatch.fillna(False).sum()),
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["status"] = np.where(result["violation_count"] == 0, "pass", "fail")
    return result


def summarize_text_time_leakage(text_df: pd.DataFrame) -> pd.DataFrame:
    frame = text_df.copy()
    for column in ["prediction_time", "first_note_time", "last_note_time"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")

    rows: list[dict] = []
    if {"prediction_time", "first_note_time"} <= set(frame.columns):
        rows.append(
            {
                "check": "text_windows_with_first_note_after_prediction",
                "violation_count": int((frame["first_note_time"] > frame["prediction_time"]).fillna(False).sum()),
            }
        )
    if {"prediction_time", "last_note_time"} <= set(frame.columns):
        rows.append(
            {
                "check": "text_windows_with_last_note_after_prediction",
                "violation_count": int((frame["last_note_time"] > frame["prediction_time"]).fillna(False).sum()),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["status"] = np.where(result["violation_count"] == 0, "pass", "fail")
    return result


def summarize_text_keyword_hits(
    text_df: pd.DataFrame,
    *,
    keywords: Sequence[str] = DEFAULT_TEXT_KEYWORDS,
) -> pd.DataFrame:
    if "aggregated_text" not in text_df.columns:
        return pd.DataFrame(columns=["keyword", "split", "sepsis3_label", "window_count", "stay_count"])

    frame = text_df.copy()
    frame["aggregated_text_lower"] = frame["aggregated_text"].fillna("").astype(str).str.lower()

    rows: list[dict] = []
    for keyword in keywords:
        keyword_mask = frame["aggregated_text_lower"].str.contains(str(keyword).lower(), regex=False, na=False)
        subset = frame.loc[keyword_mask].copy()
        if subset.empty:
            rows.append(
                {
                    "keyword": keyword,
                    "split": "all",
                    "sepsis3_label": "all",
                    "window_count": 0,
                    "stay_count": 0,
                }
            )
            continue
        grouped = subset.groupby(["split", "sepsis3_label"], dropna=False)
        for (split_name, label), rows_df in grouped:
            rows.append(
                {
                    "keyword": keyword,
                    "split": split_name,
                    "sepsis3_label": int(label),
                    "window_count": int(len(rows_df)),
                    "stay_count": int(rows_df["ICUSTAY_ID"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(["keyword", "split", "sepsis3_label"]).reset_index(drop=True)


def find_suspicious_feature_names(
    feature_manifest_df: pd.DataFrame,
    *,
    patterns: Iterable[str] = SUSPICIOUS_FEATURE_PATTERNS,
) -> pd.DataFrame:
    if feature_manifest_df.empty or "feature_name" not in feature_manifest_df.columns:
        return pd.DataFrame(columns=["feature_name", "feature_group"])

    pattern_list = [str(pattern).lower() for pattern in patterns]
    mask = feature_manifest_df["feature_name"].fillna("").astype(str).str.lower().apply(
        lambda feature_name: any(pattern in feature_name for pattern in pattern_list)
    )
    return feature_manifest_df.loc[mask].copy().reset_index(drop=True)


def compute_top_feature_label_correlations(
    structured_stay_table: pd.DataFrame,
    *,
    top_n: int = 20,
    corr_threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if structured_stay_table.empty or "sepsis3_label" not in structured_stay_table.columns:
        empty = pd.DataFrame(columns=["feature_name", "correlation", "abs_correlation"])
        return empty, empty

    frame = structured_stay_table.copy()
    if "split" in frame.columns:
        train_mask = frame["split"].astype(str) == "train"
        if train_mask.any():
            frame = frame.loc[train_mask].copy()

    numeric_columns = [
        column
        for column in frame.columns
        if column not in FEATURE_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric_columns:
        empty = pd.DataFrame(columns=["feature_name", "correlation", "abs_correlation"])
        return empty, empty

    correlations = []
    label = pd.to_numeric(frame["sepsis3_label"], errors="coerce")
    for column in numeric_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        corr = series.corr(label)
        if pd.isna(corr):
            continue
        correlations.append(
            {
                "feature_name": column,
                "correlation": float(corr),
                "abs_correlation": float(abs(corr)),
            }
        )

    correlation_df = pd.DataFrame(correlations).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    suspicious_df = correlation_df.loc[correlation_df["abs_correlation"] >= float(corr_threshold)].copy().reset_index(drop=True)
    return correlation_df.head(int(top_n)).copy(), suspicious_df


def _split_feature_table(feature_table: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    result: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for split_name in ["train", "val", "test"]:
        split_df = feature_table.loc[feature_table["split"] == split_name].copy()
        if split_df.empty:
            result[split_name] = (pd.DataFrame(), pd.Series(dtype=int))
            continue
        y = split_df["sepsis3_label"].astype(int)
        drop_columns = [column for column in FEATURE_METADATA_COLUMNS if column in split_df.columns]
        X = split_df.drop(columns=drop_columns)
        result[split_name] = (X, y)
    return result


def run_simple_model_sanity(
    structured_stay_table: pd.DataFrame,
    config: dict,
    *,
    model_name: str = "logistic_regression",
    threshold: float = 0.5,
) -> pd.DataFrame:
    models = build_baseline_models(config)
    if model_name not in models:
        return pd.DataFrame(columns=["model_name", "split", "auroc", "auprc", "accuracy", "precision", "recall", "f1"])

    splits = _split_feature_table(structured_stay_table)
    train_X, train_y = splits["train"]
    test_X, test_y = splits["test"]
    if train_X.empty or test_X.empty:
        return pd.DataFrame(columns=["model_name", "split", "auroc", "auprc", "accuracy", "precision", "recall", "f1"])

    feature_columns = list(train_X.columns)
    model = clone(models[model_name])
    model.fit(train_X[feature_columns], train_y)

    rows: list[dict] = []
    for split_name, eval_X, eval_y in [("train", train_X, train_y), ("test", test_X, test_y)]:
        probabilities = model.predict_proba(eval_X[feature_columns])[:, 1]
        metrics = compute_binary_classification_metrics(eval_y, probabilities, threshold=float(threshold))
        rows.append({"model_name": model_name, "split": split_name, **metrics})
    return pd.DataFrame(rows)


def run_shuffle_label_test(
    structured_stay_table: pd.DataFrame,
    config: dict,
    *,
    model_name: str = "xgboost",
    random_state: int = 42,
) -> pd.DataFrame:
    models = build_baseline_models(config)
    if model_name not in models:
        return pd.DataFrame(columns=["model_name", "test_auroc_with_shuffled_train_labels", "test_auprc_with_shuffled_train_labels"])

    splits = _split_feature_table(structured_stay_table)
    train_X, train_y = splits["train"]
    test_X, test_y = splits["test"]
    if train_X.empty or test_X.empty:
        return pd.DataFrame(columns=["model_name", "test_auroc_with_shuffled_train_labels", "test_auprc_with_shuffled_train_labels"])

    feature_columns = list(train_X.columns)
    rng = np.random.default_rng(int(random_state))
    shuffled_y = pd.Series(rng.permutation(train_y.to_numpy()), index=train_y.index)

    model = clone(models[model_name])
    model.fit(train_X[feature_columns], shuffled_y)
    probabilities = model.predict_proba(test_X[feature_columns])[:, 1]
    auroc = float("nan") if len(np.unique(test_y)) < 2 else float(roc_auc_score(test_y, probabilities))
    auprc = float("nan") if len(np.unique(test_y)) < 2 else float(average_precision_score(test_y, probabilities))

    return pd.DataFrame(
        [
            {
                "model_name": model_name,
                "test_auroc_with_shuffled_train_labels": auroc,
                "test_auprc_with_shuffled_train_labels": auprc,
            }
        ]
    )
