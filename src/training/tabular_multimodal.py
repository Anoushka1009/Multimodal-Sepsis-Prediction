from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.data_processing.text_processing import apply_configured_keyword_masking
from src.data_processing.sepsis3 import (
    attach_icustay_ids,
    detect_antibiotic_orders,
    detect_culture_orders,
)
from src.evaluation.metrics import compute_binary_classification_metrics
from src.models.baselines import ID_COLUMNS, build_baseline_models
from src.training.multimodal import build_text_embedder, resolve_device


STRUCTURED_EXCLUDED_COLUMNS = set(
    ID_COLUMNS
    + [
        "hour",
        "prediction_time",
        "split",
        "sepsis3_label",
        "sepsis_onset_time",
        "prediction_horizon_hours",
        "INTIME",
        "OUTTIME",
    ]
)
FEATURE_METADATA_COLUMNS = set(ID_COLUMNS + ["split", "sepsis3_label", "prediction_time", "INTIME", "OUTTIME"])
STATIC_CATEGORICAL_COLUMNS = ["GENDER", "ETHNICITY", "FIRST_CAREUNIT", "LAST_CAREUNIT"]


def _select_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str = "f1",
    default_threshold: float = 0.5,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float(default_threshold)

    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 19, dtype=np.float32),
                y_prob.astype(np.float32),
                np.asarray([default_threshold], dtype=np.float32),
            ]
        )
    )

    best_threshold = float(default_threshold)
    best_score = float("-inf")
    for threshold in candidate_thresholds:
        metrics = compute_binary_classification_metrics(y_true, y_prob, threshold=float(threshold))
        score = float(metrics.get(metric_name, float("nan")))
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _build_stay_index(structured_df: pd.DataFrame) -> pd.DataFrame:
    columns = ID_COLUMNS + ["split", "sepsis3_label"]
    optional_columns = [column for column in ["prediction_time", "INTIME", "OUTTIME"] if column in structured_df.columns]
    stay_index = structured_df[columns + optional_columns].drop_duplicates(subset=ID_COLUMNS).copy()
    return stay_index.sort_values(ID_COLUMNS).reset_index(drop=True)


def _merge_feature_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    usable_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable_frames:
        return pd.DataFrame(columns=ID_COLUMNS)

    merged = usable_frames[0].copy()
    for frame in usable_frames[1:]:
        merged = merged.merge(frame, on=ID_COLUMNS, how="outer")
    return merged


def _split_feature_table(feature_table: pd.DataFrame) -> Dict[str, tuple[pd.DataFrame, pd.Series]]:
    splits: Dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for split_name in ["train", "val", "test"]:
        split_df = feature_table.loc[feature_table["split"] == split_name].copy()
        if split_df.empty:
            splits[split_name] = (pd.DataFrame(), pd.Series(dtype=int))
            continue
        y = split_df["sepsis3_label"].astype(int)
        drop_columns = [column for column in FEATURE_METADATA_COLUMNS if column in split_df.columns]
        X = split_df.drop(columns=drop_columns)
        splits[split_name] = (X, y)
    return splits


def _build_prediction_frame(
    ids: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    *,
    dataset_name: str,
    model_name: str,
    decision_threshold: float,
) -> pd.DataFrame:
    frame = ids.reset_index(drop=True).copy()
    frame["y_true"] = np.asarray(y_true).astype(int)
    frame["y_prob"] = np.asarray(y_prob).astype(float)
    frame["y_pred"] = (frame["y_prob"] >= float(decision_threshold)).astype(int)
    frame["decision_threshold"] = float(decision_threshold)
    frame["dataset_name"] = dataset_name
    frame["model_name"] = model_name
    return frame


def build_structured_augmented_tabular_dataset(
    horizon_df: pd.DataFrame,
    *,
    aggregations: Iterable[str] = ("mean", "min", "max", "last"),
    include_missingness: bool = True,
    include_static_categoricals: bool = True,
) -> pd.DataFrame:
    if horizon_df.empty:
        return _build_stay_index(horizon_df)

    feature_columns = [
        column
        for column in horizon_df.columns
        if column not in STRUCTURED_EXCLUDED_COLUMNS and pd.api.types.is_numeric_dtype(horizon_df[column])
    ]

    grouped = horizon_df.groupby(ID_COLUMNS, dropna=False)
    frames: list[pd.DataFrame] = []
    for agg in aggregations:
        if not feature_columns:
            continue
        if agg == "last":
            frame = grouped[feature_columns].last().reset_index()
        elif agg == "mean":
            frame = grouped[feature_columns].mean().reset_index()
        elif agg == "min":
            frame = grouped[feature_columns].min().reset_index()
        elif agg == "max":
            frame = grouped[feature_columns].max().reset_index()
        else:
            continue
        frame = frame.rename(columns={column: f"{column}__{agg}" for column in feature_columns})
        frames.append(frame)

    if include_missingness and feature_columns:
        missing = horizon_df[ID_COLUMNS + feature_columns].copy()
        missing.loc[:, feature_columns] = missing.loc[:, feature_columns].isna().astype(np.float32)
        missing = missing.groupby(ID_COLUMNS, dropna=False)[feature_columns].mean().reset_index()
        missing = missing.rename(columns={column: f"{column}__missing_rate" for column in feature_columns})
        frames.append(missing)

    if include_static_categoricals:
        categorical_columns = [column for column in STATIC_CATEGORICAL_COLUMNS if column in horizon_df.columns]
        if categorical_columns:
            static = horizon_df[ID_COLUMNS + categorical_columns].drop_duplicates(subset=ID_COLUMNS).copy()
            for column in categorical_columns:
                static[column] = static[column].fillna("UNKNOWN").astype(str)
            static_dummies = pd.get_dummies(
                static[categorical_columns],
                prefix=[f"static_{column}" for column in categorical_columns],
                dtype=np.uint8,
            )
            frames.append(pd.concat([static[ID_COLUMNS].reset_index(drop=True), static_dummies.reset_index(drop=True)], axis=1))

    merged = _merge_feature_frames(frames)
    stay_index = _build_stay_index(horizon_df)
    return stay_index.merge(merged, on=ID_COLUMNS, how="left")


def build_note_feature_table(
    text_df: pd.DataFrame,
    stay_index: pd.DataFrame,
    *,
    config: dict,
    device,
) -> tuple[pd.DataFrame, str]:
    cfg = config.get("tabular_multimodal", {})
    embedding_aggs = [str(value).lower() for value in cfg.get("text_embedding_aggregations", ["mean", "closest"])]
    include_note_metadata = bool(cfg.get("include_note_metadata", True))

    if text_df.empty:
        return stay_index[ID_COLUMNS].copy(), "none"

    rows = text_df.copy()
    rows["aggregated_text"] = rows.get("aggregated_text", "").fillna("").astype(str)
    rows["categories"] = rows.get("categories", "").fillna("").astype(str)
    rows["note_count"] = pd.to_numeric(rows.get("note_count", 0), errors="coerce").fillna(0.0)
    rows["note_window_index"] = pd.to_numeric(rows.get("note_window_index", 0), errors="coerce").fillna(0).astype(int)
    rows = apply_configured_keyword_masking(rows, config, text_column="aggregated_text")

    for column in ["prediction_time", "first_note_time", "last_note_time"]:
        if column in rows.columns:
            rows[column] = pd.to_datetime(rows[column], errors="coerce")

    rows = rows.sort_values(ID_COLUMNS + ["note_window_index"]).reset_index(drop=True)

    text_frames: list[pd.DataFrame] = []
    backend_name = "none"
    if embedding_aggs:
        text_embedder = build_text_embedder(config, device)
        backend_name = getattr(text_embedder, "backend_name", "unknown")
        encoded = text_embedder.encode_texts(
            rows["aggregated_text"].tolist(),
            batch_size=int(config["text_processing"].get("embedding_batch_size", 8)),
        )
        embedding_dim = int(encoded.shape[1]) if encoded.size else int(getattr(text_embedder, "embedding_dim", 0))
        embedding_columns = [f"embedding_{index}" for index in range(embedding_dim)]
        embedding_df = pd.concat(
            [
                rows[ID_COLUMNS + ["note_window_index"]].reset_index(drop=True),
                pd.DataFrame(encoded, columns=embedding_columns),
            ],
            axis=1,
        )

        grouped = embedding_df.groupby(ID_COLUMNS, dropna=False)
        for agg in embedding_aggs:
            if agg == "mean":
                frame = grouped[embedding_columns].mean().reset_index()
            elif agg == "max":
                frame = grouped[embedding_columns].max().reset_index()
            elif agg in {"closest", "last"}:
                frame = grouped.first().reset_index()
                frame = frame.drop(columns=["note_window_index"], errors="ignore")
            else:
                continue
            frame = frame.rename(columns={column: f"text_{agg}_{column}" for column in embedding_columns})
            text_frames.append(frame)

    if include_note_metadata:
        grouped_rows = rows.groupby(ID_COLUMNS, dropna=False)
        metadata = grouped_rows.agg(
            note_window_count=("note_window_index", "size"),
            note_total_count=("note_count", "sum"),
            note_mean_count_per_window=("note_count", "mean"),
            note_max_count_per_window=("note_count", "max"),
            note_closest_window_index=("note_window_index", "min"),
            note_oldest_window_index=("note_window_index", "max"),
        ).reset_index()

        if {"prediction_time", "last_note_time"} <= set(rows.columns):
            rows["note_hours_since_last_note"] = (
                (rows["prediction_time"] - rows["last_note_time"]).dt.total_seconds() / 3600.0
            )
            closest_recency = grouped_rows["note_hours_since_last_note"].min().reset_index(name="note_closest_recency_hours")
            metadata = metadata.merge(closest_recency, on=ID_COLUMNS, how="left")

        if {"prediction_time", "first_note_time"} <= set(rows.columns):
            rows["note_hours_since_first_note"] = (
                (rows["prediction_time"] - rows["first_note_time"]).dt.total_seconds() / 3600.0
            )
            oldest_recency = grouped_rows["note_hours_since_first_note"].max().reset_index(name="note_oldest_recency_hours")
            metadata = metadata.merge(oldest_recency, on=ID_COLUMNS, how="left")

        category_names = [str(category) for category in config["text_processing"].get("note_categories", [])]
        rows["categories_lower"] = rows["categories"].str.lower()
        for category in category_names:
            safe_name = category.lower().replace(" ", "_")
            present = rows["categories_lower"].str.contains(category.lower(), regex=False, na=False).astype(int)
            rows[f"category_window_{safe_name}"] = present
            rows[f"category_note_{safe_name}"] = rows["note_count"] * present

        category_columns = [
            column
            for column in rows.columns
            if column.startswith("category_window_") or column.startswith("category_note_")
        ]
        if category_columns:
            category_frame = grouped_rows[category_columns].sum().reset_index()
            metadata = metadata.merge(category_frame, on=ID_COLUMNS, how="left")

        text_frames.append(metadata)

    merged = stay_index[ID_COLUMNS].merge(_merge_feature_frames(text_frames), on=ID_COLUMNS, how="left")
    count_like_columns = [
        column
        for column in merged.columns
        if column.startswith("note_") or column.startswith("category_window_") or column.startswith("category_note_")
    ]
    if count_like_columns:
        merged.loc[:, count_like_columns] = merged.loc[:, count_like_columns].fillna(0.0)
    return merged, backend_name


def _attach_event_times(events: pd.DataFrame, cohort: pd.DataFrame, *, source_time_column: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=ID_COLUMNS + [source_time_column])

    attached = attach_icustay_ids(
        events.rename(columns={source_time_column: "charttime"}),
        cohort,
        time_column="charttime",
    )
    if attached.empty:
        return pd.DataFrame(columns=ID_COLUMNS + [source_time_column])
    attached = attached.rename(columns={"charttime": source_time_column})
    return attached[ID_COLUMNS + [source_time_column]].copy()


def _aggregate_event_features(
    stay_index: pd.DataFrame,
    events: pd.DataFrame,
    *,
    time_column: str,
    prefix: str,
    lookback_hours: int,
) -> pd.DataFrame:
    if events.empty:
        return stay_index[ID_COLUMNS].copy()

    joined = stay_index[ID_COLUMNS + ["prediction_time"]].merge(events, on=ID_COLUMNS, how="left")
    joined = joined.dropna(subset=[time_column]).copy()
    if joined.empty:
        return stay_index[ID_COLUMNS].copy()

    lower_bound = joined["prediction_time"] - pd.to_timedelta(int(lookback_hours), unit="h")
    joined = joined.loc[(joined[time_column] <= joined["prediction_time"]) & (joined[time_column] >= lower_bound)].copy()
    if joined.empty:
        return stay_index[ID_COLUMNS].copy()

    aggregated = joined.groupby(ID_COLUMNS, dropna=False).agg(
        event_count=(time_column, "size"),
        last_event_time=(time_column, "max"),
    ).reset_index()
    aggregated = aggregated.merge(stay_index[ID_COLUMNS + ["prediction_time"]], on=ID_COLUMNS, how="left")
    aggregated[f"{prefix}_count_{lookback_hours}h"] = aggregated["event_count"].astype(float)
    aggregated[f"{prefix}_flag_{lookback_hours}h"] = (aggregated["event_count"] > 0).astype(int)
    aggregated[f"{prefix}_hours_since_last"] = (
        (aggregated["prediction_time"] - aggregated["last_event_time"]).dt.total_seconds() / 3600.0
    )
    return aggregated[
        ID_COLUMNS
        + [
            f"{prefix}_count_{lookback_hours}h",
            f"{prefix}_flag_{lookback_hours}h",
            f"{prefix}_hours_since_last",
        ]
    ]


def build_clinical_event_feature_table(
    stay_index: pd.DataFrame,
    structured_df: pd.DataFrame,
    *,
    config: dict,
    extracted_dir: str | Path,
) -> pd.DataFrame:
    cfg = config.get("tabular_multimodal", {})
    if not bool(cfg.get("include_clinical_event_features", True)):
        return stay_index[ID_COLUMNS].copy()

    if "prediction_time" not in stay_index.columns or "INTIME" not in stay_index.columns or "OUTTIME" not in stay_index.columns:
        return stay_index[ID_COLUMNS].copy()

    cohort = structured_df[ID_COLUMNS + ["INTIME", "OUTTIME"]].drop_duplicates(subset=ID_COLUMNS).copy()
    lookback_hours = int(cfg.get("clinical_event_lookback_hours", config["feature_engineering"].get("history_window_hours", 48)))
    low_memory = bool(config["dataset"].get("low_memory", True))
    sepsis_cfg = config["sepsis3"]

    antibiotics = detect_antibiotic_orders(
        extracted_dir,
        sepsis_cfg.get("antibiotic_keywords", []),
        low_memory=low_memory,
    )
    antibiotics = _attach_event_times(
        antibiotics[["SUBJECT_ID", "HADM_ID", "antibiotic_time"]].copy() if not antibiotics.empty else antibiotics,
        cohort,
        source_time_column="antibiotic_time",
    )

    vasopressors = detect_antibiotic_orders(
        extracted_dir,
        sepsis_cfg.get("vasopressor_keywords", []),
        low_memory=low_memory,
    )
    vasopressors = _attach_event_times(
        vasopressors[["SUBJECT_ID", "HADM_ID", "antibiotic_time"]].copy() if not vasopressors.empty else vasopressors,
        cohort,
        source_time_column="antibiotic_time",
    )

    cultures = detect_culture_orders(extracted_dir, low_memory=low_memory)
    cultures = _attach_event_times(
        cultures[["SUBJECT_ID", "HADM_ID", "culture_time"]].copy() if not cultures.empty else cultures,
        cohort,
        source_time_column="culture_time",
    )

    frames = [
        _aggregate_event_features(
            stay_index,
            antibiotics,
            time_column="antibiotic_time",
            prefix="antibiotic",
            lookback_hours=lookback_hours,
        ),
        _aggregate_event_features(
            stay_index,
            vasopressors,
            time_column="antibiotic_time",
            prefix="vasopressor",
            lookback_hours=lookback_hours,
        ),
        _aggregate_event_features(
            stay_index,
            cultures,
            time_column="culture_time",
            prefix="culture",
            lookback_hours=lookback_hours,
        ),
    ]
    merged = stay_index[ID_COLUMNS].merge(_merge_feature_frames(frames), on=ID_COLUMNS, how="left")

    for column in merged.columns:
        if column in ID_COLUMNS:
            continue
        if column.endswith(f"_count_{lookback_hours}h") or column.endswith(f"_flag_{lookback_hours}h"):
            merged[column] = merged[column].fillna(0.0)
    return merged


def _fit_predict_pipeline(model, train_X: pd.DataFrame, train_y: pd.Series, eval_X: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    model.fit(train_X[list(feature_columns)], train_y)
    return model.predict_proba(eval_X[list(feature_columns)])[:, 1]


def train_tabular_multimodal_models(
    *,
    structured_df: pd.DataFrame,
    text_df: pd.DataFrame,
    config: dict,
    extracted_dir: str | Path,
    dataset_name: str,
    device=None,
) -> Dict[str, object]:
    device = resolve_device(device or config["multimodal"].get("device", "auto"))
    cfg = config.get("tabular_multimodal", {})

    structured_table = build_structured_augmented_tabular_dataset(
        structured_df,
        aggregations=cfg.get("structured_aggregations", config["baselines"].get("tabular_aggregations", ["mean", "min", "max", "last"])),
        include_missingness=bool(cfg.get("include_missingness", True)),
        include_static_categoricals=bool(cfg.get("include_static_categoricals", True)),
    )
    stay_index = structured_table[ID_COLUMNS + [column for column in ["split", "sepsis3_label", "prediction_time", "INTIME", "OUTTIME"] if column in structured_table.columns]].copy()

    text_table, text_backend = build_note_feature_table(
        text_df,
        stay_index,
        config=config,
        device=device,
    )
    event_table = build_clinical_event_feature_table(
        stay_index,
        structured_df,
        config=config,
        extracted_dir=extracted_dir,
    )

    feature_table = structured_table.merge(text_table, on=ID_COLUMNS, how="left").merge(event_table, on=ID_COLUMNS, how="left")
    splits = _split_feature_table(feature_table)
    split_ids = {
        split_name: feature_table.loc[feature_table["split"] == split_name, ID_COLUMNS].reset_index(drop=True)
        for split_name in ["train", "val", "test"]
    }
    train_X, train_y = splits["train"]
    val_X, val_y = splits["val"]
    test_X, test_y = splits["test"]
    if train_X.empty or test_X.empty:
        raise ValueError("Tabular multimodal training requires non-empty train and test splits.")

    all_feature_columns = list(train_X.columns)
    text_embedding_columns = [column for column in all_feature_columns if column.startswith("text_") and "_embedding_" in column]
    note_metadata_columns = [
        column
        for column in all_feature_columns
        if column.startswith("note_") or column.startswith("category_window_") or column.startswith("category_note_")
    ]
    event_feature_columns = [
        column
        for column in all_feature_columns
        if column.startswith("antibiotic_") or column.startswith("culture_") or column.startswith("vasopressor_")
    ]
    structured_feature_columns = [
        column
        for column in all_feature_columns
        if column not in set(text_embedding_columns + note_metadata_columns + event_feature_columns)
    ]

    augmented_feature_columns = structured_feature_columns + text_embedding_columns + note_metadata_columns + event_feature_columns
    note_model_feature_columns = text_embedding_columns + note_metadata_columns + event_feature_columns

    models = build_baseline_models(config)
    if "xgboost" not in models or "logistic_regression" not in models:
        raise ValueError("Tabular multimodal models require both xgboost and logistic_regression baselines to be available.")

    threshold_metric = str(cfg.get("threshold_selection_metric", config.get("evaluation", {}).get("threshold_selection_metric", "f1")))
    default_threshold = float(config.get("evaluation", {}).get("default_threshold", 0.5))

    artifacts: Dict[str, pd.DataFrame] = {}
    results_rows: list[dict] = []

    if "xgboost_text_augmented" in set(cfg.get("models", [])):
        augmented_model = clone(models["xgboost"])
        val_prob = _fit_predict_pipeline(augmented_model, train_X, train_y, val_X, augmented_feature_columns)
        decision_threshold = _select_decision_threshold(
            val_y.to_numpy(),
            val_prob,
            metric_name=threshold_metric,
            default_threshold=default_threshold,
        )

        for split_name, eval_X, eval_y in [("val", val_X, val_y), ("test", test_X, test_y)]:
            eval_prob = augmented_model.predict_proba(eval_X[augmented_feature_columns])[:, 1]
            metrics = compute_binary_classification_metrics(eval_y, eval_prob, threshold=decision_threshold)
            results_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split": split_name,
                    "model_name": "xgboost_text_augmented",
                    "text_embedding_backend": text_backend,
                    "decision_threshold": float(decision_threshold),
                    "n_features": int(len(augmented_feature_columns)),
                    "n_structured_features": int(len(structured_feature_columns)),
                    "n_text_embedding_features": int(len(text_embedding_columns)),
                    "n_note_metadata_features": int(len(note_metadata_columns)),
                    "n_event_features": int(len(event_feature_columns)),
                    **metrics,
                }
            )
            artifacts[f"{dataset_name}_xgboost_text_augmented_{split_name}_predictions"] = _build_prediction_frame(
                split_ids[split_name],
                eval_y,
                eval_prob,
                dataset_name=dataset_name,
                model_name="xgboost_text_augmented",
                decision_threshold=decision_threshold,
            )

    if "stacked_xgboost_notes" in set(cfg.get("models", [])) and note_model_feature_columns:
        structured_model = clone(models["xgboost"])
        note_model = clone(models[str(cfg.get("note_model", "logistic_regression"))])
        meta_model = clone(models[str(cfg.get("meta_model", "logistic_regression"))])

        structured_model.fit(train_X[structured_feature_columns], train_y)
        note_model.fit(train_X[note_model_feature_columns], train_y)

        val_stack = pd.DataFrame(
            {
                "structured_prob": structured_model.predict_proba(val_X[structured_feature_columns])[:, 1],
                "note_prob": note_model.predict_proba(val_X[note_model_feature_columns])[:, 1],
            }
        )
        test_stack = pd.DataFrame(
            {
                "structured_prob": structured_model.predict_proba(test_X[structured_feature_columns])[:, 1],
                "note_prob": note_model.predict_proba(test_X[note_model_feature_columns])[:, 1],
            }
        )
        passthrough_columns = note_metadata_columns + event_feature_columns
        for column in passthrough_columns:
            val_stack[column] = val_X[column].to_numpy()
            test_stack[column] = test_X[column].to_numpy()

        meta_model.fit(val_stack, val_y)
        test_prob = meta_model.predict_proba(test_stack)[:, 1]
        decision_threshold = float(cfg.get("stacked_decision_threshold", default_threshold))
        metrics = compute_binary_classification_metrics(test_y, test_prob, threshold=decision_threshold)
        results_rows.append(
            {
                "dataset_name": dataset_name,
                "split": "test",
                "model_name": "stacked_xgboost_notes",
                "text_embedding_backend": text_backend,
                "decision_threshold": float(decision_threshold),
                "n_features": int(test_stack.shape[1]),
                "n_structured_features": int(len(structured_feature_columns)),
                "n_text_embedding_features": int(len(text_embedding_columns)),
                "n_note_metadata_features": int(len(note_metadata_columns)),
                "n_event_features": int(len(event_feature_columns)),
                "stack_train_examples": int(len(val_y)),
                **metrics,
            }
        )
        artifacts[f"{dataset_name}_stacked_xgboost_notes_test_predictions"] = _build_prediction_frame(
            split_ids["test"],
            test_y,
            test_prob,
            dataset_name=dataset_name,
            model_name="stacked_xgboost_notes",
            decision_threshold=decision_threshold,
        )

    feature_manifest_rows = []
    for column in structured_feature_columns:
        feature_manifest_rows.append({"feature_name": column, "feature_group": "structured"})
    for column in text_embedding_columns:
        feature_manifest_rows.append({"feature_name": column, "feature_group": "text_embedding"})
    for column in note_metadata_columns:
        feature_manifest_rows.append({"feature_name": column, "feature_group": "note_metadata"})
    for column in event_feature_columns:
        feature_manifest_rows.append({"feature_name": column, "feature_group": "clinical_event"})

    artifacts[f"{dataset_name}_tabular_multimodal_feature_manifest"] = pd.DataFrame(feature_manifest_rows)
    artifacts[f"{dataset_name}_tabular_multimodal_results"] = (
        pd.DataFrame(results_rows).sort_values(["dataset_name", "split", "model_name"]).reset_index(drop=True)
        if results_rows
        else pd.DataFrame()
    )
    artifacts[f"{dataset_name}_tabular_multimodal_stay_index"] = stay_index.copy()

    summary = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "text_embedding_backend": text_backend,
                "n_train": int(len(train_y)),
                "n_val": int(len(val_y)),
                "n_test": int(len(test_y)),
                "n_structured_features": int(len(structured_feature_columns)),
                "n_text_embedding_features": int(len(text_embedding_columns)),
                "n_note_metadata_features": int(len(note_metadata_columns)),
                "n_event_features": int(len(event_feature_columns)),
                "device": str(device),
            }
        ]
    )
    artifacts[f"{dataset_name}_tabular_multimodal_summary"] = summary

    return {
        "artifacts": artifacts,
        "text_embedding_backend": text_backend,
        "device": str(device),
    }
