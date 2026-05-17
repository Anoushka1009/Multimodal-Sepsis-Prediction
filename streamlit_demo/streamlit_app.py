from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import streamlit as st
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.aligned_transformer_xgboost import AlignedTransformerXGBoostEncoder
from src.models.baselines import ID_COLUMNS
from src.training.aligned_transformer_xgboost import _build_loader, _extract_aligned_embeddings
from src.training.multimodal import prepare_multimodal_dataset, resolve_device
from src.training.tabular_multimodal import (
    FEATURE_METADATA_COLUMNS,
    build_clinical_event_feature_table,
    build_note_feature_table,
    build_structured_augmented_tabular_dataset,
)
from src.utils.runtime import load_project_runtime

ARTIFACT_DIR = PROJECT_ROOT / "results" / "processed" / "10_aligned_transformer_xgboost"
STRUCTURED_DIR = PROJECT_ROOT / "results" / "processed" / "04_feature_engineering"
TEXT_DIR = PROJECT_ROOT / "results" / "processed" / "05_text_processing"


PARAMETERS = [
    {
        "key": "temperature_c",
        "label": "Temperature",
        "unit": "deg C",
        "default": 37.0,
        "minimum": 30.0,
        "maximum": 43.0,
        "step": 0.1,
        "risk": ">= 38.0 or < 36.0",
        "weight": 1.0,
    },
    {
        "key": "heart_rate",
        "label": "Heart rate",
        "unit": "beats/min",
        "default": 88.0,
        "minimum": 30.0,
        "maximum": 220.0,
        "step": 1.0,
        "risk": "> 90",
        "weight": 1.0,
    },
    {
        "key": "respiratory_rate",
        "label": "Respiratory rate",
        "unit": "breaths/min",
        "default": 18.0,
        "minimum": 5.0,
        "maximum": 60.0,
        "step": 1.0,
        "risk": ">= 22",
        "weight": 1.2,
    },
    {
        "key": "sbp",
        "label": "Systolic BP",
        "unit": "mmHg",
        "default": 120.0,
        "minimum": 50.0,
        "maximum": 240.0,
        "step": 1.0,
        "risk": "<= 100",
        "weight": 1.5,
    },
    {
        "key": "map",
        "label": "Mean arterial pressure",
        "unit": "mmHg",
        "default": 75.0,
        "minimum": 30.0,
        "maximum": 160.0,
        "step": 1.0,
        "risk": "< 65",
        "weight": 1.5,
    },
    {
        "key": "spo2",
        "label": "SpO2",
        "unit": "%",
        "default": 96.0,
        "minimum": 50.0,
        "maximum": 100.0,
        "step": 1.0,
        "risk": "< 92",
        "weight": 1.0,
    },
    {
        "key": "wbc",
        "label": "White blood cells",
        "unit": "10^9/L",
        "default": 8.0,
        "minimum": 0.1,
        "maximum": 60.0,
        "step": 0.1,
        "risk": "> 12 or < 4",
        "weight": 1.0,
    },
    {
        "key": "lactate",
        "label": "Lactate",
        "unit": "mmol/L",
        "default": 1.2,
        "minimum": 0.1,
        "maximum": 20.0,
        "step": 0.1,
        "risk": ">= 2; >= 4 is high risk",
        "weight": 1.8,
    },
    {
        "key": "platelet",
        "label": "Platelets",
        "unit": "10^9/L",
        "default": 220.0,
        "minimum": 1.0,
        "maximum": 800.0,
        "step": 1.0,
        "risk": "< 100",
        "weight": 1.1,
    },
    {
        "key": "creatinine",
        "label": "Creatinine",
        "unit": "mg/dL",
        "default": 1.0,
        "minimum": 0.1,
        "maximum": 12.0,
        "step": 0.1,
        "risk": ">= 2.0",
        "weight": 1.0,
    },
    {
        "key": "bilirubin",
        "label": "Bilirubin",
        "unit": "mg/dL",
        "default": 0.8,
        "minimum": 0.0,
        "maximum": 30.0,
        "step": 0.1,
        "risk": ">= 2.0",
        "weight": 0.8,
    },
    {
        "key": "gcs_total",
        "label": "GCS total",
        "unit": "3-15",
        "default": 15.0,
        "minimum": 3.0,
        "maximum": 15.0,
        "step": 1.0,
        "risk": "< 15",
        "weight": 1.1,
    },
    {
        "key": "urine_output_ml_kg_hr",
        "label": "Urine output",
        "unit": "mL/kg/hr",
        "default": 0.8,
        "minimum": 0.0,
        "maximum": 5.0,
        "step": 0.1,
        "risk": "< 0.5",
        "weight": 1.0,
    },
]


def available_horizons() -> list[int]:
    horizons = []
    for path in sorted(ARTIFACT_DIR.glob("horizon_*h_aligned_transformer_xgboost_results.csv")):
        token = path.name.split("_")[1]
        try:
            horizons.append(int(token.removesuffix("h")))
        except ValueError:
            continue
    return horizons or [6, 12, 24]


@st.cache_data(show_spinner=False)
def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def artifact_paths(horizon: int) -> dict[str, Path]:
    prefix = f"horizon_{horizon}h_aligned_transformer_xgboost"
    return {
        "results": ARTIFACT_DIR / f"{prefix}_results.csv",
        "manifest": ARTIFACT_DIR / f"{prefix}_feature_manifest.csv",
        "joblib": ARTIFACT_DIR / f"{prefix}_model.joblib",
        "pkl": ARTIFACT_DIR / f"{prefix}_model.pkl",
    }


def get_threshold(horizon: int) -> float:
    results = read_csv_if_exists(artifact_paths(horizon)["results"])
    if results.empty or "decision_threshold" not in results.columns:
        return 0.5
    test_rows = results.loc[results.get("split", "") == "test"]
    row = test_rows.iloc[0] if not test_rows.empty else results.iloc[0]
    return float(row["decision_threshold"])


def abnormal_findings(values: Dict[str, float]) -> list[dict]:
    checks = {
        "temperature_c": lambda v: v >= 38.0 or v < 36.0,
        "heart_rate": lambda v: v > 90.0,
        "respiratory_rate": lambda v: v >= 22.0,
        "sbp": lambda v: v <= 100.0,
        "map": lambda v: v < 65.0,
        "spo2": lambda v: v < 92.0,
        "wbc": lambda v: v > 12.0 or v < 4.0,
        "lactate": lambda v: v >= 2.0,
        "platelet": lambda v: v < 100.0,
        "creatinine": lambda v: v >= 2.0,
        "bilirubin": lambda v: v >= 2.0,
        "gcs_total": lambda v: v < 15.0,
        "urine_output_ml_kg_hr": lambda v: v < 0.5,
    }
    rows = []
    for spec in PARAMETERS:
        value = float(values[spec["key"]])
        is_abnormal = checks[spec["key"]](value)
        if is_abnormal:
            rows.append(
                {
                    "Parameter": spec["label"],
                    "Value": f"{value:g} {spec['unit']}",
                    "Risk range": spec["risk"],
                    "Weight": spec["weight"],
                }
            )
    return rows


def clinical_demo_probability(values: Dict[str, float], suspected_infection: bool, vasopressor: bool) -> float:
    findings = abnormal_findings(values)
    score = sum(float(row["Weight"]) for row in findings)
    if values["lactate"] >= 4.0:
        score += 1.2
    if suspected_infection:
        score += 1.5
    if vasopressor:
        score += 1.3
    probability = 1.0 / (1.0 + math.exp(-(score - 4.0)))
    return float(np.clip(probability, 0.01, 0.99))


def feature_manifest_stats(horizon: int) -> pd.DataFrame:
    manifest = read_csv_if_exists(artifact_paths(horizon)["manifest"])
    if manifest.empty:
        return pd.DataFrame()
    return (
        manifest.groupby("feature_group", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("feature_group")
    )


def maybe_load_serialized_model(horizon: int):
    paths = artifact_paths(horizon)
    for model_path in (paths["joblib"], paths["pkl"]):
        if not model_path.exists():
            continue
        try:
            import joblib

            return joblib.load(model_path), model_path
        except Exception as exc:
            st.warning(f"Found {model_path.name}, but could not load it: {exc}")
            return None, model_path
    return None, paths["joblib"]


def checkpoint_path(horizon: int) -> Path:
    return ARTIFACT_DIR / f"horizon_{horizon}h_aligned_transformer_encoder_best.pt"


def load_horizon_inputs(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_name = f"horizon_{horizon}h"
    structured_path = STRUCTURED_DIR / f"{dataset_name}.csv"
    text_path = TEXT_DIR / f"{dataset_name}_note_windows.csv"
    if not structured_path.exists():
        raise FileNotFoundError(f"Missing structured horizon dataset: {structured_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"Missing text horizon dataset: {text_path}")

    structured_df = pd.read_csv(
        structured_path,
        parse_dates=["hour", "prediction_time", "INTIME", "OUTTIME"],
    )
    text_df = pd.read_csv(
        text_path,
        parse_dates=["prediction_time", "first_note_time", "last_note_time"],
    )
    return structured_df, text_df


def build_encoder_from_checkpoint(checkpoint: dict, prepared, config: dict, device: torch.device):
    hybrid_cfg = config.get("aligned_transformer_xgboost", {})
    model = AlignedTransformerXGBoostEncoder(
        structured_input_dim=len(prepared.feature_columns),
        text_embedding_dim=prepared.text_embedding_dim,
        structured_summary_dim=int(prepared.structured_summary_features.shape[1]),
        hidden_dim=int(hybrid_cfg.get("hidden_dim", config["multimodal"].get("hidden_dim", 128))),
        aligned_dim=int(checkpoint.get("aligned_dim", hybrid_cfg.get("aligned_dim", 256))),
        dropout=float(hybrid_cfg.get("dropout", config["multimodal"].get("dropout", 0.2))),
        text_encoder_mode=str(checkpoint.get("text_input_mode", hybrid_cfg.get("text_encoder_mode", "frozen_embedding"))),
        text_model_name=config["text_processing"].get("pretrained_text_model_name"),
        text_local_files_only=bool(config["text_processing"].get("local_files_only", False)),
        text_finetune_unfrozen_layers=int(
            hybrid_cfg.get(
                "text_finetune_unfrozen_layers",
                config["multimodal"].get("text_finetune_unfrozen_layers", 2),
            )
        ),
        text_gradient_checkpointing=bool(
            hybrid_cfg.get(
                "text_gradient_checkpointing",
                config["multimodal"].get("text_gradient_checkpointing", False),
            )
        ),
        structured_num_heads=int(hybrid_cfg.get("structured_num_heads", config["multimodal"].get("structured_num_heads", 4))),
        structured_num_layers=int(hybrid_cfg.get("structured_num_layers", config["multimodal"].get("structured_num_layers", 2))),
        text_num_heads=int(hybrid_cfg.get("text_num_heads", config["multimodal"].get("text_num_heads", 4))),
        text_num_layers=int(hybrid_cfg.get("text_num_layers", config["multimodal"].get("text_num_layers", 1))),
        fusion_num_heads=int(hybrid_cfg.get("fusion_num_heads", config["multimodal"].get("fusion_num_heads", 4))),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def with_local_hashing_fallback(config: dict) -> dict:
    patched = dict(config)
    patched["text_processing"] = dict(config.get("text_processing", {}))
    patched["text_processing"]["embedding_backend"] = "hashing"
    patched["text_processing"]["local_files_only"] = True
    patched["multimodal"] = dict(config.get("multimodal", {}))
    patched["multimodal"]["device"] = "cpu"
    patched["aligned_transformer_xgboost"] = dict(config.get("aligned_transformer_xgboost", {}))
    patched["aligned_transformer_xgboost"]["device"] = "cpu"
    return patched


@st.cache_resource(show_spinner=False)
def build_processed_stay_inference(horizon: int):
    bundle, model_path = maybe_load_serialized_model(horizon)
    if bundle is None:
        return None
    if isinstance(bundle, dict):
        estimator = bundle.get("model", bundle)
        feature_columns = list(bundle.get("feature_columns", []))
    else:
        estimator = bundle
        feature_columns = []

    encoder_path = checkpoint_path(horizon)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing aligned-transformer checkpoint: {encoder_path}")

    checkpoint = torch.load(encoder_path, map_location="cpu")
    runtime = load_project_runtime(mount_colab_drive=False)
    config = checkpoint.get("config", runtime.config)
    hybrid_cfg = config.get("aligned_transformer_xgboost", {})
    device = resolve_device("cpu")

    structured_df, text_df = load_horizon_inputs(horizon)
    used_hashing_fallback = False
    try:
        prepared = prepare_multimodal_dataset(
            structured_df=structured_df,
            text_df=text_df,
            config=config,
            device=device,
        )
        inference_config = config
    except Exception as exc:
        message = str(exc)
        if "huggingface.co" not in message and "offline mode" not in message and "cached files" not in message:
            raise
        inference_config = with_local_hashing_fallback(config)
        prepared = prepare_multimodal_dataset(
            structured_df=structured_df,
            text_df=text_df,
            config=inference_config,
            device=device,
        )
        used_hashing_fallback = True

    encoder = build_encoder_from_checkpoint(checkpoint, prepared, inference_config, device)
    indices = np.arange(len(prepared.labels))
    batch_size = int(hybrid_cfg.get("batch_size", config["multimodal"].get("batch_size", 8)))
    loader = _build_loader(prepared, indices, batch_size=batch_size, shuffle=False)
    aligned_embeddings = _extract_aligned_embeddings(encoder, loader, device=device)
    aligned_columns = [f"aligned_embedding_{index:03d}" for index in range(aligned_embeddings.shape[1])]
    alignment_table = pd.concat(
        [
            prepared.stay_index[ID_COLUMNS].reset_index(drop=True),
            pd.DataFrame(aligned_embeddings, columns=aligned_columns),
        ],
        axis=1,
    )

    structured_table = build_structured_augmented_tabular_dataset(
        structured_df,
        aggregations=hybrid_cfg.get(
            "structured_aggregations",
            inference_config.get("tabular_multimodal", {}).get("structured_aggregations", ["mean", "min", "max", "last"]),
        ),
        include_missingness=bool(
            hybrid_cfg.get(
                "include_missingness",
                inference_config.get("tabular_multimodal", {}).get("include_missingness", True),
            )
        ),
        include_static_categoricals=bool(
            hybrid_cfg.get(
                "include_static_categoricals",
                inference_config.get("tabular_multimodal", {}).get("include_static_categoricals", True),
            )
        ),
    )
    stay_index = structured_table[
        ID_COLUMNS + [column for column in ["split", "sepsis3_label", "prediction_time", "INTIME", "OUTTIME"] if column in structured_table.columns]
    ].copy()

    metadata_config = dict(inference_config)
    metadata_config["tabular_multimodal"] = dict(inference_config.get("tabular_multimodal", {}))
    metadata_config["tabular_multimodal"]["text_embedding_aggregations"] = []
    metadata_config["tabular_multimodal"]["include_note_metadata"] = bool(hybrid_cfg.get("include_note_metadata", True))
    note_table, note_backend = build_note_feature_table(
        text_df,
        stay_index,
        config=metadata_config,
        device=device,
    )

    event_config = dict(inference_config)
    event_config["tabular_multimodal"] = dict(inference_config.get("tabular_multimodal", {}))
    event_config["tabular_multimodal"]["include_clinical_event_features"] = bool(
        hybrid_cfg.get("include_clinical_event_features", True)
    )
    event_config["tabular_multimodal"]["clinical_event_lookback_hours"] = int(
        hybrid_cfg.get(
            "clinical_event_lookback_hours",
            inference_config.get("tabular_multimodal", {}).get(
                "clinical_event_lookback_hours",
                inference_config["feature_engineering"].get("history_window_hours", 48),
            ),
        )
    )
    event_table = build_clinical_event_feature_table(
        stay_index,
        structured_df,
        config=event_config,
        extracted_dir=runtime.paths["extracted_data_dir"],
    )

    feature_table = (
        structured_table.merge(note_table, on=ID_COLUMNS, how="left")
        .merge(event_table, on=ID_COLUMNS, how="left")
        .merge(alignment_table, on=ID_COLUMNS, how="inner")
    )

    if not feature_columns:
        raise ValueError(f"Serialized model at {model_path} does not include feature_columns.")

    drop_columns = [column for column in FEATURE_METADATA_COLUMNS if column in feature_table.columns]
    model_inputs = feature_table.drop(columns=drop_columns)
    for column in feature_columns:
        if column not in model_inputs.columns:
            model_inputs[column] = np.nan
    model_inputs = model_inputs[feature_columns]

    probabilities = estimator.predict_proba(model_inputs)[:, 1]
    threshold = get_threshold(horizon)
    result_frame = feature_table[ID_COLUMNS].copy()
    for column in ["split", "sepsis3_label", "prediction_time", "INTIME", "OUTTIME"]:
        if column in feature_table.columns:
            result_frame[column] = feature_table[column]
    result_frame["y_prob"] = probabilities
    result_frame["y_pred"] = (result_frame["y_prob"] >= threshold).astype(int)
    result_frame["decision_threshold"] = threshold

    feature_groups = {
        "aligned_embedding": [column for column in feature_columns if column.startswith("aligned_embedding_")],
        "structured": [
            column
            for column in feature_columns
            if column not in set(
                [c for c in feature_columns if c.startswith("aligned_embedding_")]
                + [c for c in feature_columns if c.startswith("note_") or c.startswith("category_window_") or c.startswith("category_note_")]
                + [c for c in feature_columns if c.startswith("antibiotic_") or c.startswith("culture_") or c.startswith("vasopressor_")]
            )
        ],
        "note_metadata": [
            column for column in feature_columns if column.startswith("note_") or column.startswith("category_window_") or column.startswith("category_note_")
        ],
        "clinical_event": [
            column for column in feature_columns if column.startswith("antibiotic_") or column.startswith("culture_") or column.startswith("vasopressor_")
        ],
    }

    return {
        "results": result_frame.sort_values(["split", "y_prob"], ascending=[True, False]).reset_index(drop=True),
        "feature_table": feature_table,
        "model_inputs": model_inputs,
        "feature_columns": feature_columns,
        "feature_groups": feature_groups,
        "model_path": model_path,
        "checkpoint_path": encoder_path,
        "text_backend": note_backend,
        "used_hashing_fallback": used_hashing_fallback,
    }


def display_processed_stay_inference(horizon: int) -> None:
    try:
        inference_bundle = build_processed_stay_inference(horizon)
    except Exception as exc:
        st.error(f"Processed-stay inference could not be initialized: {exc}")
        return
    if inference_bundle is None:
        st.info("Full processed-stay inference is unavailable until the serialized combined model artifact is present.")
        return

    results = inference_bundle["results"]
    feature_table = inference_bundle["feature_table"]
    feature_groups = inference_bundle["feature_groups"]

    st.subheader("Processed Stay Inference")
    st.caption(
        "This mode reconstructs the saved hybrid feature pipeline for existing processed stays and scores them with the serialized aligned Transformer + XGBoost model."
    )

    split_options = sorted(results["split"].dropna().astype(str).unique().tolist())
    selected_split = st.selectbox("Dataset split", split_options, key=f"processed_split_{horizon}")
    split_rows = results.loc[results["split"].astype(str) == selected_split].copy()
    if split_rows.empty:
        st.warning(f"No rows available for split `{selected_split}`.")
        return

    split_rows["patient_label"] = split_rows.apply(
        lambda row: f"ICUSTAY {int(row['ICUSTAY_ID'])} | HADM {int(row['HADM_ID'])} | SUBJECT {int(row['SUBJECT_ID'])}",
        axis=1,
    )
    selected_label = st.selectbox(
        "Processed stay",
        split_rows["patient_label"].tolist(),
        key=f"processed_patient_{horizon}_{selected_split}",
    )
    selected_row = split_rows.loc[split_rows["patient_label"] == selected_label].iloc[0]

    patient_mask = np.logical_and.reduce(
        [feature_table[column] == selected_row[column] for column in ID_COLUMNS]
    )
    patient_features = feature_table.loc[patient_mask].iloc[0]

    left, right = st.columns([1.1, 1.0], gap="large")
    with left:
        st.metric("Predicted sepsis probability", f"{float(selected_row['y_prob']):.1%}")
        st.metric("Decision threshold", f"{float(selected_row['decision_threshold']):.3f}")
        if int(selected_row["y_pred"]) == 1:
            st.error("Model prediction: sepsis risk present")
        else:
            st.success("Model prediction: sepsis risk not present")

        truth_label = "positive" if int(selected_row.get("sepsis3_label", 0)) == 1 else "negative"
        st.write(f"Ground-truth label: `{truth_label}`")
        if "prediction_time" in patient_features and pd.notna(patient_features["prediction_time"]):
            st.write(f"Prediction time: `{pd.to_datetime(patient_features['prediction_time'])}`")

    with right:
        st.write("Artifact status")
        st.write(f"Model artifact: `{inference_bundle['model_path'].relative_to(PROJECT_ROOT)}`")
        st.write(f"Encoder checkpoint: `{inference_bundle['checkpoint_path'].relative_to(PROJECT_ROOT)}`")
        st.write(f"Text feature backend: `{inference_bundle['text_backend']}`")
        if inference_bundle.get("used_hashing_fallback"):
            st.warning(
                "Transformer text files were not available locally, so the app rebuilt text features with the offline hashing backend. "
                "Predictions will run, but they will not exactly match a Bio_ClinicalBERT-based run."
            )

    summary_rows = []
    for group_name, columns in feature_groups.items():
        summary_rows.append({"Feature group": group_name, "Count": len(columns)})
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    with st.expander("Selected stay features", expanded=False):
        for group_name, columns in feature_groups.items():
            if not columns:
                continue
            preview_columns = columns[:20]
            preview_frame = pd.DataFrame(
                {
                    "feature_name": preview_columns,
                    "value": [patient_features.get(column, np.nan) for column in preview_columns],
                }
            )
            st.write(group_name.replace("_", " ").title())
            st.dataframe(preview_frame, hide_index=True, use_container_width=True)
            if len(columns) > len(preview_columns):
                st.caption(f"Showing first {len(preview_columns)} of {len(columns)} features in `{group_name}`.")


def parameter_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Parameter": spec["label"],
                "Typical input range": f"{spec['minimum']:g}-{spec['maximum']:g} {spec['unit']}",
                "Sepsis-concerning range": spec["risk"],
            }
            for spec in PARAMETERS
        ]
    )


def render_inputs(specs: Iterable[dict]) -> Dict[str, float]:
    values = {}
    columns = st.columns(2)
    for index, spec in enumerate(specs):
        with columns[index % 2]:
            values[spec["key"]] = st.number_input(
                f"{spec['label']} ({spec['unit']})",
                min_value=float(spec["minimum"]),
                max_value=float(spec["maximum"]),
                value=float(spec["default"]),
                step=float(spec["step"]),
            )
    return values


st.set_page_config(page_title="Sepsis Demo", layout="wide")
st.title("Early Sepsis Prediction Demo")

with st.sidebar:
    st.header("Model")
    horizon = st.selectbox("Prediction horizon", available_horizons(), index=1 if 12 in available_horizons() else 0)
    selected_model = st.selectbox("Model", ["Aligned Transformer + XGBoost"])
    threshold = get_threshold(int(horizon))
    st.metric("Decision threshold", f"{threshold:.3f}")

paths = artifact_paths(int(horizon))
model, expected_model_path = maybe_load_serialized_model(int(horizon))

left, right = st.columns([1.3, 1.0], gap="large")

with left:
    st.subheader("Critical Parameters Demo")
    values = render_inputs(PARAMETERS)
    suspected_infection = st.checkbox("Suspected infection, culture order, or antibiotic order", value=True)
    vasopressor = st.checkbox("Vasopressor support", value=False)

with right:
    st.subheader("Manual Demo Result")
    probability = clinical_demo_probability(values, suspected_infection, vasopressor)
    prediction = int(probability >= threshold)
    st.metric("Estimated sepsis probability", f"{probability:.1%}")
    if prediction:
        st.error("Demo output: sepsis risk present")
    else:
        st.success("Demo output: sepsis risk not present")

    findings = abnormal_findings(values)
    if findings:
        st.dataframe(pd.DataFrame(findings).drop(columns=["Weight"]), hide_index=True, use_container_width=True)
    else:
        st.write("No critical threshold crossings from the selected values.")

st.subheader("Critical Parameter Ranges")
st.dataframe(parameter_table(), hide_index=True, use_container_width=True)

with st.expander("Combined model artifact status", expanded=False):
    stats = feature_manifest_stats(int(horizon))
    if stats.empty:
        st.write("No feature manifest found for this horizon.")
    else:
        st.dataframe(stats, hide_index=True, use_container_width=True)
    st.write(f"Preferred serialized model path: `{expected_model_path.relative_to(PROJECT_ROOT)}`")
    legacy_pkl_path = paths["pkl"].relative_to(PROJECT_ROOT)
    if legacy_pkl_path != expected_model_path.relative_to(PROJECT_ROOT):
        st.write(f"Legacy fallback path: `{legacy_pkl_path}`")
    st.caption(
        "The research model was trained on time-series structured data, note-derived aligned embeddings, "
        "clinical-event features, and note metadata. A few manual values can support a demo, but they are "
        "not enough to exactly reproduce the full multimodal inference path."
    )

st.caption("For education/project demonstration only. This is not a diagnostic device or clinical decision support system.")
