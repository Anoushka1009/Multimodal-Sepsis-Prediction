from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "results" / "processed" / "10_aligned_transformer_xgboost"


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
    model_path = paths["joblib"] if paths["joblib"].exists() else paths["pkl"]
    if not model_path.exists():
        return None, model_path
    try:
        import joblib

        return joblib.load(model_path), model_path
    except Exception as exc:
        st.warning(f"Found {model_path.name}, but could not load it: {exc}")
        return None, model_path


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

if model is None:
    st.info(
        "The combined model metadata is present, but the final serialized XGBoost estimator is not in this checkout. "
        "This screen uses a clinical-threshold demo score. After retraining with the updated code, place the saved "
        f"model at `{expected_model_path.relative_to(PROJECT_ROOT)}` to enable true model-backed inference."
    )
else:
    st.success(f"Found serialized model artifact: `{expected_model_path.relative_to(PROJECT_ROOT)}`")
    st.info(
        "This manual-entry screen still uses the clinical-threshold demo score. Exact combined-model inference "
        "also needs the same preprocessing path that creates aligned embeddings and final XGBoost features."
    )

left, right = st.columns([1.3, 1.0], gap="large")

with left:
    st.subheader("Critical Parameters")
    values = render_inputs(PARAMETERS)
    suspected_infection = st.checkbox("Suspected infection, culture order, or antibiotic order", value=True)
    vasopressor = st.checkbox("Vasopressor support", value=False)

with right:
    st.subheader("Result")
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
    st.write(f"Expected serialized model: `{expected_model_path.relative_to(PROJECT_ROOT)}`")
    st.caption(
        "The research model was trained on time-series structured data, note-derived aligned embeddings, "
        "clinical-event features, and note metadata. A few manual values can support a demo, but they are "
        "not enough to exactly reproduce the full multimodal inference path."
    )

st.caption("For education/project demonstration only. This is not a diagnostic device or clinical decision support system.")
