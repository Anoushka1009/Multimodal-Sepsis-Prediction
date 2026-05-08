# Streamlit Sepsis Demo

This demo gives a simple UI for entering critical vitals/labs and selecting the `Aligned Transformer + XGBoost` model horizon.

## Run

```bash
pip install -r requirements.txt
streamlit run streamlit_demo/streamlit_app.py
```

## Current Artifact Status

The repository contains aligned-transformer checkpoints, prediction CSVs, summaries, and feature manifests under:

```text
results/processed/10_aligned_transformer_xgboost/
```

The final XGBoost estimator was not serialized by the original training function, so this checkout cannot perform exact combined-model inference from the saved artifacts alone. The app therefore runs a clinical-threshold demo score and clearly marks that mode in the UI.

The training code has been updated to save future combined model artifacts as:

```text
results/processed/10_aligned_transformer_xgboost/horizon_6h_aligned_transformer_xgboost_model.joblib
results/processed/10_aligned_transformer_xgboost/horizon_12h_aligned_transformer_xgboost_model.joblib
results/processed/10_aligned_transformer_xgboost/horizon_24h_aligned_transformer_xgboost_model.joblib
```

## Critical Parameters

These ranges are useful for a project demo because they are commonly sepsis-concerning clinical thresholds, not a diagnosis by themselves.

| Parameter | Demo input range | Sepsis-concerning range |
| --- | ---: | --- |
| Temperature | 30-43 deg C | >= 38.0 or < 36.0 |
| Heart rate | 30-220 beats/min | > 90 |
| Respiratory rate | 5-60 breaths/min | >= 22 |
| Systolic BP | 50-240 mmHg | <= 100 |
| Mean arterial pressure | 30-160 mmHg | < 65 |
| SpO2 | 50-100% | < 92 |
| White blood cells | 0.1-60 x10^9/L | > 12 or < 4 |
| Lactate | 0.1-20 mmol/L | >= 2; >= 4 is high risk |
| Platelets | 1-800 x10^9/L | < 100 |
| Creatinine | 0.1-12 mg/dL | >= 2.0 |
| Bilirubin | 0-30 mg/dL | >= 2.0 |
| GCS total | 3-15 | < 15 |
| Urine output | 0-5 mL/kg/hr | < 0.5 |

## Important Limitation

The full research model uses more than manual critical values: historical structured feature aggregates, text-window embeddings from notes, aligned transformer embeddings, note metadata, and clinical event features. For an exact demo, keep the model artifact plus the preprocessing objects generated from the same training run.
