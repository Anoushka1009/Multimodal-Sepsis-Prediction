# Streamlit Sepsis Demo

This demo gives two UI paths for the `Aligned Transformer + XGBoost` model:

- A manual critical-parameter demo score for simple project walkthroughs.
- A processed-stay inference mode that reconstructs the saved multimodal feature pipeline for existing stays and runs the serialized combined model end to end.

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

This checkout contains outputs from an earlier run that wrote summaries, manifests, predictions, and encoder checkpoints, but not the serialized final XGBoost estimator. Until that model artifact is regenerated, the manual critical-threshold demo works, while processed-stay inference remains unavailable.

The training code has been updated to save future combined model artifacts as:

```text
results/processed/10_aligned_transformer_xgboost/horizon_6h_aligned_transformer_xgboost_model.joblib
results/processed/10_aligned_transformer_xgboost/horizon_12h_aligned_transformer_xgboost_model.joblib
results/processed/10_aligned_transformer_xgboost/horizon_24h_aligned_transformer_xgboost_model.joblib
```

If you need to regenerate those artifacts in a network-restricted environment, you can force the offline hashing text backend and retrain a horizon locally:

```bash
python scripts/train_aligned_transformer_xgboost_local.py \
  --horizon 24 \
  --device cpu \
  --config-override configs/offline_hashing.yaml
```

That will backfill the missing `joblib` model file, but it will not exactly match the Bio_ClinicalBERT-based research run.

Once the serialized model file is present, the Streamlit app can:

- Score existing processed stays using the full saved feature pipeline.
- Show the exact prediction probability, thresholded decision, split, and ground-truth label for the selected stay.
- Keep the manual-entry demo as a separate clearly marked non-model-backed mode.

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

The full research model uses more than manual critical values: historical structured feature aggregates, text-window embeddings from notes, aligned transformer embeddings, note metadata, and clinical event features. The processed-stay inference mode reconstructs those saved inputs for existing stays, while the manual-entry panel does not.
