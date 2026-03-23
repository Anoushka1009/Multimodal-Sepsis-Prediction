# Multimodal Early Sepsis Detection with MIMIC-III

This repository is a research-grade, Colab-friendly project scaffold for early sepsis detection from multimodal ICU data in MIMIC-III. It is designed to support:

- final-year engineering project demonstrations
- reproducible machine learning experiments
- research-paper-ready figures, tables, and drafts
- modular extension from data setup through multimodal modeling and explainability

## Project Roadmap

The implementation is intentionally staged notebook-by-notebook so the pipeline stays understandable, testable, and reproducible.

### Phase 0: Foundations

1. Repository scaffold and configuration system
2. Colab + Google Drive dataset bootstrap
3. Shared utilities for paths, logging, reproducibility, and large CSV loading

### Phase 1: Dataset Access and Exploration

1. `01_dataset_setup.ipynb`
   - mount Google Drive
   - locate MIMIC-III zip file
   - unzip into project data directory
   - validate required tables
   - persist a run manifest
2. `02_data_exploration.ipynb`
   - inspect patient counts, ICU stays, note availability, missingness
   - profile table schemas and temporal coverage
   - generate initial cohort diagnostics

### Phase 2: Research Cohort Construction

1. `03_cohort_construction.ipynb`
   - define ICU cohort from `ICUSTAYS`, `ADMISSIONS`, `PATIENTS`
   - implement Sepsis-3 suspected infection logic
   - compute SOFA components
   - derive reproducible sepsis onset times
   - split train/validation/test at patient level

### Phase 3: Feature Engineering

1. `04_feature_engineering.ipynb`
   - hourly bins for vitals, labs, demographics, optional medications
   - time-window censoring to prevent leakage
   - baseline and delta features
   - temporal trajectory tensors for 6h, 12h, 24h horizons

### Phase 4: Clinical Text Pipeline

1. `05_text_processing.ipynb`
   - filter `NOTEEVENTS` by time and note category
   - tokenize and clean notes
   - build note aggregation strategies
   - generate embeddings using ClinicalBERT or related encoders

### Phase 5: Baselines

1. `06_baseline_models.ipynb`
   - Logistic Regression
   - Random Forest
   - XGBoost
   - structured-feature baselines for each horizon

### Phase 6: Multimodal Models

1. `07_multimodal_models.ipynb`
   - LSTM/GRU structured encoder
   - Transformer time-series encoder
   - text transformer encoder
   - early fusion, late fusion, gated fusion, cross-modal attention

### Phase 7: Evaluation

1. `08_evaluation.ipynb`
   - AUROC, AUPRC, precision, recall, F1
   - sensitivity, specificity
   - Brier score and calibration
   - lead-time analysis
   - confusion matrices and error slices

### Phase 8: Ablation and Explainability

1. `09_ablation_experiments.ipynb`
   - vitals only
   - vitals + labs
   - structured only
   - text only
   - full multimodal
   - fusion strategy comparison
2. `10_explainability.ipynb`
   - SHAP for structured features
   - attention visualization for notes
   - temporal feature importance analysis

### Phase 9: Paper Support

1. paper outline
2. methodology draft
3. experimental setup draft
4. results draft
5. reusable paper-ready tables and figures

## Design Decisions

- The repository is organized around reproducible stages rather than a single monolithic notebook.
- Notebook cells call code from `src/` so logic remains testable and reusable.
- Configuration lives in YAML files to support Colab, local experimentation, and future cluster execution.
- Data processing is table-selective and chunk-aware to handle large MIMIC CSV files efficiently.
- The sepsis labeling plan follows Sepsis-3 principles and will be fully documented in the cohort notebook before model training starts.
- We use patient-level splits to reduce leakage across ICU stays and time windows.

## Repository Layout

```text
multimodal-early-sepsis/
├── configs/
├── figures/
├── notebooks/
├── paper/
├── results/
├── src/
│   ├── data_processing/
│   ├── evaluation/
│   ├── fusion/
│   ├── models/
│   ├── training/
│   └── utils/
└── tests/
```

## Notebook Status

- `01_dataset_setup.ipynb`: implemented
- `02_data_exploration.ipynb`: placeholder scaffold
- `03_cohort_construction.ipynb`: placeholder scaffold
- `04_feature_engineering.ipynb`: placeholder scaffold
- `05_text_processing.ipynb`: placeholder scaffold
- `06_baseline_models.ipynb`: placeholder scaffold
- `07_multimodal_models.ipynb`: placeholder scaffold
- `08_evaluation.ipynb`: placeholder scaffold
- `09_ablation_experiments.ipynb`: placeholder scaffold
- `10_explainability.ipynb`: placeholder scaffold

## Quick Start

1. Open `notebooks/01_dataset_setup.ipynb` in Google Colab.
2. Mount Google Drive.
3. Set the zip path in the notebook or config.
4. Unzip the required MIMIC-III tables into the configured project data directory.
5. Run the validation cell to confirm table availability.

## Reproducibility Notes

- Random seeds are centralized in config.
- Intermediate artifacts should be written to `results/intermediate/`.
- Figures should be written to `figures/`.
- Every notebook should save a run manifest with config and timestamps.

## Planned Next Step

After dataset setup, the next implementation slice will be `02_data_exploration.ipynb` plus the schema-profiling utilities it depends on.
