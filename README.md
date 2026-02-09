# Multimodal-Sepsis-Prediction
Final Year Project Repository

## Project Structure
multimodal-early-sepsis/
│
├── README.md
├── requirements.txt
├── environment.yml (optional)
│
├── data/
│   ├── raw/
│   │   └── mimiciii_csv/
│   │
│   ├── processed/
│   │   ├── cohort.csv
│   │   ├── sepsis_labels_12h.csv
│   │   ├── features_hourly.parquet
│   │   ├── splits/
│   │   │   ├── train.pkl
│   │   │   ├── val.pkl
│   │   │   └── test.pkl
│   │
│   └── text/
│       ├── notes_cleaned.csv
│       └── text_embeddings.pkl
│
├── notebooks/
│   ├── 01_setup_environment.ipynb
│   ├── 02_load_mimic_to_postgres.ipynb
│   ├── 03_cohort_selection.ipynb
│   ├── 04_sepsis_labeling_12h.ipynb
│   ├── 05_hourly_feature_engineering.ipynb
│   ├── 06_dataset_exploration.ipynb
│   ├── 07_baseline_models.ipynb
│   ├── 08_transformer_timeseries.ipynb
│   ├── 09_multimodal_transformer_text.ipynb
│   └── 10_results_and_demo.ipynb
│
├── src/
│   ├── config.py
│   ├── sql/
│   │   ├── cohort.sql
│   │   ├── sofa.sql
│   │   └── infection.sql
│   │
│   ├── features/
│   │   ├── vitals.py
│   │   └── labs.py
│   │
│   ├── models/
│   │   ├── transformer_ts.py
│   │   ├── text_encoder.py
│   │   └── fusion.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   └── evaluation/
│       └── metrics.py
│
└── figures/
    ├── roc_curves/
    ├── attention_maps/
    └── demo_cases/

