# Final Pipeline and Implemented Model Details

This note summarizes the final end-to-end workflow of the project, the models that were implemented, the main improvements made over the initial approach, and the major practical challenges encountered during development.

It is written to support the **Methods**, **Experimental Setup**, **Results**, and **Discussion** sections of the report.

## 1. Final Pipeline

The final workflow of the project follows a fixed six-stage pipeline:

1. Data extraction
2. Preprocessing
3. Feature engineering
4. Text processing
5. Model training
6. Evaluation and ablation

In the repository, these stages are organized through the notebooks:

| Stage | Notebook / Script | Main Output |
|---|---|---|
| Data extraction | `01_dataset_setup.ipynb` | extracted MIMIC-III tables under `data/mimiciii` |
| Data exploration | `02_data_exploration.ipynb` | schema previews and exploratory summaries |
| Cohort construction | `03_cohort_construction.ipynb` | adult ICU cohort with Sepsis-3 labels |
| Feature engineering | `04_feature_engineering.ipynb` | horizon-specific structured tables |
| Text processing | `05_text_processing.ipynb` | horizon-specific note-window tables |
| Structured baselines + tabular multimodal | `06_baseline_models.ipynb`, `train_tabular_multimodal_local.py` | baseline and XGBoost-based multimodal results |
| Deep multimodal model | `07_multimodal_models.ipynb`, `train_multimodal_local.py` | deep Transformer/GRU + BERT fusion results |
| Unified evaluation | `08_evaluation.ipynb` | `evaluation_summary.csv` and curve tables |
| Ablation study | `09_ablation_experiments.ipynb` | custom-model ablation results |
| Explainability | `10_explainability.ipynb` | SHAP and interpretability artifacts |
| Custom model visualization | `11_custom_model_visualization.ipynb` | ROC, PR, calibration, confusion, leakage checks |
| Hybrid model evaluation | `12_aligned_transformer_xgboost_evaluation.ipynb` | separate evaluation for aligned Transformer + XGBoost |
| Hybrid model ablation | `13_aligned_transformer_xgboost_ablation.ipynb` | separate ablation for aligned Transformer + XGBoost |

### 1.1 Data extraction

The project starts from a compressed MIMIC-III dataset archive. The extraction stage validates the presence of required raw tables such as:

- `PATIENTS.csv`
- `ADMISSIONS.csv`
- `ICUSTAYS.csv`
- `CHARTEVENTS.csv`
- `LABEVENTS.csv`
- `NOTEEVENTS.csv`
- `PRESCRIPTIONS.csv`
- `MICROBIOLOGYEVENTS.csv`

Optional ICU event tables such as `INPUTEVENTS_MV.csv`, `INPUTEVENTS_CV.csv`, `OUTPUTEVENTS.csv`, and `PROCEDUREEVENTS_MV.csv` are also supported when available.

This stage is intentionally lightweight: it ensures that the raw files are available locally and consistently laid out for the downstream chunked processing pipeline.

### 1.2 Preprocessing

The preprocessing stage constructs the study cohort and the prediction labels.

The main preprocessing operations are:

- merge patient, admission, and ICU stay tables
- compute age at ICU admission
- retain only adult ICU stays
- retain only stays with minimum ICU length of stay
- preserve stay-level identifiers: `SUBJECT_ID`, `HADM_ID`, `ICUSTAY_ID`
- assign patient-level data splits into `train`, `val`, and `test`
- derive Sepsis-3 suspicion and onset logic from antibiotics, cultures, and organ-dysfunction indicators

The cohort construction is patient-aware rather than row-wise. This is important because the prediction problem is stay-level and leakage can occur if the same patient contributes related observations across splits.

### 1.3 Feature engineering

The structured branch uses a 48-hour lookback window before each prediction point. Hourly bins are created and populated using chart and lab events extracted from the large MIMIC event tables.

The engineered structured features include:

- vital signs: heart rate, systolic blood pressure, diastolic blood pressure, mean arterial pressure, respiratory rate, temperature, oxygen saturation, bedside glucose
- additional clinically important chart features: `FiO2`, `GCS total`, `urine output`
- laboratory features: white blood cell count, hemoglobin, creatinine, platelet count, bilirubin, lactate, sodium, potassium, bicarbonate, blood urea nitrogen

For each hourly step, the project computes the configured summary statistics:

- `mean`
- `min`
- `max`
- `last`

This produces the structured hourly representation used by the deep multimodal models, and also the stay-level aggregated representation used by the tabular baselines and the XGBoost-based multimodal models.

Three prediction horizons are supported:

- `6h`
- `12h`
- `24h`

Each horizon gets its own processed structured table:

- `horizon_6h.csv`
- `horizon_12h.csv`
- `horizon_24h.csv`

### 1.4 Text processing

Clinical note processing is performed separately from structured feature engineering.

The text pipeline:

- filters note categories to `Physician`, `Nursing`, and `Radiology`
- removes notes outside the valid stay context
- censors notes at the prediction time to avoid future leakage
- groups notes into 6-hour windows
- concatenates all notes within the same window
- cleans whitespace and trims text length
- optionally masks explicit sepsis-related keywords such as `sepsis`, `septic`, and `septic shock`

The keyword-masking step was introduced as a leakage-sensitivity control. Importantly, the stored note-window CSVs remain audit-friendly, while masking is applied to the model input text during training and inference.

The processed note-window outputs are saved separately for each horizon:

- `horizon_6h_note_windows.csv`
- `horizon_12h_note_windows.csv`
- `horizon_24h_note_windows.csv`

### 1.5 Model training

The project finally trains multiple model families rather than a single classifier. This is an important part of the final methodology, because the implementation evolved from simpler structured baselines to stronger multimodal systems.

The implemented model families are:

1. Structured baselines
2. Deep multimodal fusion model
3. Tabular multimodal XGBoost model
4. Aligned Transformer + XGBoost hybrid model

These are described in detail in Section 2 below.

### 1.6 Evaluation and ablation

Evaluation is performed on saved prediction files rather than only in-memory results. This allows:

- horizon-wise comparison
- ROC and PR curve generation
- calibration analysis
- confusion matrices
- threshold-aware metrics
- overfitting checks using validation vs test predictions
- leakage audit cells in the custom visualization notebook

Two parallel evaluation paths are used:

- the standard evaluation path for structured baselines, deep multimodal models, and tabular multimodal models
- a separate evaluation path for the aligned Transformer + XGBoost hybrid model so that its artifacts do not overwrite the earlier results

Similarly, ablation studies are run in separate stages for:

- the XGBoost-based tabular multimodal model
- the aligned Transformer + XGBoost hybrid model

## 2. Implemented Models

The final project contains four main model families.

### 2.1 Structured baseline models

These models operate only on stay-level structured tabular features.

Implemented baseline models:

- `logistic_regression`
- `random_forest`
- `xgboost`

#### Input

The structured hourly features are converted into a stay-level table using:

- `mean`
- `min`
- `max`
- `last`

over the 48-hour history window.

#### Model role

These baselines serve two purposes:

- provide a strong structured-only benchmark
- establish whether multimodal fusion actually improves over a well-engineered tabular baseline

#### Implementation details

- Logistic regression uses median imputation, standardization, and class balancing.
- Random forest uses class-balanced subsampling and tree-based nonlinear decision boundaries.
- XGBoost uses boosted decision trees with tuned depth, learning rate, subsampling, and column subsampling.

#### Practical finding

These structured baselines, especially XGBoost and random forest, were already very strong, which later influenced the design of the final multimodal systems.

### 2.2 Deep multimodal custom model

This was the first major custom multimodal architecture.

Implemented variants:

- `early_fusion`
- `late_fusion`
- `gated_fusion`
- `cross_modal_attention`

#### Input representation

- Structured branch: `48 x 75` hourly numeric tensor
- Text branch: up to `8` note windows, represented either by frozen BioClinicalBERT embeddings or by a fine-tuned BERT token path depending on the experiment configuration

#### Structured branch

The structured branch supports:

- `GRU`
- `Transformer`

In the later experiments, the default structured encoder was changed to a Transformer encoder with masking and temporal pooling support.

#### Text branch

The text branch is based on BioClinicalBERT:

- pretrained backbone: `emilyalsentzer/Bio_ClinicalBERT`
- note windows grouped in 6-hour bins
- either frozen embeddings or a lightweight fine-tuning mode with limited unfrozen layers

#### Fusion strategies

- `early_fusion`: concatenate modality representations before classification
- `late_fusion`: combine modality-specific logits
- `gated_fusion`: learn a gate that adaptively mixes structured and text features
- `cross_modal_attention`: use attention between structured and text representations

#### Why it was important

This model represented the original neural multimodal idea of the project:

- learn temporal patterns from vitals and labs
- learn semantic patterns from notes
- fuse both modalities in a trainable end-to-end network

#### Practical finding

Although conceptually strong, the deep multimodal fusion models did not outperform the strongest tree-based multimodal systems on this dataset. Their AUROC and AUPRC were consistently lower than the final XGBoost-based multimodal models, especially at the longer prediction horizons.

This made the deep multimodal model more useful as:

- an architectural comparison model
- a fusion-strategy ablation model
- an alignment feature extractor for the later hybrid model

The detailed architecture is documented separately in:

- `paper/custom_model_architecture.md`

### 2.3 Tabular multimodal XGBoost custom model

This became the strongest practical custom model family for the project.

Implemented variants:

- `xgboost_text_augmented`
- `stacked_xgboost_notes`

#### Core idea

Instead of asking a neural fusion model to learn everything from raw sequences, this design converts each modality into strong stay-level features and then lets XGBoost perform the final classification.

#### Inputs

The final tabular multimodal vector contains:

- structured summary features from vitals, labs, and static context
- ClinicalBERT note embeddings
- note metadata features
- recent event features from antibiotics, vasopressors, and cultures

#### Structured features

The structured branch adds:

- `mean`, `min`, `max`, `last` aggregations
- missingness indicators
- one-hot static categorical features from `GENDER`, `ETHNICITY`, `FIRST_CAREUNIT`, and `LAST_CAREUNIT`

For the saved `24h` run, the structured feature count is:

- `428`

#### Text features

Clinical notes are embedded using frozen BioClinicalBERT and summarized at the stay level using:

- mean note embedding across windows
- closest-window embedding

For the saved `24h` run:

- text embedding features = `1536`

#### Additional feature blocks

- note metadata features = `14`
- clinical-event features = `9`

#### Final fused feature vector

For the saved `24h` `xgboost_text_augmented` run:

- total features = `1987`

#### Why this model performed well

This architecture preserved the strongest part of the project:

- tree-based learning on structured clinical features

while still incorporating the text modality in a meaningful way through:

- ClinicalBERT note embeddings
- note usage patterns
- clinically relevant event context

#### Practical finding

This model delivered the strongest custom multimodal performance across horizons and became the main practical multimodal model for the report.

The detailed architecture is documented separately in:

- `paper/custom_model_xgboost_architecture.md`

### 2.4 Aligned Transformer + XGBoost hybrid model

This was the final hybrid extension added after the XGBoost-based multimodal pipeline became successful.

Implemented variants:

- `aligned_transformer_encoder`
- `aligned_transformer_xgboost`

#### Core idea

This model separates multimodal learning into two stages:

1. a neural alignment stage
2. a tree-based prediction stage

The alignment stage learns a shared multimodal embedding from:

- structured ICU trajectories
- note-window embeddings

The second stage then trains XGBoost on:

- the learned aligned representation
- structured summary features
- note metadata
- clinical-event features

#### Why this model was added

This hybrid design was introduced to combine the strengths of:

- Transformer-style temporal modeling and cross-modal alignment
- XGBoost-style tabular discrimination

In other words, the model uses the neural block to understand temporal and multimodal structure, but uses XGBoost for the final decision boundary.

#### Stage 1: aligned neural encoder

The alignment encoder receives:

- structured sequence: `48 x 75`
- note-window sequence: up to `8 x 768`
- structured summary side features

It then learns a fixed multimodal aligned embedding:

- aligned dimension = `256`

This intermediate model is saved and evaluated as:

- `aligned_transformer_encoder`

#### Stage 2: XGBoost on fused features

The second stage concatenates:

- aligned neural embedding
- stay-level structured summary features
- note metadata features
- event features

and trains an XGBoost classifier on the fused feature vector.

This final model is saved and evaluated as:

- `aligned_transformer_xgboost`

#### Practical finding

The neural encoder alone was only moderately predictive, but once its aligned representation was combined with XGBoost, the hybrid system became very strong. This supports the idea that the alignment stage was useful, but tree-based learning was still the better final classifier for this dataset.

The detailed architecture is documented separately in:

- `paper/aligned_transformer_xgboost_architecture.md`

### 2.5 Ablation variants

Two ablation frameworks are implemented:

- ablation on the tabular multimodal custom model
- ablation on the aligned Transformer + XGBoost hybrid model

The executable structured variants are:

- `vitals_only`
- `vitals_labs`
- `structured_full`

These ablations answer the question:

- how much performance comes from basic vital signs alone
- how much is gained by laboratory measurements
- how much additional benefit comes from the full structured feature set before adding text and event information

## 3. Recommended Model Positioning in the Paper

For the final report, the implemented models can be described in the following hierarchy:

| Model family | Role in the paper | Main purpose |
|---|---|---|
| Structured baselines | baseline comparison | strong structured-only benchmarks |
| Deep multimodal fusion | original custom neural model | end-to-end multimodal sequence fusion |
| Tabular multimodal XGBoost | main custom model | strongest practical multimodal model |
| Aligned Transformer + XGBoost | advanced hybrid model | neural alignment plus tree-based prediction |

A clean narrative is:

- start from structured baselines
- introduce the first neural multimodal fusion model
- show that feature-level multimodal XGBoost performed better in practice
- then present the aligned Transformer + XGBoost model as a hybrid extension that combines alignment and strong classification

## 4. Key Improvements Over the Initial Approach

The project changed substantially from the initial version to the final version.

### 4.1 Added multimodal fusion

The earliest practical benchmark was based mainly on structured ICU variables. The final project added multimodal information through:

- clinical-note embeddings from BioClinicalBERT
- note metadata features
- recent antibiotic, culture, and vasopressor event features
- explicit deep fusion strategies
- explicit hybrid alignment strategies

This moved the project from a structured-only early sepsis benchmark to a true multimodal prediction framework.

### 4.2 Better preprocessing and data hygiene

Several preprocessing improvements were added during development:

- patient-aware cohort construction
- time-based censoring of notes and structured data at prediction time
- keyword masking for explicit sepsis terms
- decision-threshold selection on validation data instead of always using `0.5`
- leakage audit cells for patient overlap, time leakage, suspicious labels, and shuffled-label sanity checks

These changes improved both the reliability of the experiments and the defensibility of the final paper.

### 4.3 Better feature engineering

The final structured pipeline became richer than the initial version by adding or emphasizing:

- `FiO2`
- `GCS total`
- `urine output`
- missingness-aware features
- one-hot static categorical context
- note metadata features
- event features from antibiotics, cultures, and vasopressors

These additions were especially important for the stronger XGBoost-based multimodal models.

### 4.4 Better model design

The project evolved through three successive custom-model ideas:

1. deep end-to-end multimodal fusion
2. feature-level XGBoost multimodal fusion
3. aligned Transformer + XGBoost hybrid fusion

This progression is important because it shows that the final design was not chosen arbitrarily. It emerged from comparing what worked best empirically on this dataset.

### 4.5 Better performance

In practice, the final strong custom models were:

- `xgboost_text_augmented`
- `aligned_transformer_xgboost`

Representative test-set results from the saved runs are:

| Horizon | Deep multimodal best variant | Tabular multimodal XGBoost | Aligned Transformer + XGBoost |
|---|---:|---:|---:|
| `6h` AUROC / AUPRC | `0.797 / 0.277` (`cross_modal_attention`) | `0.999 / 0.979` | `0.999 / 0.975` |
| `12h` AUROC / AUPRC | `0.670 / 0.100` (`cross_modal_attention`) | `0.999 / 0.971` | `0.999 / 0.961` |
| `24h` AUROC / AUPRC | `0.745 / 0.164` (`cross_modal_attention`) or `0.743 / 0.169` (`late_fusion`) | `0.999 / 0.982` | `0.999 / 0.979` |

The important methodological conclusion is:

- neural multimodal fusion alone was not the strongest predictor on this dataset
- multimodal feature fusion with XGBoost was substantially stronger
- the hybrid aligned model preserved strong performance while retaining an explicit neural alignment stage

## 5. Challenges Faced During Implementation

The final pipeline was shaped not only by model accuracy, but also by several real implementation challenges.

### 5.1 GPU and training issues

The deep multimodal models were significantly more expensive than the tabular ones.

Main GPU-related issues included:

- long runtimes for BERT fine-tuning
- out-of-memory or terminated runs when using larger batch sizes
- the need to reduce batch size and unfrozen BERT layers
- the need for gradient checkpointing in some text fine-tuning experiments
- difficulty monitoring long runs, which later led to live progress logging

These issues directly influenced the decision to explore tree-based multimodal fusion and the aligned hybrid pipeline.

### 5.2 Preprocessing bottlenecks

The largest preprocessing bottleneck came from chunked extraction of measurements from:

- `CHARTEVENTS.csv`
- `LABEVENTS.csv`

This was especially slow because:

- the event tables are very large
- item IDs had to be resolved by concept keywords
- timestamps had to be parsed
- some events needed ICU-stay attachment repair

As a result, feature extraction was a major runtime bottleneck, and rerunning early pipeline stages unnecessarily was expensive.

### 5.3 Multimodal alignment issues

Data alignment was one of the most important technical challenges in the project.

The system had to align:

- hourly structured measurements
- ICU stay boundaries
- Sepsis-3 onset and suspicion times
- prediction times for each horizon
- note timestamps
- note windows
- recent event windows

Specific alignment risks included:

- future leakage from notes or measurements after prediction time
- patient overlap across splits
- mismatch between sepsis onset time and prediction horizon
- padding effects in neural sequence models
- misleading text shortcuts from explicit mention of sepsis

These issues motivated:

- strict time censoring
- patient-level split checks
- keyword masking sensitivity analysis
- dedicated leakage-audit cells in the visualization notebook

### 5.4 Evaluation pitfalls

Another challenge was metric interpretation.

Because the dataset is imbalanced, raw accuracy can be misleading. For example, a model can achieve high accuracy by predicting mostly negatives while still having poor AUROC or especially poor AUPRC.

This led to the final evaluation emphasis on:

- AUPRC as the primary metric
- AUROC as the secondary discrimination metric
- threshold-aware precision, recall, and F1
- calibration and confusion matrices

## 6. Final Summary

The final system is no longer a single model. It is a multimodel framework for early sepsis prediction built on a common data pipeline.

The final implementation includes:

- structured tabular baselines
- a deep multimodal Transformer/GRU + BERT fusion model
- a stronger XGBoost-based multimodal feature-fusion model
- a hybrid aligned Transformer + XGBoost model
- separate evaluation, visualization, leakage-audit, and ablation pipelines

The most important conceptual evolution of the project is:

- from structured-only modeling
- to neural multimodal fusion
- to stronger practical multimodal tabular fusion
- to a final hybrid model that combines neural alignment with XGBoost classification

This progression is itself one of the strongest aspects of the project, because it shows a complete experimental journey rather than a single static model choice.
