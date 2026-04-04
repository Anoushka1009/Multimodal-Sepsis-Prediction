# Custom Tabular Multimodal Architecture

## Model Summary

The strongest custom multimodal model in this project is not the deep fusion network. It is a tabular multimodal pipeline that combines:

- structured ICU features derived from vitals, laboratory values, and static patient context
- clinical-note embeddings extracted with BioClinicalBERT
- note metadata features
- recent treatment and microbiology event features
- an XGBoost classifier as the primary prediction model

The primary deployed variant is `xgboost_text_augmented`. A secondary stacked variant, `stacked_xgboost_notes`, is also implemented for comparison.

This architecture is well suited to the project because it preserves the strong predictive performance of tree-based structured models while still incorporating clinical notes in a multimodal way.

## High-Level Pipeline

The model processes each ICU stay in five stages:

1. Build a stay-level structured feature vector from the 48-hour pre-prediction history.
2. Build a stay-level text representation from note windows using BioClinicalBERT embeddings.
3. Build note metadata and recent clinical-event features.
4. Concatenate all feature groups into one multimodal tabular vector.
5. Train an XGBoost classifier on the fused multimodal feature vector.

Unlike the earlier neural fusion model, this architecture does not fuse modalities through attention or recurrent layers. Fusion occurs through feature concatenation at the tabular level, followed by tree-based learning.

## Input Construction

### 1. Structured branch

The starting structured table comes from the processed horizon dataset generated in Notebook 04. Each ICU stay contains hourly measurements over a 48-hour history window before the prediction time.

The tabular multimodal model converts this hourly sequence into stay-level summary features using:

- `mean`
- `min`
- `max`
- `last`

In addition, it adds:

- per-feature missingness rates
- one-hot encoded static categorical variables

For the saved `24h` run, this produces `428` structured features in total.

Structured feature breakdown in the current `24h` model:

| Structured component | Count | Description |
|---|---:|---|
| Aggregated numeric features | 300 | `75` numeric channels aggregated with `mean/min/max/last` |
| Missingness features | 75 | one missing-rate feature per numeric channel |
| Static categorical one-hot features | 53 | derived from `GENDER`, `ETHNICITY`, `FIRST_CAREUNIT`, `LAST_CAREUNIT` |
| Total structured features | 428 | final structured tabular branch |

The `75` numeric channels correspond to the engineered hourly features available in the structured horizon table, including dynamic vitals/labs and contextual numeric variables such as age and ICU time context.

### 2. Text branch

Clinical notes are preprocessed into 6-hour windows before prediction. Only configured clinical note categories are retained:

- Physician
- Nursing
- Radiology

Each note window is represented by an aggregated text string. That text is embedded using the configured pretrained model:

- `emilyalsentzer/Bio_ClinicalBERT`

The saved `24h` run used the backend:

- `transformer:emilyalsentzer/Bio_ClinicalBERT`

Each note window embedding is `768` dimensions. The tabular multimodal model does not fine-tune BioClinicalBERT end to end. Instead, it extracts fixed embeddings and summarizes them at the stay level using two aggregations:

- `mean` embedding across note windows
- `closest` embedding, meaning the embedding of the note window closest to prediction time

This yields:

- `768` mean-aggregated text features
- `768` closest-window text features
- total text embedding features = `1536`

### 3. Note metadata branch

The model also includes handcrafted note-usage features:

- `note_window_count`
- `note_total_count`
- `note_mean_count_per_window`
- `note_max_count_per_window`
- `note_closest_window_index`
- `note_oldest_window_index`
- `note_closest_recency_hours`
- `note_oldest_recency_hours`
- category-window counts for Physician, Nursing, and Radiology
- category-note counts for Physician, Nursing, and Radiology

This yields `14` note metadata features in the saved `24h` run.

### 4. Clinical-event branch

To strengthen early sepsis signal capture, the model also includes event features from the recent lookback window:

- antibiotic orders
- vasopressor orders
- culture orders

For each event family, three features are created:

- event count in the last 48 hours
- binary event flag in the last 48 hours
- hours since the most recent event

This yields `9` clinical-event features:

- `antibiotic_count_48h`
- `antibiotic_flag_48h`
- `antibiotic_hours_since_last`
- `vasopressor_count_48h`
- `vasopressor_flag_48h`
- `vasopressor_hours_since_last`
- `culture_count_48h`
- `culture_flag_48h`
- `culture_hours_since_last`

## Table 1. Feature-Level Architecture of `xgboost_text_augmented`

This table reflects the saved `24h` model in `results/processed/06_baseline_models/horizon_24h_tabular_multimodal_results.csv`.

| Stage | Operation | Output Dimension | Details |
|---|---|---:|---|
| 1 | Structured history input | hourly table | 48-hour pre-prediction structured history |
| 2 | Structured aggregation | 300 | `75` numeric channels with `mean/min/max/last` |
| 3 | Missingness encoding | 75 | one missing-rate feature per numeric channel |
| 4 | Static categorical one-hot encoding | 53 | gender, ethnicity, first care unit, last care unit |
| 5 | Structured branch output | 428 | final structured feature vector |
| 6 | BioClinicalBERT note embedding: mean | 768 | mean pooled across available note windows |
| 7 | BioClinicalBERT note embedding: closest | 768 | most recent note-window embedding |
| 8 | Text embedding branch output | 1536 | concatenated mean and closest note embeddings |
| 9 | Note metadata branch | 14 | note counts, recency, window positions, category counts |
| 10 | Clinical-event branch | 9 | antibiotic, vasopressor, and culture features |
| 11 | Multimodal concatenation | 1987 | `428 + 1536 + 14 + 9` |
| 12 | Classifier | 1 probability output | XGBoost binary classifier |

## Table 2. Classifier Architecture and Hyperparameters

The final classifier is XGBoost operating on the full `1987`-dimensional fused feature vector.

| Component | Value |
|---|---|
| Model | `XGBClassifier` |
| Input dimension | `1987` |
| Trees (`n_estimators`) | `300` |
| Maximum depth | `6` |
| Learning rate | `0.05` |
| Row subsampling | `0.8` |
| Column subsampling | `0.8` |
| Evaluation metric | `logloss` |
| Preprocessing before XGBoost | `SimpleImputer` |
| Threshold selection | validation-set threshold chosen to maximize `F1` |

For the saved `24h` run:

- validation-selected decision threshold = `0.5846581459`
- test AUROC = `0.9992728895`
- test AUPRC = `0.9787114930`

## Stacked Variant: `stacked_xgboost_notes`

The repository also contains a second multimodal variant called `stacked_xgboost_notes`. This is a two-stage ensemble rather than a single XGBoost classifier.

### Stage A: modality-specific models

Two separate models are trained:

1. A structured model:
   - XGBoost on the `428` structured features
2. A note model:
   - logistic regression on the note-derived features
   - input dimension = `1536 + 14 + 9 = 1559`

### Stage B: meta-classifier

A logistic-regression meta-model is then trained on:

- `structured_prob`
- `note_prob`
- the `14` note metadata features
- the `9` clinical-event features

So the meta-classifier input dimension is:

- `2 + 14 + 9 = 25`

This stacked variant achieved the following saved `24h` test performance:

- AUROC = `0.9965666704`
- AUPRC = `0.9290099638`

## Why This Architecture Performed Better

This model performed better than the deep multimodal fusion network for three main reasons:

1. The structured branch is handled by a tree-based learner, which is very effective for aggregated clinical tabular data.
2. Text is incorporated as additional multimodal evidence without forcing brittle end-to-end neural fusion.
3. Clinically meaningful auxiliary features such as note recency, note burden, antibiotic exposure, vasopressor use, and culture timing are made explicit instead of being left for a deep model to infer.

In other words, this custom architecture treats multimodality as a feature-fusion problem rather than a sequence-to-sequence fusion problem.

## Report-Ready Description

If you need a direct Methods-style description, use this:

> We implemented a tabular multimodal sepsis prediction model in which structured ICU data formed the primary predictive branch and clinical notes were incorporated as additional fused features. First, hourly structured measurements from the 48-hour pre-prediction window were converted into stay-level summary statistics using mean, minimum, maximum, and last-value aggregation, with additional missingness indicators and one-hot encoded static demographic and care-unit variables. Second, clinical notes from Physician, Nursing, and Radiology categories were grouped into 6-hour windows and embedded using BioClinicalBERT. For each stay, we concatenated the mean note embedding and the embedding of the note window closest to the prediction time. Third, we added note metadata features describing note frequency, timing, and category counts, along with recent antibiotic, vasopressor, and culture-order features. The resulting multimodal feature vector was used to train an XGBoost classifier for binary sepsis prediction. In the saved 24-hour prediction model, the fused feature vector contained 1,987 features: 428 structured features, 1,536 text-embedding features, 14 note metadata features, and 9 clinical-event features.

## Short Figure-Caption Version

> Architecture of the proposed tabular multimodal sepsis model. Hourly structured ICU measurements are summarized into stay-level statistical and missingness features, while clinical notes are embedded with BioClinicalBERT and supplemented with note metadata and recent treatment/culture event features. These modality-specific representations are concatenated into a single 1,987-dimensional multimodal vector and classified using XGBoost.

## Important Implementation Notes

- BioClinicalBERT is used as a frozen feature extractor in this pipeline, not as an end-to-end fine-tuned text branch.
- The saved `24h` feature counts are model-artifact counts from the current repository state and may change slightly if Notebook 04 is rerun with additional engineered features.
- The stacked variant is useful as a secondary comparison model, but `xgboost_text_augmented` is the cleanest primary custom model for the paper.
