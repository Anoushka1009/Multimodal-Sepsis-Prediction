# Aligned Transformer + XGBoost Architecture

## Model Summary

This document describes the separate hybrid multimodal model implemented in the repository as:

- `aligned_transformer_encoder`
- `aligned_transformer_xgboost`

This model is a two-stage architecture:

1. A neural alignment stage learns a fixed-length multimodal representation from structured ICU trajectories and temporally grouped clinical notes.
2. A second-stage XGBoost classifier uses that learned aligned representation together with engineered structured, note-metadata, and clinical-event features.

In other words, the model does not ask XGBoost to learn directly from raw sequences. Instead, the sequence encoders first compress structured and text information into an aligned patient embedding `z_align`, and XGBoost then performs the final decision-making on a fused tabular feature vector.

## Why This Model Exists

This architecture was added to combine the strengths of both modeling families already present in the project:

- transformer-based sequence modeling for temporal and cross-modal alignment
- tree-based tabular learning for strong final discrimination on aggregated EHR features

The goal is to keep the structured and note-alignment inductive bias of the neural multimodal model, while allowing the final predictor to use the stronger tabular decision boundary of XGBoost.

## Saved Run Documented Here

The description below reflects the saved `24h` hybrid run stored under:

- `results/processed/10_aligned_transformer_xgboost/horizon_24h_aligned_transformer_xgboost_summary.csv`
- `results/processed/10_aligned_transformer_xgboost/horizon_24h_aligned_transformer_xgboost_experiment_plan.csv`
- `results/processed/10_aligned_transformer_xgboost/horizon_24h_aligned_transformer_xgboost_feature_manifest.csv`

For this saved run:

- structured encoder = `transformer`
- text input mode = `frozen_embedding`
- text embedding backend = `transformer:emilyalsentzer/Bio_ClinicalBERT`
- hidden dimension = `128`
- aligned embedding dimension = `256`
- batch size = `8`
- epochs = `10`

Important implementation note:

- Although the conceptual design is "Transformer + BERT + XGBoost", the saved run does **not** fine-tune BERT end to end.
- BioClinicalBERT is used as a frozen note-embedding backend that produces `768`-dimensional window embeddings.
- The trainable text branch in the alignment model is therefore the window-level transformer and attention pooling layers on top of those embeddings.

## High-Level Pipeline

The hybrid model processes each ICU stay in six stages:

1. Build a 48-hour structured sequence tensor.
2. Build up to 8 temporally ordered note-window embeddings from BioClinicalBERT.
3. Encode the structured sequence with a transformer encoder.
4. Encode the note windows with a window-level transformer and cross-modal alignment block to obtain a learned aligned representation `z_align`.
5. Concatenate `z_align` with engineered structured summary features, note metadata features, and clinical-event features.
6. Train XGBoost on the resulting fused tabular feature vector.

## Input Representation

### 1. Structured sequence input

Each ICU stay contributes the most recent 48 hourly steps before the prediction time:

- structured sequence shape: `B x 48 x 75`

The `75` numeric channels are the same numeric multimodal channels already used by the neural multimodal pipeline:

- `72` dynamic vitals/lab summary channels from hourly engineering
- `age_at_icu_intime`
- `hours_since_icu_admit`
- `hours_to_prediction`

These 75 channels are normalized using training-split statistics before entering the neural alignment stage.

### 2. Structured summary side input

In parallel to the structured transformer sequence branch, the model also builds a stay-level structured summary vector for the neural stage:

- aggregations: `mean`, `min`, `max`, `last`
- plus per-channel missingness rate

Since this summary is built only from the 75 numeric sequence channels, the summary dimension is:

- `75 x 4 = 300` aggregation features
- `75` missingness features
- total structured summary dimension = `375`

This `375`-dimensional vector is used only inside the neural alignment stage. It is not identical to the full `428`-dimensional structured feature block used later by XGBoost.

### 3. Text input

Clinical notes are grouped into 6-hour windows before prediction:

- maximum windows retained: `8`
- note-window embedding size: `768`
- text tensor shape: `B x 8 x 768`

For the saved run, these note-window embeddings come from frozen BioClinicalBERT:

- backend: `emilyalsentzer/Bio_ClinicalBERT`
- current text mode: `frozen_embedding`

So the saved alignment encoder receives note-window embeddings rather than raw tokens.

## Stage 1: `aligned_transformer_encoder`

### Architectural role

`aligned_transformer_encoder` is the neural first stage. Its job is not to be the final production classifier. Its main purpose is to learn an aligned multimodal representation:

- `z_align ∈ R^256`

During training, it uses an auxiliary single-neuron classification head so that the aligned embedding is supervised toward sepsis discrimination. After training, the aligned embedding is exported and used by stage 2.

### Active forward path

The active neural forward path is:

1. Structured transformer encoder
2. Text window transformer encoder
3. Attention pooling over note windows
4. Structured summary encoder
5. Structured-context fusion
6. Bidirectional cross-modal alignment encoder
7. Auxiliary classifier head

### Table 1. Detailed Architecture of `aligned_transformer_encoder`

This table reflects the saved `24h` run with `hidden_dim = 128`, `aligned_dim = 256`, `structured_input_dim = 75`, `structured_summary_dim = 375`, and frozen BioClinicalBERT note embeddings of size `768`.

| Stage | Layer / Operation | Input Shape | Output Shape | Key Hyperparameters | Parameters |
|---|---|---:|---:|---|---:|
| 1 | Structured sequence input | `B x 48 x 75` | `B x 48 x 75` | 48 hourly steps, 75 numeric channels | 0 |
| 2 | Structured input projection | `B x 48 x 75` | `B x 48 x 128` | linear `75 -> 128` | 9,728 |
| 3 | Structured transformer encoder | `B x 48 x 128` | `B x 48 x 128` | 2 layers, 4 heads, feedforward dim `512`, dropout `0.2` | 396,544 |
| 4 | Masked mean pooling | `B x 48 x 128` | `B x 128` | sequence summary for structured branch | 0 |
| 5 | Text window input | `B x 8 x 768` | `B x 8 x 768` | up to 8 BioClinicalBERT note windows | 0 |
| 6 | Text input projection | `B x 8 x 768` | `B x 8 x 128` | linear `768 -> 128` | 98,432 |
| 7 | Text window transformer encoder | `B x 8 x 128` | `B x 8 x 128` | 1 layer, 4 heads, feedforward dim `512`, dropout `0.2` | 198,272 |
| 8 | Attention score layer | `B x 8 x 128` | `B x 8 x 1` | linear `128 -> 1` | 129 |
| 9 | Softmax over note windows | `B x 8` | `B x 8` | normalized temporal attention | 0 |
| 10 | Weighted note-window pooling | `B x 8 x 128` | `B x 128` | attention-pooled text representation | 0 |
| 11 | Structured summary input | `B x 375` | `B x 375` | `mean/min/max/last + missingness` | 0 |
| 12 | Summary encoder MLP | `B x 375` | `B x 128` | `375 -> 128 -> 128`, ReLU, dropout `0.2` | 64,640 |
| 13 | Structured-context fusion | `B x 256` | `B x 128` | concat structured pooled vector and summary vector, then `256 -> 128` | 32,896 |
| 14 | Structured projection inside alignment block | `B x 48 x 128` | `B x 48 x 128` | linear `128 -> 128` | 16,512 |
| 15 | Text projection inside alignment block | `B x 8 x 128` | `B x 8 x 128` | linear `128 -> 128` | 16,512 |
| 16 | Structured-to-text multi-head attention | `B x 48 x 128` query, `B x 8 x 128` key/value | `B x 48 x 128` | 4 heads | 66,048 |
| 17 | Text-to-structured multi-head attention | `B x 8 x 128` query, `B x 48 x 128` key/value | `B x 8 x 128` | 4 heads | 66,048 |
| 18 | Layer normalization | `B x 48 x 128`, `B x 8 x 128` | unchanged | 2 layer norms | 512 |
| 19 | Cross-modal summary pooling | structured + text sequences | `B x 128`, `B x 128` | masked mean over each aligned sequence | 0 |
| 20 | Alignment output MLP | `B x 512` | `B x 256` | concat aligned structured summary, aligned text summary, pooled structured repr, pooled text repr; then `512 -> 256` | 131,328 |
| 21 | Auxiliary classifier hidden layer | `B x 256` | `B x 128` | linear `256 -> 128`, ReLU, dropout `0.2` | 32,896 |
| 22 | Auxiliary binary output | `B x 128` | `B x 1` | linear `128 -> 1` | 129 |

### Parameter totals for `aligned_transformer_encoder`

Active trainable parameters in the saved frozen-embedding configuration:

- Structured transformer branch: `406,272`
- Text transformer branch: `296,704`
- Attention text aggregator: `129`
- Structured summary encoder: `64,640`
- Structured-context fusion: `32,896`
- Cross-modal alignment encoder: `296,960`
- Auxiliary classifier head: `33,025`
- Total neural-stage parameters: `1,130,626`

If you want to describe only the representation-learning backbone and exclude the auxiliary supervision head, subtract `33,025`, which gives:

- alignment backbone without final classifier = `1,097,601`

### What `aligned_transformer_encoder` produces

The main output of stage 1 is:

- aligned embedding `z_align ∈ R^256`

This aligned embedding is then exported as `aligned_embedding_000` through `aligned_embedding_255` and passed to the stage-2 XGBoost model.

### Saved `24h` performance of `aligned_transformer_encoder`

The saved `24h` alignment encoder alone is only moderately predictive:

- validation AUROC = `0.7184`
- validation AUPRC = `0.1177`
- test AUROC = `0.7452`
- test AUPRC = `0.1468`

This is expected. The encoder is mainly useful as a learned feature extractor. The final performance gain comes after passing the aligned embedding into XGBoost together with the engineered tabular features.

## Stage 2: `aligned_transformer_xgboost`

### Architectural role

`aligned_transformer_xgboost` is the final deployed hybrid model. It takes the learned aligned embedding from stage 1 and fuses it with explicit engineered feature groups before classification.

### Stage-2 feature groups

For the saved `24h` run, the XGBoost input vector contains `707` features total:

| Feature group | Count | Description |
|---|---:|---|
| Aligned embedding features | 256 | learned cross-modal representation exported by stage 1 |
| Structured tabular features | 428 | stay-level structured summaries, missingness, and static categorical one-hot features |
| Note metadata features | 14 | note count, recency, window position, and category-count features |
| Clinical-event features | 9 | antibiotic, vasopressor, and culture recency/count features |
| Total fused XGBoost input | 707 | final tabular vector for XGBoost |

### Structured tabular branch used by stage 2

The `428` structured stage-2 features are:

- `300` aggregated numeric features from the 75 structured channels using `mean/min/max/last`
- `75` missingness-rate features
- `53` static categorical one-hot features from:
  - `GENDER`
  - `ETHNICITY`
  - `FIRST_CAREUNIT`
  - `LAST_CAREUNIT`

This is broader than the `375`-dimensional structured summary side input used inside stage 1, because stage 2 additionally includes static categorical context.

### Table 2. Detailed Architecture of `aligned_transformer_xgboost`

| Stage | Operation | Output Dimension | Details |
|---|---|---:|---|
| 1 | Export aligned embedding from stage 1 | 256 | learned multimodal representation `z_align` |
| 2 | Structured tabular branch | 428 | structured summaries, missingness, and static categorical one-hot features |
| 3 | Note metadata branch | 14 | note counts, note recency, window positions, category-window counts, category-note counts |
| 4 | Clinical-event branch | 9 | antibiotic, vasopressor, and culture count/flag/recency features |
| 5 | Multimodal concatenation | 707 | `256 + 428 + 14 + 9` |
| 6 | Classifier | 1 probability output | XGBoost binary classifier |

### Table 3. XGBoost Hyperparameters

The second-stage classifier uses the baseline XGBoost configuration already defined in the project:

| Component | Value |
|---|---|
| Model | `XGBClassifier` |
| Input dimension | `707` |
| Trees (`n_estimators`) | `300` |
| Maximum depth | `6` |
| Learning rate | `0.05` |
| Row subsampling | `0.8` |
| Column subsampling | `0.8` |
| Evaluation metric | `logloss` |
| Preprocessing | `SimpleImputer` |
| Threshold selection | validation-set threshold chosen to maximize `F1` |

For the saved `24h` run:

- validation-selected decision threshold = `0.3192067444`

### Saved `24h` performance of `aligned_transformer_xgboost`

The final hybrid model is much stronger than the neural stage alone:

- validation AUROC = `0.9982`
- validation AUPRC = `0.9527`
- test AUROC = `0.9992`
- test AUPRC = `0.9789`
- test F1 = `0.9497`

This result shows that the aligned neural representation is useful once it is combined with the explicit engineered tabular features and passed through a strong tree-based classifier.

## Interpretation of the Two Stages

The simplest way to explain the two components in a paper is:

- `aligned_transformer_encoder` learns a cross-modal patient representation from temporally ordered structured data and note windows.
- `aligned_transformer_xgboost` uses that learned representation as one feature group inside a broader multimodal tabular predictor.

So the final hybrid model is not "neural versus XGBoost." It is:

- neural alignment for representation learning
- XGBoost for final decision-making

## Why the Hybrid Model Can Perform Better

This design can outperform a pure neural fusion classifier for three main reasons:

1. The transformer-based stage can still model temporal and cross-modal relationships.
2. The aligned representation captures interaction information that a purely tabular model would not see directly.
3. XGBoost remains very effective on the final fused tabular feature vector, especially in class-imbalanced EHR prediction tasks.

So the architecture keeps the alignment bias of deep multimodal learning while using a more robust final predictor.

## Report-Ready Methods Description

If you want a direct Methods-style paragraph, use this:

> We implemented a two-stage hybrid multimodal model for early sepsis prediction. In the first stage, hourly structured ICU measurements from the 48-hour pre-prediction window were encoded using a transformer-based structured branch, while temporally grouped clinical notes were embedded using BioClinicalBERT and processed with a window-level transformer and attention pooling module. A bidirectional cross-modal attention block then combined the structured and text sequence representations to produce a 256-dimensional aligned patient embedding. In the current saved implementation, BioClinicalBERT was used as a frozen note-embedding backend rather than being fine-tuned end to end. In the second stage, the learned aligned embedding was concatenated with engineered structured summary features, note metadata features, and recent clinical-event features, and the resulting 707-dimensional multimodal feature vector was classified using XGBoost for binary sepsis prediction.

## Short Figure-Caption Version

> Architecture of the aligned Transformer + XGBoost hybrid model. A transformer-based structured encoder and a note-window text encoder first learn a 256-dimensional aligned multimodal patient representation from 48-hour structured trajectories and BioClinicalBERT note-window embeddings. This aligned representation is then concatenated with structured tabular summaries, note metadata, and recent clinical-event features and classified using XGBoost.

## Important Implementation Notes

- The saved hybrid results correspond to `text_input_mode = frozen_embedding`, not end-to-end BERT fine-tuning.
- Therefore, the parameter counts above exclude the external BioClinicalBERT backbone and describe only the active trainable neural alignment stage.
- If future runs switch to `text_encoder_mode = bert_finetune`, the stage-1 architecture stays conceptually similar but the trainable parameter count increases substantially.
- The alignment encoder is trained with a classification head, but the final reported hybrid model uses the exported aligned embedding plus XGBoost for prediction.
