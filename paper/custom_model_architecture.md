# Custom Multimodal Model Architecture

## Model Summary

The custom model in this project is a dual-branch multimodal classifier for early sepsis prediction. One branch encodes structured ICU time-series data, and the other branch encodes temporally grouped clinical notes. The shared backbone is combined with one of four fusion heads:

- `early_fusion`
- `late_fusion`
- `gated_fusion`
- `cross_modal_attention`

In the saved experiments, the default structured encoder is a GRU and the text embedding backend is BioClinicalBERT.

## Input Representation

Each ICU stay is converted into two fixed-size tensors:

- Structured input: `X_struct ∈ R^(48 x 75)`
- Text input: `X_text ∈ R^(8 x 768)`

### Structured branch input

The structured tensor contains the most recent 48 hourly steps before the prediction time. Each hourly step contains 75 numeric features:

- 72 dynamic features from hourly aggregation of 8 chart variables and 10 lab variables using `mean`, `min`, `max`, and `last`
- `age_at_icu_intime`
- `hours_since_icu_admit`
- `hours_to_prediction`

Categorical static variables such as gender, ethnicity, and care unit are present in the raw table but are not used by the neural model because only numeric columns are selected during multimodal dataset preparation.

### Text branch input

Clinical notes are filtered to Physician, Nursing, and Radiology categories, censored at the prediction time, and grouped into 6-hour windows. All notes inside each window are concatenated into a single text string. Each window is embedded with BioClinicalBERT, using mean pooling over token hidden states, producing a 768-dimensional embedding per window. The first 8 note windows after sorting by `note_window_index` are retained, which corresponds to the 8 most recent 6-hour windows before prediction.

## Backbone Architecture

The active forward path is:

1. Structured sequence encoder
2. Attention-based text window aggregator
3. Fusion head
4. Single-neuron binary classifier output

An additional `TextWindowEncoder` module is instantiated in the codebase but is not used in the current `forward()` path, so it should not be treated as part of the active architecture.

## Table 1. Shared Backbone Architecture

This table reflects the active computation graph used by all four multimodal variants under the default configuration.

| Stage | Layer / Operation | Input Shape | Output Shape | Key Hyperparameters | Parameters |
|---|---|---:|---:|---|---:|
| 1 | Structured input | `B x 48 x 75` | `B x 48 x 75` | 48 hourly steps, 75 numeric features | 0 |
| 2 | GRU structured encoder | `B x 48 x 75` | `B x 128` | hidden size = 128, 1 layer, batch first | 78,720 |
| 3 | Text input | `B x 8 x 768` | `B x 8 x 768` | 8 note windows, 768-d BioClinicalBERT embeddings | 0 |
| 4 | Linear projection inside text attention | `B x 8 x 768` | `B x 8 x 128` | linear `768 -> 128` | 98,432 |
| 5 | `tanh` + dropout | `B x 8 x 128` | `B x 8 x 128` | dropout = 0.2 | 0 |
| 6 | Attention score layer | `B x 8 x 128` | `B x 8 x 1` | linear `128 -> 1` | 129 |
| 7 | Softmax across note windows | `B x 8` | `B x 8` | normalized attention over 8 windows | 0 |
| 8 | Weighted sum of note-window states | `B x 8 x 128` | `B x 128` | attention pooling | 0 |

### Shared backbone total

- Active shared parameters = `78,720 + 98,432 + 129 = 177,281`
- Shared latent dimensions:
  - Structured representation: `128`
  - Text representation: `128`

## Table 2. Fusion Head Variants

### A. Early Fusion

| Stage | Layer / Operation | Input Shape | Output Shape | Key Hyperparameters | Parameters |
|---|---|---:|---:|---|---:|
| 1 | Concatenate structured and text embeddings | `B x 128` + `B x 128` | `B x 256` | feature fusion | 0 |
| 2 | Linear | `B x 256` | `B x 128` | `256 -> 128` | 32,896 |
| 3 | ReLU + dropout | `B x 128` | `B x 128` | dropout = 0.2 | 0 |
| 4 | Output classifier | `B x 128` | `B x 1` | `128 -> 1` | 129 |

- Fusion head parameters: `33,025`
- Active model parameters: `177,281 + 33,025 = 210,306`

### B. Late Fusion

| Stage | Layer / Operation | Input Shape | Output Shape | Key Hyperparameters | Parameters |
|---|---|---:|---:|---|---:|
| 1 | Structured logit head | `B x 128` | `B x 1` | linear `128 -> 1` | 129 |
| 2 | Text logit head | `B x 128` | `B x 1` | linear `128 -> 1` | 129 |
| 3 | Average logits | `B x 1`, `B x 1` | `B x 1` | weight = 0.5 / 0.5 | 0 |

- Fusion head parameters: `258`
- Active model parameters: `177,281 + 258 = 177,539`

### C. Gated Fusion

| Stage | Layer / Operation | Input Shape | Output Shape | Key Hyperparameters | Parameters |
|---|---|---:|---:|---|---:|
| 1 | Structured projection | `B x 128` | `B x 128` | linear `128 -> 128`, `tanh` | 16,512 |
| 2 | Text projection | `B x 128` | `B x 128` | linear `128 -> 128`, `tanh` | 16,512 |
| 3 | Gate generation | `B x 256` | `B x 128` | linear `256 -> 128`, sigmoid | 32,896 |
| 4 | Elementwise gated mixture | `B x 128`, `B x 128` | `B x 128` | `g * s + (1-g) * t` | 0 |
| 5 | Output classifier | `B x 128` | `B x 1` | linear `128 -> 1` | 129 |

- Fusion head parameters: `66,049`
- Active model parameters: `177,281 + 66,049 = 243,330`

### D. Cross-Modal Attention

| Stage | Layer / Operation | Input Shape | Output Shape | Key Hyperparameters | Parameters |
|---|---|---:|---:|---|---:|
| 1 | Structured projection to query | `B x 128` | `B x 128` | linear `128 -> 128` | 16,512 |
| 2 | Text projection to key/value | `B x 128` | `B x 128` | linear `128 -> 128` | 16,512 |
| 3 | Add singleton sequence dimension | `B x 128` | `B x 1 x 128` | single query token, single key/value token | 0 |
| 4 | Multi-head attention | `B x 1 x 128` | `B x 1 x 128` | 4 heads, embed dim = 128 | 66,048 |
| 5 | Linear | `B x 128` | `B x 128` | `128 -> 128` | 16,512 |
| 6 | ReLU + dropout | `B x 128` | `B x 128` | dropout = 0.2 | 0 |
| 7 | Output classifier | `B x 128` | `B x 1` | `128 -> 1` | 129 |

- Fusion head parameters: `115,713`
- Active model parameters: `177,281 + 115,713 = 292,994`

## Table 3. Complete Model Comparison

This table reports active parameters only, meaning layers actually used in the forward path.

| Variant | Structured Encoder | Text Aggregator | Fusion Head | Active Parameters | Final Output |
|---|---|---|---|---:|---|
| Early Fusion | GRU, hidden = 128 | Attention pooling over 8 windows | Concatenation + MLP | 210,306 | Binary logit |
| Late Fusion | GRU, hidden = 128 | Attention pooling over 8 windows | Average of modality-specific logits | 177,539 | Binary logit |
| Gated Fusion | GRU, hidden = 128 | Attention pooling over 8 windows | Learned elementwise gate | 243,330 | Binary logit |
| Cross-Modal Attention | GRU, hidden = 128 | Attention pooling over 8 windows | 4-head attention over single structured/text tokens | 292,994 | Binary logit |

## Training Setup

The multimodal variants are trained with:

- optimizer: AdamW
- learning rate: `5e-4`
- weight decay: `1e-4`
- batch size: `16`
- epochs: `10`
- dropout: `0.2`
- gradient clipping norm: `1.0`
- loss: binary cross-entropy with logits
- class imbalance handling: positive class weighting using `N_negative / N_positive`
- checkpoint selection metric: validation AUPRC

## Important Implementation Note

The repository instantiates a `TextWindowEncoder` block with 98,432 parameters, but it is not used in the model's current `forward()` method. Therefore:

- it is part of the instantiated PyTorch module
- it contributes to the raw parameter count reported by `model.parameters()`
- it is not part of the active forward computation
- it should not be included in the architecture table unless you explicitly want to document unused code

If you want to report raw instantiated parameter counts instead of active forward-path counts, add 98,432 parameters to each variant:

- Early Fusion raw instantiated total: `308,738`
- Late Fusion raw instantiated total: `275,971`
- Gated Fusion raw instantiated total: `341,762`
- Cross-Modal Attention raw instantiated total: `391,426`

## Recommended Paper Description

If you need one concise report-ready paragraph, use this:

> We implemented a dual-branch multimodal neural network for early sepsis prediction. The structured branch receives a 48-hour sequence of 75 normalized hourly ICU features and encodes it using a GRU with 128 hidden units. The text branch groups clinical notes into 6-hour windows before the prediction time, embeds each window using BioClinicalBERT, and applies attention pooling across up to 8 recent note windows to obtain a 128-dimensional text representation. These structured and text embeddings are then fused using one of four strategies: early feature fusion, late logit fusion, gated fusion, or cross-modal attention, followed by a single-neuron binary output layer for sepsis risk prediction.

## Prompt To Paste Into ChatGPT

If you want ChatGPT to turn this into a cleaner publication-style table, paste the following:

```text
I am writing the Methods section for a sepsis prediction paper. Please convert the following architecture specification into:
1. a publication-style architecture table
2. a compact paragraph description
3. a figure-caption style summary

Use clear academic wording. Keep the layer names faithful to the implementation. Do not invent layers that are not present.

Model type:
- Dual-branch multimodal classifier for early sepsis prediction

Inputs:
- Structured branch input: B x 48 x 75
- Text branch input: B x 8 x 768

Structured branch:
- GRU, input size 75, hidden size 128, 1 layer, batch_first=True
- Output is final hidden state: B x 128
- Parameters: 78,720

Text branch:
- BioClinicalBERT embeddings are precomputed, not trained end-to-end in the multimodal model
- Each note window embedding is 768-d
- Attention text aggregator:
  - Linear 768 -> 128, parameters 98,432
  - tanh
  - dropout 0.2
  - Linear 128 -> 1, parameters 129
  - softmax across 8 note windows
  - weighted sum to produce B x 128 text representation

Fusion variants:

Early fusion:
- concatenate structured and text vectors: B x 128 + B x 128 -> B x 256
- Linear 256 -> 128, parameters 32,896
- ReLU
- dropout 0.2
- Linear 128 -> 1, parameters 129
- Fusion head total parameters: 33,025
- Active total parameters: 210,306

Late fusion:
- Linear 128 -> 1 on structured vector, parameters 129
- Linear 128 -> 1 on text vector, parameters 129
- average the two logits
- Fusion head total parameters: 258
- Active total parameters: 177,539

Gated fusion:
- structured projection: Linear 128 -> 128, tanh, parameters 16,512
- text projection: Linear 128 -> 128, tanh, parameters 16,512
- gate generation: concatenate projected vectors to 256-d, Linear 256 -> 128, sigmoid, parameters 32,896
- elementwise gated fusion: g*s + (1-g)*t
- output Linear 128 -> 1, parameters 129
- Fusion head total parameters: 66,049
- Active total parameters: 243,330

Cross-modal attention:
- structured projection to query: Linear 128 -> 128, parameters 16,512
- text projection to key/value: Linear 128 -> 128, parameters 16,512
- add singleton token dimension so each modality is represented as one token
- MultiheadAttention with embed_dim=128, num_heads=4, parameters 66,048
- Linear 128 -> 128, parameters 16,512
- ReLU
- dropout 0.2
- Linear 128 -> 1, parameters 129
- Fusion head total parameters: 115,713
- Active total parameters: 292,994

Important note:
- There is an unused TextWindowEncoder module instantiated in code, but it is not used in forward propagation. Exclude it from the main architecture table unless adding an implementation note.

Please output the table in Markdown.
```
