from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_processing.text_processing import apply_configured_keyword_masking
from src.evaluation.metrics import compute_binary_classification_metrics
from src.models.multimodal import MultimodalClassifier


ID_COLUMNS = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]
EXCLUDED_STRUCTURED_COLUMNS = set(
    ID_COLUMNS
    + [
        "hour",
        "prediction_time",
        "split",
        "sepsis3_label",
        "sepsis_onset_time",
        "prediction_horizon_hours",
    ]
)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class PreparedMultimodalDataset:
    stay_index: pd.DataFrame
    structured_sequences: np.ndarray
    text_embeddings: np.ndarray
    text_token_ids: np.ndarray
    text_token_attention_masks: np.ndarray
    structured_summary_features: np.ndarray
    structured_padding_masks: np.ndarray
    text_padding_masks: np.ndarray
    labels: np.ndarray
    feature_columns: List[str]
    structured_summary_columns: List[str]
    text_embedding_dim: int
    text_embedding_backend: str
    text_input_mode: str


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str = "auto") -> torch.device:
    preferred = str(preferred).lower()
    if preferred == "auto":
        preferred = "cuda" if torch.cuda.is_available() else "cpu"
    if preferred == "cuda" and not torch.cuda.is_available():
        preferred = "cpu"
    return torch.device(preferred)


class HashingTextEmbedder:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = int(embedding_dim)
        self.backend_name = "hashing"

    def encode_texts(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        del batch_size
        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for index, text in enumerate(texts):
            tokens = TOKEN_RE.findall(str(text).lower())
            if not tokens:
                continue
            for token in tokens:
                bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.embedding_dim
                sign = 1.0 if int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % 2 == 0 else -1.0
                embeddings[index, bucket] += sign
            embeddings[index] /= math.sqrt(len(tokens))
        return embeddings


class TransformerTextEmbedder:
    def __init__(
        self,
        *,
        model_name: str,
        device: torch.device,
        max_tokens: int,
        local_files_only: bool,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_tokens = int(max_tokens)
        self.embedding_dim = int(self.model.config.hidden_size)
        self.backend_name = f"transformer:{model_name}"

    def encode_texts(self, texts: Sequence[str], batch_size: int = 8) -> np.ndarray:
        rows: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = [str(text) for text in texts[start : start + batch_size]]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                rows.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(rows) if rows else np.zeros((0, self.embedding_dim), dtype=np.float32)


def build_text_tokenizer(config: dict):
    from transformers import AutoTokenizer

    text_cfg = config["text_processing"]
    model_name = text_cfg.get("pretrained_text_model_name")
    if not model_name:
        raise ValueError("A pretrained_text_model_name is required for BERT fine-tuning.")

    local_files_only = bool(text_cfg.get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    return tokenizer


def build_text_embedder(config: dict, device: torch.device):
    text_cfg = config["text_processing"]
    backend = str(text_cfg.get("embedding_backend", "auto")).lower()
    model_name = text_cfg.get("pretrained_text_model_name")
    max_tokens = int(text_cfg.get("tokenizer_max_length", config["multimodal"].get("max_tokens_per_window", 128)))
    local_files_only = bool(text_cfg.get("local_files_only", False))

    if backend in {"auto", "transformer", "clinicalbert"} and model_name:
        try:
            return TransformerTextEmbedder(
                model_name=model_name,
                device=device,
                max_tokens=max_tokens,
                local_files_only=local_files_only,
            )
        except Exception:
            if backend != "auto":
                raise

    embedding_dim = int(config["multimodal"]["text_embedding_dim"])
    return HashingTextEmbedder(embedding_dim=embedding_dim)


def _select_feature_columns(structured_df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in structured_df.columns
        if column not in EXCLUDED_STRUCTURED_COLUMNS
        and pd.api.types.is_numeric_dtype(structured_df[column])
    ]


def _normalize_structured_features(structured_df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    normalized = structured_df.copy()
    train_rows = normalized.loc[normalized["split"] == "train", list(feature_columns)]
    medians = train_rows.median(numeric_only=True).fillna(0.0)
    means = train_rows.fillna(medians).mean(numeric_only=True)
    stds = train_rows.fillna(medians).std(numeric_only=True).replace(0.0, 1.0).fillna(1.0)

    filled = normalized.loc[:, feature_columns].fillna(medians)
    normalized.loc[:, feature_columns] = (filled - means) / stds
    normalized.loc[:, feature_columns] = normalized.loc[:, feature_columns].fillna(0.0)
    return normalized


def _select_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str = "f1",
    default_threshold: float = 0.5,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float(default_threshold)

    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 19, dtype=np.float32),
                y_prob.astype(np.float32),
                np.asarray([default_threshold], dtype=np.float32),
            ]
        )
    )

    best_threshold = float(default_threshold)
    best_score = float("-inf")
    for threshold in candidate_thresholds:
        metrics = compute_binary_classification_metrics(y_true, y_prob, threshold=float(threshold))
        score = float(metrics.get(metric_name, float("nan")))
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def _build_structured_summary_vector(
    sequence: np.ndarray,
    *,
    raw_sequence: np.ndarray | None = None,
    aggregations: Sequence[str] = ("mean", "min", "max", "last"),
    include_missingness: bool = True,
) -> tuple[np.ndarray, list[str]]:
    if sequence.ndim != 2:
        raise ValueError("Structured summary expects a 2D [time, features] sequence.")

    features = []
    suffixes = []
    if "mean" in aggregations:
        features.append(sequence.mean(axis=0))
        suffixes.extend(["mean"] * sequence.shape[1])
    if "min" in aggregations:
        features.append(sequence.min(axis=0))
        suffixes.extend(["min"] * sequence.shape[1])
    if "max" in aggregations:
        features.append(sequence.max(axis=0))
        suffixes.extend(["max"] * sequence.shape[1])
    if "last" in aggregations:
        features.append(sequence[-1])
        suffixes.extend(["last"] * sequence.shape[1])

    if include_missingness:
        if raw_sequence is None:
            missing_rate = np.zeros(sequence.shape[1], dtype=np.float32)
        else:
            missing_rate = np.isnan(raw_sequence).mean(axis=0).astype(np.float32)
        features.append(missing_rate)
        suffixes.extend(["missing_rate"] * sequence.shape[1])

    summary = np.concatenate(features, axis=0).astype(np.float32) if features else np.zeros(0, dtype=np.float32)
    return summary, suffixes


def _build_structured_summary_column_names(
    feature_columns: Sequence[str],
    aggregations: Sequence[str],
    include_missingness: bool,
) -> list[str]:
    names: list[str] = []
    for suffix in aggregations:
        names.extend([f"{column}__{suffix}" for column in feature_columns])
    if include_missingness:
        names.extend([f"{column}__missing_rate" for column in feature_columns])
    return names


def prepare_multimodal_dataset(
    *,
    structured_df: pd.DataFrame,
    text_df: pd.DataFrame,
    config: dict,
    device: torch.device | None = None,
) -> PreparedMultimodalDataset:
    feature_columns = _select_feature_columns(structured_df)
    if not feature_columns:
        raise ValueError("No numeric structured feature columns were found for multimodal training.")

    raw_structured_df = structured_df.copy()
    structured_df = _normalize_structured_features(structured_df, feature_columns)
    stay_index = structured_df[ID_COLUMNS + ["split", "sepsis3_label"]].drop_duplicates().copy()
    stay_index = stay_index.sort_values(ID_COLUMNS).reset_index(drop=True)

    resolved_device = device or resolve_device(config["multimodal"].get("device", "auto"))
    text_input_mode = str(config["multimodal"].get("text_encoder_mode", "frozen_embedding")).lower()
    max_tokens = int(config["text_processing"].get("tokenizer_max_length", config["multimodal"].get("max_tokens_per_window", 128)))

    max_structured_steps = int(config["multimodal"]["max_structured_steps"])
    max_text_windows = int(config["multimodal"]["max_text_windows"])
    summary_cfg = config["multimodal"].get("structured_summary", {})
    summary_enabled = bool(summary_cfg.get("enabled", True))
    summary_aggregations = tuple(summary_cfg.get("aggregations", ["mean", "min", "max", "last"]))
    include_missingness = bool(summary_cfg.get("include_missingness", True))

    text_df = text_df.copy()
    if "aggregated_text" not in text_df.columns:
        text_df["aggregated_text"] = ""
    if "note_window_index" not in text_df.columns:
        text_df["note_window_index"] = 0
    text_df["aggregated_text"] = text_df["aggregated_text"].fillna("").astype(str)
    text_df["note_window_index"] = pd.to_numeric(text_df["note_window_index"], errors="coerce").fillna(0).astype(int)
    text_df = apply_configured_keyword_masking(text_df, config, text_column="aggregated_text")
    text_df = text_df.sort_values(ID_COLUMNS + ["note_window_index"]).reset_index(drop=True)

    text_embedding_table = pd.DataFrame(columns=ID_COLUMNS + ["note_window_index"])
    text_token_table = pd.DataFrame(columns=ID_COLUMNS + ["note_window_index"])
    if text_input_mode == "bert_finetune":
        tokenizer = build_text_tokenizer(config)
        if text_df.empty:
            embedding_dim = int(config["multimodal"]["text_embedding_dim"])
        else:
            tokenized = tokenizer(
                text_df["aggregated_text"].tolist(),
                padding="max_length",
                truncation=True,
                max_length=max_tokens,
                return_attention_mask=True,
            )
            token_input_ids = np.asarray(tokenized["input_ids"], dtype=np.int64)
            token_attention_masks = np.asarray(tokenized["attention_mask"], dtype=np.int64)
            text_token_table = pd.DataFrame(
                {
                    **{column: text_df[column].to_numpy() for column in ID_COLUMNS + ["note_window_index"]},
                    "token_input_ids": list(token_input_ids),
                    "token_attention_mask": list(token_attention_masks),
                }
            )
            embedding_dim = int(config["multimodal"]["text_embedding_dim"])
    else:
        text_embedder = build_text_embedder(config, resolved_device)
        if text_df.empty:
            embedding_dim = int(getattr(text_embedder, "embedding_dim", config["multimodal"]["text_embedding_dim"]))
        else:
            encoded = text_embedder.encode_texts(
                text_df["aggregated_text"].tolist(),
                batch_size=int(config["text_processing"].get("embedding_batch_size", 8)),
            )
            embedding_dim = int(encoded.shape[1]) if encoded.size else int(getattr(text_embedder, "embedding_dim", 0))
            embedding_columns = [f"embedding_{index}" for index in range(embedding_dim)]
            text_embedding_table = pd.concat(
                [
                    text_df[ID_COLUMNS + ["note_window_index"]].reset_index(drop=True),
                    pd.DataFrame(encoded, columns=embedding_columns),
                ],
                axis=1,
            )
        text_embedder_backend = text_embedder.backend_name

    if text_input_mode == "bert_finetune":
        model_name = config["text_processing"].get("pretrained_text_model_name")
        text_embedder_backend = f"finetune:{model_name}"

    structured_groups = {
        key: group.sort_values("hour")
        for key, group in structured_df.groupby(ID_COLUMNS, sort=False)
    }
    raw_structured_groups = {
        key: group.sort_values("hour")
        for key, group in raw_structured_df.groupby(ID_COLUMNS, sort=False)
    }
    text_groups = {
        key: group.sort_values("note_window_index")
        for key, group in text_embedding_table.groupby(ID_COLUMNS, sort=False)
    }
    text_token_groups = {
        key: group.sort_values("note_window_index")
        for key, group in text_token_table.groupby(ID_COLUMNS, sort=False)
    }

    structured_sequences = []
    text_sequences = []
    text_token_ids = []
    text_token_attention_masks = []
    structured_summary_features = []
    structured_padding_masks = []
    text_padding_masks = []
    labels = []
    filtered_stays = []
    structured_summary_columns: list[str] | None = None

    for stay in stay_index.itertuples(index=False):
        key = (stay.SUBJECT_ID, stay.HADM_ID, stay.ICUSTAY_ID)
        structured_rows = structured_groups.get(key)
        if structured_rows is None or structured_rows.empty:
            continue

        sequence = structured_rows.loc[:, feature_columns].to_numpy(dtype=np.float32)
        sequence = sequence[-max_structured_steps:]
        raw_structured_rows = raw_structured_groups.get(key)
        raw_sequence = None
        if raw_structured_rows is not None and not raw_structured_rows.empty:
            raw_sequence = raw_structured_rows.loc[:, feature_columns].to_numpy(dtype=np.float32)
            raw_sequence = raw_sequence[-len(sequence) :]
        structured_tensor = np.zeros((max_structured_steps, len(feature_columns)), dtype=np.float32)
        structured_padding_mask = np.ones(max_structured_steps, dtype=bool)
        structured_tensor[: len(sequence)] = sequence
        structured_padding_mask[: len(sequence)] = False

        if summary_enabled:
            summary_vector, _ = _build_structured_summary_vector(
                sequence,
                raw_sequence=raw_sequence,
                aggregations=summary_aggregations,
                include_missingness=include_missingness,
            )
            if structured_summary_columns is None:
                structured_summary_columns = _build_structured_summary_column_names(
                    feature_columns,
                    summary_aggregations,
                    include_missingness,
                )
        else:
            summary_vector = np.zeros((0,), dtype=np.float32)

        note_rows = text_groups.get(key)
        note_tensor = np.zeros((1, 1), dtype=np.float32) if text_input_mode == "bert_finetune" else np.zeros((max_text_windows, embedding_dim), dtype=np.float32)
        token_id_tensor = np.zeros((1, 1), dtype=np.int64)
        token_attention_tensor = np.zeros((1, 1), dtype=np.int64)
        text_padding_mask = np.ones(max_text_windows, dtype=bool)
        if text_input_mode == "bert_finetune":
            note_token_rows = text_token_groups.get(key)
            token_id_tensor = np.zeros((max_text_windows, max_tokens), dtype=np.int64)
            token_attention_tensor = np.zeros((max_text_windows, max_tokens), dtype=np.int64)
            if note_token_rows is not None and not note_token_rows.empty:
                token_ids = np.stack(note_token_rows["token_input_ids"].to_list()).astype(np.int64)
                token_masks = np.stack(note_token_rows["token_attention_mask"].to_list()).astype(np.int64)
                token_ids = token_ids[:max_text_windows][::-1].copy()
                token_masks = token_masks[:max_text_windows][::-1].copy()
                token_id_tensor[: len(token_ids)] = token_ids
                token_attention_tensor[: len(token_masks)] = token_masks
                text_padding_mask[: len(token_ids)] = False
        else:
            if note_rows is not None and not note_rows.empty and embedding_dim > 0:
                embedding_values = note_rows.drop(columns=ID_COLUMNS + ["note_window_index"]).to_numpy(dtype=np.float32)
                embedding_values = embedding_values[:max_text_windows]
                embedding_values = embedding_values[::-1].copy()
                note_tensor[: len(embedding_values)] = embedding_values
                text_padding_mask[: len(embedding_values)] = False
            else:
                text_padding_mask[0] = False
            token_id_tensor = np.zeros((1, 1), dtype=np.int64)
            token_attention_tensor = np.zeros((1, 1), dtype=np.int64)

        structured_sequences.append(structured_tensor)
        text_sequences.append(note_tensor)
        text_token_ids.append(token_id_tensor)
        text_token_attention_masks.append(token_attention_tensor)
        structured_summary_features.append(summary_vector)
        structured_padding_masks.append(structured_padding_mask)
        text_padding_masks.append(text_padding_mask)
        labels.append(int(stay.sepsis3_label))
        filtered_stays.append(
            {
                "SUBJECT_ID": stay.SUBJECT_ID,
                "HADM_ID": stay.HADM_ID,
                "ICUSTAY_ID": stay.ICUSTAY_ID,
                "split": stay.split,
                "sepsis3_label": stay.sepsis3_label,
            }
        )

    if not structured_sequences:
        raise ValueError("No multimodal stay examples were created from the prepared structured/text artifacts.")

    return PreparedMultimodalDataset(
        stay_index=pd.DataFrame(filtered_stays).reset_index(drop=True),
        structured_sequences=np.stack(structured_sequences).astype(np.float32),
        text_embeddings=np.stack(text_sequences).astype(np.float32),
        text_token_ids=np.stack(text_token_ids).astype(np.int64),
        text_token_attention_masks=np.stack(text_token_attention_masks).astype(np.int64),
        structured_summary_features=np.stack(structured_summary_features).astype(np.float32),
        structured_padding_masks=np.stack(structured_padding_masks).astype(bool),
        text_padding_masks=np.stack(text_padding_masks).astype(bool),
        labels=np.asarray(labels, dtype=np.float32),
        feature_columns=list(feature_columns),
        structured_summary_columns=list(structured_summary_columns or []),
        text_embedding_dim=embedding_dim,
        text_embedding_backend=text_embedder_backend,
        text_input_mode=text_input_mode,
    )


def _split_indices(stay_index: pd.DataFrame, split_name: str) -> np.ndarray:
    return np.flatnonzero(stay_index["split"].astype(str).to_numpy() == split_name)


def _build_loader(
    structured_sequences: np.ndarray,
    text_embeddings: np.ndarray,
    text_token_ids: np.ndarray,
    text_token_attention_masks: np.ndarray,
    structured_summary_features: np.ndarray,
    structured_padding_masks: np.ndarray,
    text_padding_masks: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(structured_sequences[indices], dtype=torch.float32),
        torch.tensor(text_embeddings[indices], dtype=torch.float32),
        torch.tensor(text_token_ids[indices], dtype=torch.long),
        torch.tensor(text_token_attention_masks[indices], dtype=torch.long),
        torch.tensor(structured_summary_features[indices], dtype=torch.float32),
        torch.tensor(structured_padding_masks[indices], dtype=torch.bool),
        torch.tensor(text_padding_masks[indices], dtype=torch.bool),
        torch.tensor(labels[indices], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    losses = []
    probs = []
    labels = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for structured_batch, text_batch, text_token_ids_batch, text_token_attention_mask_batch, summary_batch, structured_mask_batch, text_mask_batch, label_batch in loader:
            structured_batch = structured_batch.to(device)
            text_batch = text_batch.to(device)
            text_token_ids_batch = text_token_ids_batch.to(device)
            text_token_attention_mask_batch = text_token_attention_mask_batch.to(device)
            summary_batch = summary_batch.to(device)
            structured_mask_batch = structured_mask_batch.to(device)
            text_mask_batch = text_mask_batch.to(device)
            label_batch = label_batch.to(device)

            logits, _ = model(
                structured_batch,
                text_window_embeddings=text_batch,
                text_input_ids=text_token_ids_batch,
                text_token_attention_mask=text_token_attention_mask_batch,
                structured_summary_features=summary_batch,
                structured_padding_mask=structured_mask_batch,
                text_padding_mask=text_mask_batch,
            )
            loss = criterion(logits, label_batch)
            losses.append(float(loss.detach().cpu()))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(label_batch.detach().cpu().numpy())

    if not probs:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), float("nan")
    return (
        np.concatenate(probs).astype(np.float32),
        np.concatenate(labels).astype(np.float32),
        float(np.mean(losses)),
    )


def _write_live_training_artifacts(
    *,
    output_dir: Path,
    dataset_name: str,
    history_rows: list[dict],
    progress_payload: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / f"{dataset_name}_multimodal_training_history_live.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    progress_path = output_dir / f"{dataset_name}_multimodal_progress.json"
    progress_path.write_text(json.dumps(progress_payload, indent=2, sort_keys=True), encoding="utf-8")


def train_multimodal_models(
    *,
    prepared: PreparedMultimodalDataset,
    config: dict,
    output_dir: str | Path,
    dataset_name: str = "multimodal",
) -> Dict[str, object]:
    seed = int(config["project"]["seed"])
    set_random_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config["multimodal"].get("device", "auto"))
    batch_size = int(config["multimodal"]["batch_size"])
    epochs = int(config["multimodal"]["epochs"])
    learning_rate = float(config["multimodal"]["learning_rate"])
    weight_decay = float(config["multimodal"].get("weight_decay", 1e-4))
    gradient_clip_norm = float(config["multimodal"].get("gradient_clip_norm", 1.0))
    scheduler_name = str(config["multimodal"].get("scheduler", "cosine")).lower()
    min_learning_rate = float(config["multimodal"].get("min_learning_rate", learning_rate * 0.1))
    early_stopping_patience = int(config["multimodal"].get("early_stopping_patience", 5))
    threshold_selection_metric = str(config["multimodal"].get("threshold_selection_metric", "f1"))

    train_indices = _split_indices(prepared.stay_index, "train")
    val_indices = _split_indices(prepared.stay_index, "val")
    test_indices = _split_indices(prepared.stay_index, "test")

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Multimodal training requires non-empty train and test splits.")

    selection_indices = val_indices if len(val_indices) > 0 else test_indices
    train_loader = _build_loader(
        prepared.structured_sequences,
        prepared.text_embeddings,
        prepared.text_token_ids,
        prepared.text_token_attention_masks,
        prepared.structured_summary_features,
        prepared.structured_padding_masks,
        prepared.text_padding_masks,
        prepared.labels,
        train_indices,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = _build_loader(
        prepared.structured_sequences,
        prepared.text_embeddings,
        prepared.text_token_ids,
        prepared.text_token_attention_masks,
        prepared.structured_summary_features,
        prepared.structured_padding_masks,
        prepared.text_padding_masks,
        prepared.labels,
        selection_indices,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = _build_loader(
        prepared.structured_sequences,
        prepared.text_embeddings,
        prepared.text_token_ids,
        prepared.text_token_attention_masks,
        prepared.structured_summary_features,
        prepared.structured_padding_masks,
        prepared.text_padding_masks,
        prepared.labels,
        test_indices,
        batch_size=batch_size,
        shuffle=False,
    )

    positive_count = float(prepared.labels[train_indices].sum())
    negative_count = float(len(train_indices) - positive_count)
    if positive_count > 0 and negative_count > 0:
        pos_weight = torch.tensor([negative_count / positive_count], device=device, dtype=torch.float32)
    else:
        pos_weight = torch.tensor([1.0], device=device, dtype=torch.float32)

    result_rows = []
    history_rows = []
    checkpoint_paths: Dict[str, str] = {}
    artifact_tables: Dict[str, pd.DataFrame] = {}
    fusion_strategies = list(config["multimodal"]["fusion_strategies"])

    for strategy_index, fusion_strategy in enumerate(fusion_strategies, start=1):
        model = MultimodalClassifier(
            structured_input_dim=len(prepared.feature_columns),
            text_embedding_dim=prepared.text_embedding_dim,
            structured_summary_dim=prepared.structured_summary_features.shape[1],
            hidden_dim=int(config["multimodal"]["hidden_dim"]),
            dropout=float(config["multimodal"]["dropout"]),
            structured_encoder_type=config["multimodal"]["structured_encoder"],
            text_encoder_mode=prepared.text_input_mode,
            text_model_name=config["text_processing"].get("pretrained_text_model_name"),
            text_local_files_only=bool(config["text_processing"].get("local_files_only", False)),
            text_finetune_unfrozen_layers=int(config["multimodal"].get("text_finetune_unfrozen_layers", 2)),
            text_gradient_checkpointing=bool(config["multimodal"].get("text_gradient_checkpointing", False)),
            fusion_strategy=fusion_strategy,
            structured_num_heads=int(config["multimodal"].get("structured_num_heads", 4)),
            structured_num_layers=int(config["multimodal"].get("structured_num_layers", 2)),
            text_num_heads=int(config["multimodal"].get("text_num_heads", 4)),
            text_num_layers=int(config["multimodal"].get("text_num_layers", 1)),
            fusion_num_heads=int(config["multimodal"].get("fusion_num_heads", 4)),
        ).to(device)

        text_finetune_learning_rate = float(config["multimodal"].get("text_finetune_learning_rate", 2e-5))
        text_window_embedder = getattr(model, "text_window_embedder", None)
        if text_window_embedder is not None and getattr(text_window_embedder, "is_finetuning_enabled", False):
            text_params = [parameter for parameter in text_window_embedder.parameters() if parameter.requires_grad]
            text_param_ids = {id(parameter) for parameter in text_params}
            other_params = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad and id(parameter) not in text_param_ids
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "lr": learning_rate},
                    {"params": text_params, "lr": text_finetune_learning_rate},
                ],
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = None
        if scheduler_name == "cosine" and epochs > 1:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=min_learning_rate,
            )
        elif scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=2,
                min_lr=min_learning_rate,
            )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_metric = float("-inf")
        best_state = copy.deepcopy(model.state_dict())
        best_threshold = float(config.get("evaluation", {}).get("default_threshold", 0.5))
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            model.train()
            batch_losses = []
            for structured_batch, text_batch, text_token_ids_batch, text_token_attention_mask_batch, summary_batch, structured_mask_batch, text_mask_batch, label_batch in train_loader:
                structured_batch = structured_batch.to(device)
                text_batch = text_batch.to(device)
                text_token_ids_batch = text_token_ids_batch.to(device)
                text_token_attention_mask_batch = text_token_attention_mask_batch.to(device)
                summary_batch = summary_batch.to(device)
                structured_mask_batch = structured_mask_batch.to(device)
                text_mask_batch = text_mask_batch.to(device)
                label_batch = label_batch.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(
                    structured_batch,
                    text_window_embeddings=text_batch,
                    text_input_ids=text_token_ids_batch,
                    text_token_attention_mask=text_token_attention_mask_batch,
                    structured_summary_features=summary_batch,
                    structured_padding_mask=structured_mask_batch,
                    text_padding_mask=text_mask_batch,
                )
                loss = criterion(logits, label_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))

            val_prob, val_true, val_loss = _predict_model(model, val_loader, device)
            val_metrics = compute_binary_classification_metrics(val_true, val_prob) if len(val_true) else {}
            selection_score = float(val_metrics.get("auprc", float("nan")))
            if np.isnan(selection_score):
                selection_score = float(val_metrics.get("auroc", float("nan")))
            if np.isnan(selection_score):
                selection_score = -float(val_loss) if not np.isnan(val_loss) else float("-inf")

            if selection_score > best_metric:
                best_metric = selection_score
                best_state = copy.deepcopy(model.state_dict())
                best_threshold = _select_decision_threshold(
                    val_true,
                    val_prob,
                    metric_name=threshold_selection_metric,
                    default_threshold=float(config.get("evaluation", {}).get("default_threshold", 0.5)),
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(selection_score)
                else:
                    scheduler.step()

            history_rows.append(
                {
                    "fusion_strategy": fusion_strategy,
                    "epoch": epoch,
                    "train_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "selection_loss": val_loss,
                    "selection_auroc": float(val_metrics.get("auroc", float("nan"))),
                    "selection_auprc": float(val_metrics.get("auprc", float("nan"))),
                    "selection_precision": float(val_metrics.get("precision", float("nan"))),
                    "selection_recall": float(val_metrics.get("recall", float("nan"))),
                    "selection_f1": float(val_metrics.get("f1", float("nan"))),
                    "decision_threshold": float(best_threshold),
                }
            )

            _write_live_training_artifacts(
                output_dir=output_dir,
                dataset_name=dataset_name,
                history_rows=history_rows,
                progress_payload={
                    "dataset_name": dataset_name,
                    "status": "training",
                    "strategy_index": int(strategy_index),
                    "strategy_count": int(len(fusion_strategies)),
                    "current_fusion_strategy": fusion_strategy,
                    "current_epoch": int(epoch),
                    "total_epochs": int(epochs),
                    "best_selection_score": float(best_metric),
                    "best_decision_threshold": float(best_threshold),
                    "latest_train_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
                    "latest_selection_loss": float(val_loss) if not np.isnan(val_loss) else None,
                    "latest_selection_auroc": float(val_metrics.get("auroc", float("nan"))),
                    "latest_selection_auprc": float(val_metrics.get("auprc", float("nan"))),
                    "latest_selection_f1": float(val_metrics.get("f1", float("nan"))),
                    "completed_fusion_strategies": fusion_strategies[: strategy_index - 1],
                },
            )

            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                break

        model.load_state_dict(best_state)
        checkpoint_path = output_dir / f"{dataset_name}_{fusion_strategy}_best.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "fusion_strategy": fusion_strategy,
                "structured_encoder": config["multimodal"]["structured_encoder"],
                "feature_columns": prepared.feature_columns,
                "text_embedding_dim": prepared.text_embedding_dim,
                "text_embedding_backend": prepared.text_embedding_backend,
                "config": config,
            },
            checkpoint_path,
        )
        checkpoint_paths[fusion_strategy] = str(checkpoint_path)

        for split_name, indices, loader in [
            (
                "val",
                val_indices,
                _build_loader(
                    prepared.structured_sequences,
                    prepared.text_embeddings,
                    prepared.text_token_ids,
                    prepared.text_token_attention_masks,
                    prepared.structured_summary_features,
                    prepared.structured_padding_masks,
                    prepared.text_padding_masks,
                    prepared.labels,
                    val_indices,
                    batch_size,
                    False,
                ) if len(val_indices) else None,
            ),
            ("test", test_indices, test_loader),
        ]:
            if loader is None or len(indices) == 0:
                continue

            split_prob, split_true, split_loss = _predict_model(model, loader, device)
            metrics = compute_binary_classification_metrics(split_true, split_prob, threshold=best_threshold)
            result_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split": split_name,
                    "model_name": fusion_strategy,
                    "structured_encoder": config["multimodal"]["structured_encoder"],
                    "text_embedding_backend": prepared.text_embedding_backend,
                    "device": str(device),
                    "n_features": len(prepared.feature_columns),
                    "n_summary_features": int(prepared.structured_summary_features.shape[1]),
                    "n_examples": int(len(split_true)),
                    "loss": split_loss,
                    "decision_threshold": float(best_threshold),
                    **metrics,
                }
            )

            predictions = prepared.stay_index.iloc[indices][ID_COLUMNS].reset_index(drop=True).copy()
            predictions["y_true"] = split_true.astype(int)
            predictions["y_prob"] = split_prob.astype(float)
            predictions["y_pred"] = (predictions["y_prob"] >= float(best_threshold)).astype(int)
            predictions["decision_threshold"] = float(best_threshold)
            predictions["dataset_name"] = dataset_name
            predictions["model_name"] = fusion_strategy
            predictions["structured_encoder"] = config["multimodal"]["structured_encoder"]
            artifact_tables[f"{dataset_name}_{fusion_strategy}_{split_name}_predictions"] = predictions

        _write_live_training_artifacts(
            output_dir=output_dir,
            dataset_name=dataset_name,
            history_rows=history_rows,
            progress_payload={
                "dataset_name": dataset_name,
                "status": "strategy_completed",
                "strategy_index": int(strategy_index),
                "strategy_count": int(len(fusion_strategies)),
                "current_fusion_strategy": fusion_strategy,
                "current_epoch": int(history_rows[-1]["epoch"]) if history_rows else 0,
                "total_epochs": int(epochs),
                "best_decision_threshold": float(best_threshold),
                "completed_fusion_strategies": fusion_strategies[:strategy_index],
                "latest_result_rows": int(len(result_rows)),
            },
        )

    experiment_plan = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "structured_encoder": config["multimodal"]["structured_encoder"],
                "fusion_strategy": fusion_strategy,
                "text_embedding_backend": prepared.text_embedding_backend,
                "text_embedding_dim": prepared.text_embedding_dim,
                "structured_summary_dim": int(prepared.structured_summary_features.shape[1]),
                "n_train": int(len(train_indices)),
                "n_val": int(len(val_indices)),
                "n_test": int(len(test_indices)),
                "device": str(device),
                "scheduler": scheduler_name,
                "threshold_selection_metric": threshold_selection_metric,
            }
            for fusion_strategy in config["multimodal"]["fusion_strategies"]
        ]
    )

    artifact_tables[f"{dataset_name}_multimodal_results"] = pd.DataFrame(result_rows)
    artifact_tables[f"{dataset_name}_multimodal_training_history"] = pd.DataFrame(history_rows)
    artifact_tables[f"{dataset_name}_multimodal_experiment_plan"] = experiment_plan
    artifact_tables[f"{dataset_name}_multimodal_stay_index"] = prepared.stay_index.copy()

    _write_live_training_artifacts(
        output_dir=output_dir,
        dataset_name=dataset_name,
        history_rows=history_rows,
        progress_payload={
            "dataset_name": dataset_name,
            "status": "completed",
            "strategy_index": int(len(fusion_strategies)),
            "strategy_count": int(len(fusion_strategies)),
            "current_fusion_strategy": None,
            "current_epoch": int(history_rows[-1]["epoch"]) if history_rows else 0,
            "total_epochs": int(epochs),
            "completed_fusion_strategies": fusion_strategies,
            "result_row_count": int(len(result_rows)),
        },
    )

    return {
        "artifacts": artifact_tables,
        "checkpoint_paths": checkpoint_paths,
        "device": str(device),
        "text_embedding_backend": prepared.text_embedding_backend,
        "feature_columns": prepared.feature_columns,
        "structured_summary_columns": prepared.structured_summary_columns,
    }
