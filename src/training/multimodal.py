from __future__ import annotations

import copy
import hashlib
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
    labels: np.ndarray
    feature_columns: List[str]
    text_embedding_dim: int
    text_embedding_backend: str


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

    structured_df = _normalize_structured_features(structured_df, feature_columns)
    stay_index = structured_df[ID_COLUMNS + ["split", "sepsis3_label"]].drop_duplicates().copy()
    stay_index = stay_index.sort_values(ID_COLUMNS).reset_index(drop=True)

    resolved_device = device or resolve_device(config["multimodal"].get("device", "auto"))
    text_embedder = build_text_embedder(config, resolved_device)

    max_structured_steps = int(config["multimodal"]["max_structured_steps"])
    max_text_windows = int(config["multimodal"]["max_text_windows"])

    if text_df.empty:
        text_embedding_table = pd.DataFrame(columns=ID_COLUMNS + ["note_window_index"])
        embedding_dim = int(getattr(text_embedder, "embedding_dim", config["multimodal"]["text_embedding_dim"]))
    else:
        text_df = text_df.copy()
        if "aggregated_text" not in text_df.columns:
            text_df["aggregated_text"] = ""
        if "note_window_index" not in text_df.columns:
            text_df["note_window_index"] = 0
        text_df["aggregated_text"] = text_df["aggregated_text"].fillna("").astype(str)
        text_df["note_window_index"] = pd.to_numeric(text_df["note_window_index"], errors="coerce").fillna(0).astype(int)
        text_df = text_df.sort_values(ID_COLUMNS + ["note_window_index"]).reset_index(drop=True)
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

    structured_groups = {
        key: group.sort_values("hour")
        for key, group in structured_df.groupby(ID_COLUMNS, sort=False)
    }
    text_groups = {
        key: group.sort_values("note_window_index")
        for key, group in text_embedding_table.groupby(ID_COLUMNS, sort=False)
    }

    structured_sequences = []
    text_sequences = []
    labels = []
    filtered_stays = []

    for stay in stay_index.itertuples(index=False):
        key = (stay.SUBJECT_ID, stay.HADM_ID, stay.ICUSTAY_ID)
        structured_rows = structured_groups.get(key)
        if structured_rows is None or structured_rows.empty:
            continue

        sequence = structured_rows.loc[:, feature_columns].to_numpy(dtype=np.float32)
        sequence = sequence[-max_structured_steps:]
        if len(sequence) < max_structured_steps:
            pad = np.zeros((max_structured_steps - len(sequence), len(feature_columns)), dtype=np.float32)
            sequence = np.vstack([pad, sequence])

        note_rows = text_groups.get(key)
        note_tensor = np.zeros((max_text_windows, embedding_dim), dtype=np.float32)
        if note_rows is not None and not note_rows.empty and embedding_dim > 0:
            embedding_values = note_rows.drop(columns=ID_COLUMNS + ["note_window_index"]).to_numpy(dtype=np.float32)
            embedding_values = embedding_values[:max_text_windows]
            note_tensor[: len(embedding_values)] = embedding_values

        structured_sequences.append(sequence)
        text_sequences.append(note_tensor)
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
        labels=np.asarray(labels, dtype=np.float32),
        feature_columns=list(feature_columns),
        text_embedding_dim=embedding_dim,
        text_embedding_backend=text_embedder.backend_name,
    )


def _split_indices(stay_index: pd.DataFrame, split_name: str) -> np.ndarray:
    return np.flatnonzero(stay_index["split"].astype(str).to_numpy() == split_name)


def _build_loader(
    structured_sequences: np.ndarray,
    text_embeddings: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(structured_sequences[indices], dtype=torch.float32),
        torch.tensor(text_embeddings[indices], dtype=torch.float32),
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
        for structured_batch, text_batch, label_batch in loader:
            structured_batch = structured_batch.to(device)
            text_batch = text_batch.to(device)
            label_batch = label_batch.to(device)

            logits, _ = model(structured_batch, text_batch)
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

    train_indices = _split_indices(prepared.stay_index, "train")
    val_indices = _split_indices(prepared.stay_index, "val")
    test_indices = _split_indices(prepared.stay_index, "test")

    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Multimodal training requires non-empty train and test splits.")

    selection_indices = val_indices if len(val_indices) > 0 else test_indices
    train_loader = _build_loader(
        prepared.structured_sequences,
        prepared.text_embeddings,
        prepared.labels,
        train_indices,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = _build_loader(
        prepared.structured_sequences,
        prepared.text_embeddings,
        prepared.labels,
        selection_indices,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = _build_loader(
        prepared.structured_sequences,
        prepared.text_embeddings,
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

    for fusion_strategy in config["multimodal"]["fusion_strategies"]:
        model = MultimodalClassifier(
            structured_input_dim=len(prepared.feature_columns),
            text_embedding_dim=prepared.text_embedding_dim,
            hidden_dim=int(config["multimodal"]["hidden_dim"]),
            dropout=float(config["multimodal"]["dropout"]),
            structured_encoder_type=config["multimodal"]["structured_encoder"],
            fusion_strategy=fusion_strategy,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_metric = float("-inf")
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(1, epochs + 1):
            model.train()
            batch_losses = []
            for structured_batch, text_batch, label_batch in train_loader:
                structured_batch = structured_batch.to(device)
                text_batch = text_batch.to(device)
                label_batch = label_batch.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(structured_batch, text_batch)
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

            history_rows.append(
                {
                    "fusion_strategy": fusion_strategy,
                    "epoch": epoch,
                    "train_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
                    "selection_loss": val_loss,
                    "selection_auroc": float(val_metrics.get("auroc", float("nan"))),
                    "selection_auprc": float(val_metrics.get("auprc", float("nan"))),
                    "selection_precision": float(val_metrics.get("precision", float("nan"))),
                    "selection_recall": float(val_metrics.get("recall", float("nan"))),
                    "selection_f1": float(val_metrics.get("f1", float("nan"))),
                }
            )

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
            ("val", val_indices, _build_loader(prepared.structured_sequences, prepared.text_embeddings, prepared.labels, val_indices, batch_size, False) if len(val_indices) else None),
            ("test", test_indices, test_loader),
        ]:
            if loader is None or len(indices) == 0:
                continue

            split_prob, split_true, split_loss = _predict_model(model, loader, device)
            metrics = compute_binary_classification_metrics(split_true, split_prob)
            result_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split": split_name,
                    "model_name": fusion_strategy,
                    "structured_encoder": config["multimodal"]["structured_encoder"],
                    "text_embedding_backend": prepared.text_embedding_backend,
                    "device": str(device),
                    "n_features": len(prepared.feature_columns),
                    "n_examples": int(len(split_true)),
                    "loss": split_loss,
                    **metrics,
                }
            )

            predictions = prepared.stay_index.iloc[indices][ID_COLUMNS].reset_index(drop=True).copy()
            predictions["y_true"] = split_true.astype(int)
            predictions["y_prob"] = split_prob.astype(float)
            predictions["dataset_name"] = dataset_name
            predictions["model_name"] = fusion_strategy
            predictions["structured_encoder"] = config["multimodal"]["structured_encoder"]
            artifact_tables[f"{dataset_name}_{fusion_strategy}_{split_name}_predictions"] = predictions

    experiment_plan = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "structured_encoder": config["multimodal"]["structured_encoder"],
                "fusion_strategy": fusion_strategy,
                "text_embedding_backend": prepared.text_embedding_backend,
                "text_embedding_dim": prepared.text_embedding_dim,
                "n_train": int(len(train_indices)),
                "n_val": int(len(val_indices)),
                "n_test": int(len(test_indices)),
                "device": str(device),
            }
            for fusion_strategy in config["multimodal"]["fusion_strategies"]
        ]
    )

    artifact_tables[f"{dataset_name}_multimodal_results"] = pd.DataFrame(result_rows)
    artifact_tables[f"{dataset_name}_multimodal_training_history"] = pd.DataFrame(history_rows)
    artifact_tables[f"{dataset_name}_multimodal_experiment_plan"] = experiment_plan
    artifact_tables[f"{dataset_name}_multimodal_stay_index"] = prepared.stay_index.copy()

    return {
        "artifacts": artifact_tables,
        "checkpoint_paths": checkpoint_paths,
        "device": str(device),
        "text_embedding_backend": prepared.text_embedding_backend,
        "feature_columns": prepared.feature_columns,
    }
