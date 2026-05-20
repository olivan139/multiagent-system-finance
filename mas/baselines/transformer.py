from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..config import TransformerConfig


class _SentimentTorchDataset(TorchDataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class TransformerBaseline:
    """FinBERT-based sentiment classifier (zero-shot or fine-tuned)."""

    def __init__(self, config: TransformerConfig | None = None):
        self.config = config or TransformerConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
        )

        self._id2label: dict[int, str] = {int(k): v for k, v in self.model.config.id2label.items()}
        self._label2id: dict[str, int] = {v: k for k, v in self._id2label.items()}
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def _encode(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

    def _predict_batch(self, texts: list[str], batch_size: int | None = None) -> list[str]:
        """Run inference and map output IDs through the model's native labels."""
        bs = batch_size or self.config.batch_size
        self.model.to(self.device)
        self.model.eval()
        all_preds: list[str] = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            enc = self._encode(batch)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(self._id2label[p] for p in pred_ids)

        return all_preds

    def predict_pretrained(self, texts: list[str], batch_size: int | None = None) -> list[str]:
        """Zero-shot prediction using the model's pre-trained weights."""
        return self._predict_batch(texts, batch_size)

    def predict(self, texts: list[str], batch_size: int | None = None) -> list[str]:
        """Prediction (same as predict_pretrained; works after fine-tuning too)."""
        return self._predict_batch(texts, batch_size)

    def predict_proba(
        self,
        texts: list[str],
        batch_size: int | None = None,
        label_order: list[str] | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return softmax probabilities aligned to ``label_order``.

        If ``label_order`` is None, columns follow the model's native id order.
        Pass ``label_order=LABELS`` from ``mas.config`` to get the canonical
        ``[negative, neutral, positive]`` ordering used elsewhere in the project.
        """
        bs = batch_size or self.config.batch_size
        self.model.to(self.device)
        self.model.eval()
        probs_native: list[np.ndarray] = []

        from tqdm import tqdm

        n_batches = (len(texts) + bs - 1) // bs
        rng = range(0, len(texts), bs)
        iterator = (
            tqdm(rng, total=n_batches, desc=f"FinBERT[{self.device}]") if show_progress else rng
        )
        for i in iterator:
            batch = texts[i : i + bs]
            enc = self._encode(batch)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_native.append(p)

        probs = np.vstack(probs_native)
        if label_order is None:
            return probs

        col_idx = [self._label2id[label] for label in label_order]
        return probs[:, col_idx]

    def fine_tune(
        self,
        train_texts: list[str],
        train_labels: list[str],
        val_texts: list[str],
        val_labels: list[str],
        output_dir: str = "./transformer_output",
    ) -> None:
        """Fine-tune the pretrained model, keeping its native label mapping."""
        train_enc = self._encode(train_texts)
        val_enc = self._encode(val_texts)
        train_lab = [self._label2id[lab] for lab in train_labels]
        val_lab = [self._label2id[lab] for lab in val_labels]

        train_ds = _SentimentTorchDataset(train_enc, train_lab)
        val_ds = _SentimentTorchDataset(val_enc, val_lab)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            seed=42,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
        )
        trainer.train()
        self.model = trainer.model

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
