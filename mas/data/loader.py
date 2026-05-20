from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from ..config import DataConfig

_TWITTER_FIN_ID2LABEL = {0: "negative", 1: "positive", 2: "neutral"}
_PHRASEBANK_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

SEMEVAL2017_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "semeval2017"
SEMEVAL2017_THRESHOLD = 0.15

FIQA2018_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "fiqa2018"
FIQA2018_THRESHOLD = 0.15


@dataclass
class SentimentDataset:
    texts: list[str]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.texts[idx], self.labels[idx]


def load_financial_phrasebank(
    config: DataConfig | None = None,
) -> tuple[SentimentDataset, SentimentDataset, SentimentDataset]:
    """Load a financial sentiment dataset and split into train / val / test.

    Supports two HuggingFace datasets out of the box:
      - "zeroshot/twitter-financial-news-sentiment" (default, 9.5K+ tweets)
      - "financial_phrasebank" via the mirror "warwickai/financial_phrasebank_mirror"
    """
    if config is None:
        config = DataConfig()

    name = config.dataset_name

    if name in ("financial_phrasebank", "warwickai/financial_phrasebank_mirror"):
        return _load_phrasebank_mirror(config)

    if name in ("semeval2017", "semeval2017_task5"):
        return _load_semeval2017(config)

    if name in ("fiqa2018", "fiqa_2018", "fiqa"):
        return _load_fiqa2018(config)

    return _load_twitter_financial(config)


def _load_twitter_financial(
    config: DataConfig,
) -> tuple[SentimentDataset, SentimentDataset, SentimentDataset]:
    """Twitter Financial News Sentiment (bearish/bullish/neutral)."""
    ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    id2label = _TWITTER_FIN_ID2LABEL

    train_texts: list[str] = ds["train"]["text"]
    train_labels: list[str] = [id2label[i] for i in ds["train"]["label"]]
    val_texts: list[str] = ds["validation"]["text"]
    val_labels: list[str] = [id2label[i] for i in ds["validation"]["label"]]

    if config.max_samples is not None:
        train_texts = train_texts[: config.max_samples]
        train_labels = train_labels[: config.max_samples]

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        val_texts,
        val_labels,
        test_size=0.5,
        random_state=config.random_seed,
        stratify=val_labels,
    )

    return (
        SentimentDataset(train_texts, train_labels),
        SentimentDataset(val_texts, val_labels),
        SentimentDataset(test_texts, test_labels),
    )


def _load_phrasebank_mirror(
    config: DataConfig,
) -> tuple[SentimentDataset, SentimentDataset, SentimentDataset]:
    """Financial PhraseBank mirror (negative/neutral/positive)."""
    ds = load_dataset("warwickai/financial_phrasebank_mirror", split="train")
    id2label = _PHRASEBANK_ID2LABEL

    texts: list[str] = ds["sentence"]
    labels: list[str] = [id2label[i] for i in ds["label"]]

    if config.max_samples is not None:
        texts = texts[: config.max_samples]
        labels = labels[: config.max_samples]

    holdout_ratio = config.test_size + config.val_size
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=holdout_ratio,
        random_state=config.random_seed,
        stratify=labels,
    )

    val_frac = config.val_size / holdout_ratio
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=1 - val_frac,
        random_state=config.random_seed,
        stratify=temp_labels,
    )

    return (
        SentimentDataset(train_texts, train_labels),
        SentimentDataset(val_texts, val_labels),
        SentimentDataset(test_texts, test_labels),
    )


def _semeval_score_to_label(score: float, threshold: float) -> str:
    """Bin a SemEval-2017 continuous sentiment score to 3 classes."""
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def _load_semeval2017(
    config: DataConfig,
) -> tuple[SentimentDataset, SentimentDataset, SentimentDataset]:
    """SemEval-2017 Task 5 (microblogs + headlines, training data only).

    The official test split was distributed without gold labels, so the
    *training* JSON files are the only labelled portion of the corpus.
    We combine both subtasks (1,700 microblogs + 1,142 headlines = 2,842
    rows), bin the continuous sentiment score to a 3-class label via a
    symmetric threshold, and apply the same 70/15/15 stratified split that
    every other dataset in this codebase uses.
    """
    threshold = SEMEVAL2017_THRESHOLD

    mb_path = SEMEVAL2017_DATA_DIR / "Microblog_Trainingdata.json"
    hl_path = SEMEVAL2017_DATA_DIR / "Headline_Trainingdata.json"
    if not mb_path.exists() or not hl_path.exists():
        raise FileNotFoundError(
            "SemEval-2017 Task 5 training JSON files not found in "
            f"{SEMEVAL2017_DATA_DIR}. See data/semeval2017/README.md."
        )

    texts: list[str] = []
    labels: list[str] = []

    for row in json.loads(mb_path.read_text()):
        spans = row.get("spans")
        if isinstance(spans, list):
            text = " ".join(s.strip() for s in spans if s).strip()
        else:
            text = str(spans).strip()
        if not text:
            continue
        try:
            score = float(row["sentiment score"])
        except (KeyError, TypeError, ValueError):
            continue
        texts.append(text)
        labels.append(_semeval_score_to_label(score, threshold))

    for row in json.loads(hl_path.read_text()):
        title = (row.get("title") or "").strip()
        if not title:
            continue
        try:
            score = float(row["sentiment"])
        except (KeyError, TypeError, ValueError):
            continue
        texts.append(title)
        labels.append(_semeval_score_to_label(score, threshold))

    if config.max_samples is not None:
        texts = texts[: config.max_samples]
        labels = labels[: config.max_samples]

    holdout_ratio = config.test_size + config.val_size
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=holdout_ratio,
        random_state=config.random_seed,
        stratify=labels,
    )
    val_frac = config.val_size / holdout_ratio
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=1 - val_frac,
        random_state=config.random_seed,
        stratify=temp_labels,
    )

    return (
        SentimentDataset(train_texts, train_labels),
        SentimentDataset(val_texts, val_labels),
        SentimentDataset(test_texts, test_labels),
    )


def _load_fiqa2018(
    config: DataConfig,
) -> tuple[SentimentDataset, SentimentDataset, SentimentDataset]:
    """FiQA-2018 Task 1: sentiment on financial microblogs + news headlines.

    Loads the union of the official ``train`` (822), ``valid`` (117) and
    ``test`` (234) splits from ``data/fiqa2018/fiqa_all.json``, discards the
    official partitioning, and applies the same 70/15/15 stratified split
    that every other dataset in this codebase uses so that cross-dataset
    comparisons stay apples-to-apples.
    """
    threshold = FIQA2018_THRESHOLD
    src = FIQA2018_DATA_DIR / "fiqa_all.json"
    if not src.exists():
        raise FileNotFoundError(f"FiQA-2018 JSON not found at {src}. See data/fiqa2018/README.md.")

    texts: list[str] = []
    labels: list[str] = []
    for row in json.loads(src.read_text()):
        sent = (row.get("sentence") or "").strip()
        if not sent:
            continue
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
        texts.append(sent)
        labels.append(_semeval_score_to_label(score, threshold))

    if config.max_samples is not None:
        texts = texts[: config.max_samples]
        labels = labels[: config.max_samples]

    holdout_ratio = config.test_size + config.val_size
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=holdout_ratio,
        random_state=config.random_seed,
        stratify=labels,
    )
    val_frac = config.val_size / holdout_ratio
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=1 - val_frac,
        random_state=config.random_seed,
        stratify=temp_labels,
    )
    return (
        SentimentDataset(train_texts, train_labels),
        SentimentDataset(val_texts, val_labels),
        SentimentDataset(test_texts, test_labels),
    )


def load_csv_dataset(
    path: str,
    text_col: str = "text",
    label_col: str = "label",
    config: DataConfig | None = None,
) -> tuple[SentimentDataset, SentimentDataset, SentimentDataset]:
    """Load a sentiment dataset from a CSV file with text and label columns."""
    import pandas as pd

    if config is None:
        config = DataConfig()

    df = pd.read_csv(path)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].str.lower().str.strip().tolist()

    valid_labels = {"negative", "neutral", "positive"}
    mask = [lab in valid_labels for lab in labels]
    texts = [t for t, m in zip(texts, mask) if m]
    labels = [lab for lab, m in zip(labels, mask) if m]

    if len(texts) == 0:
        raise ValueError(
            f"No valid labels found in column '{label_col}'. "
            "Expected 'negative', 'neutral', or 'positive'."
        )

    holdout_ratio = config.test_size + config.val_size
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=holdout_ratio,
        random_state=config.random_seed,
        stratify=labels,
    )
    val_frac = config.val_size / holdout_ratio
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=1 - val_frac,
        random_state=config.random_seed,
        stratify=temp_labels,
    )

    return (
        SentimentDataset(train_texts, train_labels),
        SentimentDataset(val_texts, val_labels),
        SentimentDataset(test_texts, test_labels),
    )
