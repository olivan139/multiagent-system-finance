from __future__ import annotations

import re


def preprocess_text(text: str) -> str:
    """Clean a single text sample for sentiment analysis."""
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_batch(texts: list[str]) -> list[str]:
    return [preprocess_text(t) for t in texts]
