from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class DataConfig:
    dataset_name: str = "zeroshot/twitter-financial-news-sentiment"
    dataset_config: str = ""
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42
    max_samples: int | None = None


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    max_retries: int = 3
    retry_delay: float = 1.0

    @property
    def api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")


@dataclass
class TransformerConfig:
    model_name: str = "ProsusAI/finbert"
    num_labels: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_length: int = 128


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    random_seed: int = 42
