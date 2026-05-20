from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


class TfidfLogRegBaseline:
    """TF-IDF + multinomial Logistic Regression baseline for sentiment."""

    def __init__(
        self,
        max_features: int = 20_000,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 1.0,
        random_seed: int = 42,
    ):
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        sublinear_tf=True,
                        min_df=2,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=random_seed,
                    ),
                ),
            ]
        )

    def train(self, texts: list[str], labels: list[str]) -> None:
        self.pipeline.fit(texts, labels)

    def predict(self, texts: list[str]) -> list[str]:
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    def cross_validate(self, texts: list[str], labels: list[str], cv: int = 5) -> dict[str, float]:
        scores = cross_val_score(self.pipeline, texts, labels, cv=cv, scoring="f1_weighted")
        return {
            "mean_f1": float(scores.mean()),
            "std_f1": float(scores.std()),
            "scores": scores.tolist(),
        }
