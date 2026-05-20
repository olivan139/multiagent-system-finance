"""Heterogeneous multi-agent ensemble for sentiment classification.

Each "agent" is a fundamentally different model paradigm:
    - Agent 1: TF-IDF + Logistic Regression (classical ML, supervised)
    - Agent 2: Fine-tuned FinBERT (transformer, supervised)
    - Agent 3: GPT-4o-mini (LLM, zero-shot)
    - Agent 4: Loughran-McDonald lexicon (dictionary-based, unsupervised)

Outputs are combined via three strategies:
    - Majority vote: simple democratic decision
    - Weighted average: average probability distributions
    - Stacking: a trained meta-learner (Logistic Regression) over agent
      probabilities — this is the strong, properly-trained variant
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from ..baselines.lexicon import LoughranMcDonaldAgent
from ..config import LABEL2ID, LABELS

AggregationStrategy = Literal["majority", "weighted_average", "stacking"]


AGENT_ORDER: tuple[str, ...] = ("tfidf", "finbert", "llm", "lexicon")


@dataclass
class EnsembleAgentPredictions:
    """Per-agent predictions for one batch."""

    tfidf_proba: np.ndarray
    finbert_proba: np.ndarray
    llm_labels: list[str]
    llm_confidence: np.ndarray
    llm_proba: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    lexicon_proba: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


def _llm_to_pseudo_proba(labels: list[str], confidences: np.ndarray) -> np.ndarray:
    """Convert LLM (label, confidence) pairs to a soft probability matrix.

    Uses a simple smoothing scheme: chosen class gets ``confidence``, the
    remaining mass ``1 - confidence`` is split equally between the other two.
    This lets the LLM participate in soft aggregations and stacking.
    """
    n = len(labels)
    proba = np.zeros((n, len(LABELS)), dtype=np.float64)
    for i, (lab, c) in enumerate(zip(labels, confidences)):
        c = float(np.clip(c, 0.0, 1.0))
        chosen = LABEL2ID.get(lab, LABEL2ID["neutral"])
        proba[i, :] = (1.0 - c) / (len(LABELS) - 1)
        proba[i, chosen] = c
    return proba


class HeterogeneousEnsemble:
    """Stacking-style ensemble combining classical ML, transformer, LLM, and
    lexicon agents.

    The ensemble does not subclass ``SentimentAgent`` because its agents
    operate in batch mode (FinBERT especially), and the predict-then-aggregate
    pattern is far more efficient than per-sample analysis.
    """

    def __init__(
        self,
        tfidf_baseline,
        finbert_baseline,
        llm_agent,
        lexicon_agent: LoughranMcDonaldAgent | None = None,
        strategy: AggregationStrategy = "stacking",
        weights: tuple[float, ...] | None = None,
        meta_model_C: float = 1.0,
        random_seed: int = 42,
        active_agents: tuple[str, ...] = AGENT_ORDER,
    ):
        self.tfidf = tfidf_baseline
        self.finbert = finbert_baseline
        self.llm = llm_agent

        if lexicon_agent is None and "lexicon" in active_agents:
            lexicon_agent = LoughranMcDonaldAgent()
        self.lexicon = lexicon_agent
        self.strategy = strategy

        self.weights = weights or tuple(1.0 for _ in AGENT_ORDER)
        self.meta_model: LogisticRegression | None = None
        self.meta_model_C = meta_model_C
        self.random_seed = random_seed
        self.active_agents = tuple(active_agents)
        if len(self.active_agents) < 2:
            raise ValueError("Ensemble needs at least 2 active agents.")
        self.name = f"Ensemble-{strategy}-{'+'.join(self.active_agents)}"

        self._predict_latency_ms: float = 0.0
        self._llm_total_cost_usd: float = 0.0
        self._llm_total_tokens: int = 0

    def collect_agent_predictions(
        self, texts: list[str], show_progress: bool = True
    ) -> EnsembleAgentPredictions:
        """Public alias of ``_collect_agent_predictions`` for caching/reuse."""
        return self._collect_agent_predictions(texts, show_progress=show_progress)

    def _collect_agent_predictions(
        self, texts: list[str], show_progress: bool = True
    ) -> EnsembleAgentPredictions:
        """Run active agents on the given texts and gather their outputs.

        Inactive agents (per ``self.active_agents``) get a uniform 1/3 dummy
        distribution, so the columns shape stays stable but they contribute
        nothing to majority/weighted aggregations and stacking simply zeros
        out their coefficients.
        """
        n = len(texts)
        uniform = np.full((n, len(LABELS)), 1.0 / len(LABELS))

        if "tfidf" in self.active_agents:
            if show_progress:
                print(f"  [Ensemble] Agent: TF-IDF + LogReg on {n} samples")
            tfidf_proba_raw = self.tfidf.predict_proba(texts)
            tfidf_classes = list(self.tfidf.pipeline.classes_)

            tfidf_proba = np.zeros((n, len(LABELS)), dtype=np.float64)
            for j, label in enumerate(LABELS):
                if label in tfidf_classes:
                    tfidf_proba[:, j] = tfidf_proba_raw[:, tfidf_classes.index(label)]
        else:
            tfidf_proba = uniform.copy()

        if "finbert" in self.active_agents:
            if show_progress:
                print(f"  [Ensemble] Agent: FinBERT on {n} samples")
            finbert_proba = self.finbert.predict_proba(texts, label_order=LABELS)
        else:
            finbert_proba = uniform.copy()

        if "llm" in self.active_agents:
            if show_progress:
                print(f"  [Ensemble] Agent: LLM on {n} samples")
            llm_results = self.llm.analyze_batch(texts, show_progress=show_progress)
            llm_labels = [r.sentiment for r in llm_results]
            llm_conf = np.array([r.confidence for r in llm_results], dtype=np.float64)
            llm_proba = _llm_to_pseudo_proba(llm_labels, llm_conf)
            if hasattr(self.llm, "total_cost_usd"):
                self._llm_total_cost_usd = float(self.llm.total_cost_usd)
            if hasattr(self.llm, "total_tokens"):
                self._llm_total_tokens = int(self.llm.total_tokens)
        else:
            llm_labels = ["neutral"] * n
            llm_conf = np.full(n, 1.0 / len(LABELS))
            llm_proba = uniform.copy()

        if "lexicon" in self.active_agents:
            if show_progress:
                print(f"  [Ensemble] Agent: LM-Lexicon on {n} samples")
            assert self.lexicon is not None
            lexicon_proba = self.lexicon.predict_proba(texts, label_order=LABELS)
        else:
            lexicon_proba = uniform.copy()

        return EnsembleAgentPredictions(
            tfidf_proba=tfidf_proba,
            finbert_proba=finbert_proba,
            llm_labels=llm_labels,
            llm_confidence=llm_conf,
            llm_proba=llm_proba,
            lexicon_proba=lexicon_proba,
        )

    def _majority_vote(
        self,
        tfidf_proba: np.ndarray,
        finbert_proba: np.ndarray,
        llm_labels: list[str],
        lexicon_proba: np.ndarray,
    ) -> list[str]:
        """Each active agent casts one vote (its argmax). Ties → neutral."""
        tfidf_pred = np.argmax(tfidf_proba, axis=1)
        finbert_pred = np.argmax(finbert_proba, axis=1)
        llm_pred = np.array([LABEL2ID.get(l, LABEL2ID["neutral"]) for l in llm_labels])
        lexicon_pred = np.argmax(lexicon_proba, axis=1)
        n = len(llm_labels)
        out = []
        for i in range(n):
            votes = []
            if "tfidf" in self.active_agents:
                votes.append(tfidf_pred[i])
            if "finbert" in self.active_agents:
                votes.append(finbert_pred[i])
            if "llm" in self.active_agents:
                votes.append(llm_pred[i])
            if "lexicon" in self.active_agents:
                votes.append(lexicon_pred[i])
            counts = np.bincount(votes, minlength=len(LABELS))
            top = int(np.argmax(counts))

            if counts[top] == 1:
                top = LABEL2ID["neutral"]
            out.append(LABELS[top])
        return out

    def _weighted_average(
        self,
        tfidf_proba: np.ndarray,
        finbert_proba: np.ndarray,
        llm_proba: np.ndarray,
        lexicon_proba: np.ndarray,
    ) -> list[str]:
        w_map = dict(zip(AGENT_ORDER, self.weights))
        proba_map = {
            "tfidf": tfidf_proba,
            "finbert": finbert_proba,
            "llm": llm_proba,
            "lexicon": lexicon_proba,
        }
        total = sum(w_map[a] for a in self.active_agents)
        avg = sum(w_map[a] * proba_map[a] for a in self.active_agents) / total
        return [LABELS[i] for i in np.argmax(avg, axis=1)]

    def _stack_features(self, p: EnsembleAgentPredictions) -> np.ndarray:
        """Concatenate active agents' probabilities into one feature matrix.

        Shape: (N, 3 * |active_agents|). With all four agents active that
        is (N, 12).
        """
        cols: list[np.ndarray] = []
        if "tfidf" in self.active_agents:
            cols.append(p.tfidf_proba)
        if "finbert" in self.active_agents:
            cols.append(p.finbert_proba)
        if "llm" in self.active_agents:
            cols.append(p.llm_proba)
        if "lexicon" in self.active_agents:
            cols.append(p.lexicon_proba)
        return np.hstack(cols)

    def fit_meta_learner(
        self,
        val_texts: list[str],
        val_labels: list[str],
        precomputed: EnsembleAgentPredictions | None = None,
    ) -> dict:
        """Train the stacking meta-learner on validation-set agent outputs.

        IMPORTANT: this must be the validation set, NOT the training set —
        otherwise the supervised agents (TF-IDF, FinBERT) leak training-set
        performance and the meta-learner overfits to their false confidence.

        Pass ``precomputed`` to avoid re-running the agents (useful when
        evaluating multiple strategies on the same validation split).
        """
        if self.strategy != "stacking":
            return {"info": "meta_learner not used for non-stacking strategy"}

        preds = precomputed or self._collect_agent_predictions(val_texts, show_progress=True)
        X = self._stack_features(preds)
        y = np.array([LABEL2ID[lab] for lab in val_labels])

        self.meta_model = LogisticRegression(
            C=self.meta_model_C,
            solver="lbfgs",
            max_iter=2000,
            random_state=self.random_seed,
        )
        self.meta_model.fit(X, y)

        train_acc = float(self.meta_model.score(X, y))
        return {
            "meta_train_accuracy": train_acc,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_names": self._feature_names(),
        }

    def _feature_names(self) -> list[str]:
        return [f"{agent}_{cls}" for agent in self.active_agents for cls in LABELS]

    def predict_labels(
        self,
        texts: list[str],
        show_progress: bool = True,
        precomputed: EnsembleAgentPredictions | None = None,
    ) -> list[str]:
        """Run the ensemble end-to-end and return final predicted labels.

        Pass ``precomputed`` (output of ``collect_agent_predictions``) to
        avoid re-running expensive agents when evaluating multiple
        aggregation strategies on the same texts.
        """
        t0 = time.time()
        preds = precomputed or self._collect_agent_predictions(texts, show_progress=show_progress)

        if self.strategy == "majority":
            out = self._majority_vote(
                preds.tfidf_proba,
                preds.finbert_proba,
                preds.llm_labels,
                preds.lexicon_proba,
            )
        elif self.strategy == "weighted_average":
            out = self._weighted_average(
                preds.tfidf_proba,
                preds.finbert_proba,
                preds.llm_proba,
                preds.lexicon_proba,
            )
        elif self.strategy == "stacking":
            if self.meta_model is None:
                raise RuntimeError(
                    "Meta-learner not trained. Call fit_meta_learner(val_texts, val_labels) first."
                )
            X = self._stack_features(preds)
            pred_ids = self.meta_model.predict(X)
            out = [LABELS[int(i)] for i in pred_ids]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self._predict_latency_ms = (time.time() - t0) * 1000
        return out

    @property
    def total_cost_usd(self) -> float:
        return self._llm_total_cost_usd

    @property
    def total_tokens(self) -> int:
        return self._llm_total_tokens

    def feature_importance(self) -> dict[str, float]:
        """Return per-feature absolute mean coefficient from the meta-learner.

        Higher values indicate features (agent × class) the meta-learner
        relies on more heavily for classification.
        """
        if self.meta_model is None:
            return {}
        coefs = np.abs(self.meta_model.coef_).mean(axis=0)
        names = self._feature_names()
        importance = dict(zip(names, [float(c) for c in coefs]))

        per_agent: dict[str, float] = {a: 0.0 for a in self.active_agents}
        for name, val in importance.items():
            agent = name.split("_")[0]
            if agent in per_agent:
                per_agent[agent] += val
        importance.update({f"_total_{k}": v for k, v in per_agent.items()})
        return importance
