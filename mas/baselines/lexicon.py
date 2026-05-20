"""Loughran-McDonald financial sentiment lexicon agent.

A lightweight, fully interpretable sentiment classifier driven entirely by
the Loughran-McDonald (LM) finance-specific positive/negative word lists.
Serves as a fourth, fundamentally different paradigm alongside TF-IDF +
LogReg (classical ML), FinBERT (transformer), and the LLM agent.

Reference: Loughran, T., & McDonald, B. (2011). "When Is a Liability Not a
Liability? Textual Analysis, Dictionaries, and 10-Ks." Journal of Finance.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from ..config import LABELS

_TOKEN_RE = re.compile(r"[A-Za-z]+")


_FALLBACK_POSITIVE: tuple[str, ...] = (
    "able",
    "achieve",
    "achieved",
    "achieves",
    "advantage",
    "advantageous",
    "beneficial",
    "benefit",
    "benefits",
    "boost",
    "boosted",
    "boosts",
    "constructive",
    "delight",
    "delighted",
    "effective",
    "efficient",
    "enhance",
    "enhanced",
    "enhances",
    "enjoy",
    "enjoyed",
    "excellence",
    "excellent",
    "exceptional",
    "favorable",
    "favorably",
    "gain",
    "gained",
    "gains",
    "good",
    "great",
    "greater",
    "highest",
    "improve",
    "improved",
    "improvement",
    "improvements",
    "improves",
    "improving",
    "leading",
    "outperform",
    "outperformed",
    "perfect",
    "pleased",
    "positive",
    "profit",
    "profitability",
    "profitable",
    "profits",
    "progress",
    "progressed",
    "rebound",
    "rebounded",
    "record",
    "satisfactory",
    "strength",
    "strengthen",
    "strengthened",
    "strong",
    "stronger",
    "strongest",
    "succeed",
    "succeeded",
    "success",
    "successful",
    "successfully",
    "superior",
    "surpass",
    "surpassed",
    "upturn",
    "win",
    "winner",
    "winning",
)

_FALLBACK_NEGATIVE: tuple[str, ...] = (
    "adverse",
    "adversely",
    "against",
    "bad",
    "bankrupt",
    "bankruptcy",
    "burden",
    "burdened",
    "challenge",
    "challenges",
    "concern",
    "concerned",
    "concerns",
    "crisis",
    "critical",
    "damage",
    "damaged",
    "damages",
    "decline",
    "declined",
    "declines",
    "declining",
    "default",
    "defaulted",
    "deficit",
    "delay",
    "delayed",
    "delays",
    "deteriorate",
    "deteriorated",
    "deterioration",
    "difficult",
    "difficulties",
    "difficulty",
    "disappoint",
    "disappointed",
    "disappointing",
    "disappointment",
    "doubt",
    "doubtful",
    "down",
    "downgrade",
    "downgraded",
    "downturn",
    "drop",
    "dropped",
    "drops",
    "fail",
    "failed",
    "failing",
    "fails",
    "failure",
    "failures",
    "fall",
    "fallen",
    "falling",
    "falls",
    "fear",
    "fears",
    "fraud",
    "harm",
    "harmed",
    "harmful",
    "hurt",
    "impair",
    "impaired",
    "impairment",
    "investigation",
    "lawsuit",
    "lawsuits",
    "litigation",
    "lose",
    "loses",
    "losing",
    "loss",
    "losses",
    "lost",
    "miss",
    "missed",
    "missing",
    "negative",
    "negatively",
    "penalty",
    "plunge",
    "plunged",
    "poor",
    "poorly",
    "problem",
    "problems",
    "recession",
    "restate",
    "restated",
    "shortfall",
    "slowdown",
    "slump",
    "slumped",
    "struggle",
    "struggled",
    "struggles",
    "suffer",
    "suffered",
    "suffering",
    "suffers",
    "trouble",
    "troubled",
    "troubles",
    "uncertain",
    "uncertainty",
    "underperform",
    "underperformed",
    "unfavorable",
    "warn",
    "warned",
    "warning",
    "weak",
    "weaken",
    "weakened",
    "weakness",
    "worse",
    "worsen",
    "worsened",
    "worst",
    "wrong",
)


def _load_lexicon() -> tuple[frozenset[str], frozenset[str], str, object | None]:
    """Return (positive_words, negative_words, source_tag, pysentiment2_lm).

    Tries ``pysentiment2`` first (full LM master dictionary); falls back to
    the inline reduced lexicon when the package is unavailable.

    Important: pysentiment2 stores its LM dictionary as Porter-stemmed
    tokens (``achiev``, ``advanc``, ``compliment`` ...), and its
    ``tokenize`` method applies the same stemmer plus stopword removal.
    To get the right matching behaviour we keep the loaded ``LM``
    instance and let it tokenise the inputs at predict time. The bare
    ``positive``/``negative`` sets are still returned so that the
    fallback path keeps working with raw lowercase tokens.
    """
    try:
        import pysentiment2 as ps

        lm = ps.LM()
        pos: set[str] = set()
        neg: set[str] = set()
        if hasattr(lm, "static") and isinstance(lm.static, dict):
            for key, words in lm.static.items():
                k = str(key).lower()
                if "pos" in k:
                    pos.update(str(w).lower() for w in words)
                elif "neg" in k:
                    neg.update(str(w).lower() for w in words)
        if not pos or not neg:

            for attr in ("_posset", "posset", "_pos_set", "pos_set"):
                if hasattr(lm, attr):
                    pos.update(str(w).lower() for w in getattr(lm, attr))
            for attr in ("_negset", "negset", "_neg_set", "neg_set"):
                if hasattr(lm, attr):
                    neg.update(str(w).lower() for w in getattr(lm, attr))
        if pos and neg:
            return frozenset(pos), frozenset(neg), "pysentiment2", lm
    except Exception:
        pass

    return (
        frozenset(_FALLBACK_POSITIVE),
        frozenset(_FALLBACK_NEGATIVE),
        "inline-fallback",
        None,
    )


class LoughranMcDonaldAgent:
    """Dictionary-based sentiment agent using the Loughran-McDonald lists.

    Contract matches the other baselines (see ``TransformerBaseline``):
        - ``predict(texts) -> list[str]`` of canonical labels
        - ``predict_proba(texts, label_order=None) -> np.ndarray`` of shape
          ``(N, 3)`` aligned to ``label_order`` (default ``mas.config.LABELS``).
    """

    def __init__(self) -> None:
        self.positive, self.negative, self.source, self._lm = _load_lexicon()
        self.name = "lexicon-lm"

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs) -> "LoughranMcDonaldAgent":
        return cls()

    def _tokenise(self, text: str) -> list[str]:
        """Tokenise an input string for lexicon lookup.

        When pysentiment2 is available, delegate to its ``tokenize`` which
        applies Porter stemming + stopword removal, matching the form the
        LM dictionary is stored in. Without pysentiment2 we fall back to
        raw lowercase word tokens (this only works correctly against the
        inline fallback word lists, which are not pre-stemmed).
        """
        if self._lm is not None:
            try:
                return list(self._lm.tokenize(str(text or "")))
            except Exception:
                pass
        return [tok for tok in _TOKEN_RE.findall((text or "").lower()) if tok]

    def _counts(self, text: str) -> tuple[int, int]:
        if self._lm is not None:
            try:
                sc = self._lm.get_score(self._lm.tokenize(str(text or "")))
                return int(sc.get("Positive", 0)), int(sc.get("Negative", 0))
            except Exception:
                pass
        pos = neg = 0
        for tok in self._tokenise(text):
            if tok in self.positive:
                pos += 1
            if tok in self.negative:
                neg += 1
        return pos, neg

    def predict(self, texts: Iterable[str] | str) -> list[str] | str:
        """Predict canonical labels.

        Accepts either a single string (returns a single label, for parity
        with simple sentiment helpers) or an iterable of strings (returns a
        list of labels).
        """
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        out: list[str] = []
        for t in items:
            pos, neg = self._counts(t)
            if pos > neg:
                out.append("positive")
            elif neg > pos:
                out.append("negative")
            else:
                out.append("neutral")
        return out[0] if single else out

    def predict_proba(
        self,
        texts: Iterable[str],
        label_order: list[str] | None = None,
    ) -> np.ndarray:
        """Soft probabilities derived from positive/negative word counts.

        Scoring scheme:
            score = (pos - neg) / max(pos + neg, 1)
            |score| < 0.05  → (neg=0.15, neu=0.70, pos=0.15)
            score > 0       → pos = 0.5 + 0.5*|s|,  neu = 0.4 - 0.4*|s|,
                              neg = max(1 - pos - neu, 0)
            score < 0       → mirror of the above
        Rows are renormalised to sum to 1.
        """
        order = list(label_order) if label_order is not None else list(LABELS)
        texts_list = list(texts)
        n = len(texts_list)

        canonical = np.zeros((n, 3), dtype=np.float64)
        for i, t in enumerate(texts_list):
            pos, neg = self._counts(t)
            denom = max(pos + neg, 1)
            score = (pos - neg) / denom
            mag = abs(score)
            if mag < 0.05:
                p_neg, p_neu, p_pos = 0.15, 0.70, 0.15
            elif score > 0:
                p_pos = 0.5 + 0.5 * mag
                p_neu = 0.4 - 0.4 * mag
                p_neg = max(1.0 - p_pos - p_neu, 0.0)
            else:
                p_neg = 0.5 + 0.5 * mag
                p_neu = 0.4 - 0.4 * mag
                p_pos = max(1.0 - p_neg - p_neu, 0.0)
            canonical[i] = (p_neg, p_neu, p_pos)

        row_sums = canonical.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        canonical = canonical / row_sums

        col_idx = [LABELS.index(lab) for lab in order]
        return canonical[:, col_idx]
