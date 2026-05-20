from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ..config import LABELS, LLMConfig
from .base import AgentResult, SentimentAgent

ANALYST_SYSTEM = """\
You are a senior financial analyst specializing in crypto markets. \
Analyze the sentiment of the given text.

Consider:
- Market implications (bullish/bearish signals)
- Tone and language (optimistic, pessimistic, factual)
- Context clues about market direction

Respond ONLY with a JSON object:
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <float 0-1>,
  "key_signals": ["signal1", "signal2"],
  "reasoning": "<detailed analysis>"
}"""

FACT_CHECKER_SYSTEM = """\
You are a financial fact-checker and context analyst. Given a piece of \
financial/crypto news and an initial analysis, verify the analysis and add \
context.

Consider:
- Is the initial sentiment assessment accurate?
- Are there nuances the analyst may have missed?
- What broader context might change the interpretation?

Respond ONLY with a JSON object:
{
  "agrees_with_analyst": true | false,
  "adjusted_sentiment": "positive" | "negative" | "neutral",
  "confidence": <float 0-1>,
  "context_notes": "<additional context>",
  "reasoning": "<verification reasoning>"
}"""

AGGREGATOR_SYSTEM = """\
You are a senior decision-maker who synthesizes multiple expert opinions \
into a final sentiment verdict.

Given the original text, an analyst's assessment, and a fact-checker's review, \
produce the final sentiment classification.

Rules:
- Weigh both opinions; favor fact-checked conclusions when they add context.
- If they agree, increase confidence.
- If they disagree, explain why you chose one over the other.

Respond ONLY with a JSON object:
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <float 0-1>,
  "reasoning": "<synthesis reasoning>"
}"""


class AnalysisState(TypedDict):
    text: str
    analyst_output: str
    fact_checker_output: str
    final_sentiment: str
    final_confidence: float
    final_reasoning: str
    total_tokens: int


class MultiAgentSystem(SentimentAgent):
    """Tier 2: three specialized LLM agents (Analyst -> FactChecker -> Aggregator)."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.name = "MultiAgent-Pipeline"
        self._llm = ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        self._total_tokens = 0
        self._total_cost = 0.0
        self._graph = self._build_graph()

    def _call_llm(self, system: str, user: str) -> tuple[dict, int]:
        response = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        tokens = 0
        if response.usage_metadata:
            tokens = response.usage_metadata.get("total_tokens", 0)
        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            parsed = {
                "sentiment": "neutral",
                "confidence": 0.0,
                "reasoning": response.content,
            }
        return parsed, tokens

    def _analyst_node(self, state: AnalysisState) -> dict:
        result, tokens = self._call_llm(ANALYST_SYSTEM, state["text"])
        return {
            "analyst_output": json.dumps(result),
            "total_tokens": state.get("total_tokens", 0) + tokens,
        }

    def _fact_checker_node(self, state: AnalysisState) -> dict:
        user_msg = (
            f"Original text: {state['text']}\n\n" f"Analyst's assessment: {state['analyst_output']}"
        )
        result, tokens = self._call_llm(FACT_CHECKER_SYSTEM, user_msg)
        return {
            "fact_checker_output": json.dumps(result),
            "total_tokens": state.get("total_tokens", 0) + tokens,
        }

    def _aggregator_node(self, state: AnalysisState) -> dict:
        user_msg = (
            f"Original text: {state['text']}\n\n"
            f"Analyst's assessment: {state['analyst_output']}\n\n"
            f"Fact-checker's review: {state['fact_checker_output']}"
        )
        result, tokens = self._call_llm(AGGREGATOR_SYSTEM, user_msg)
        sentiment = result.get("sentiment", "neutral").lower().strip()
        if sentiment not in LABELS:
            sentiment = "neutral"
        return {
            "final_sentiment": sentiment,
            "final_confidence": float(result.get("confidence", 0.5)),
            "final_reasoning": result.get("reasoning", ""),
            "total_tokens": state.get("total_tokens", 0) + tokens,
        }

    def _build_graph(self):
        graph = StateGraph(AnalysisState)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("fact_checker", self._fact_checker_node)
        graph.add_node("aggregator", self._aggregator_node)

        graph.add_edge(START, "analyst")
        graph.add_edge("analyst", "fact_checker")
        graph.add_edge("fact_checker", "aggregator")
        graph.add_edge("aggregator", END)

        return graph.compile()

    def analyze(self, text: str) -> AgentResult:
        initial: AnalysisState = {
            "text": text,
            "analyst_output": "",
            "fact_checker_output": "",
            "final_sentiment": "",
            "final_confidence": 0.0,
            "final_reasoning": "",
            "total_tokens": 0,
        }

        try:
            result = self._graph.invoke(initial)
            tokens = result.get("total_tokens", 0)
            self._total_tokens += tokens
            self._total_cost += (tokens / 1_000_000) * 0.375

            return AgentResult(
                sentiment=result["final_sentiment"],
                confidence=result["final_confidence"],
                reasoning=result["final_reasoning"],
                metadata={
                    "tokens": tokens,
                    "analyst": result.get("analyst_output", ""),
                    "fact_checker": result.get("fact_checker_output", ""),
                },
            )
        except Exception as e:
            return AgentResult(
                sentiment="neutral",
                confidence=0.0,
                reasoning=f"Error: {e}",
                metadata={"error": str(e)},
            )

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return self._total_tokens
