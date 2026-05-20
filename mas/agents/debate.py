from __future__ import annotations

import json
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ..config import LABELS, LLMConfig
from .base import AgentResult, SentimentAgent

INITIAL_ASSESS_SYSTEM = """\
You are a quick sentiment screener for financial texts. Rapidly assess the \
sentiment of this text.

Respond ONLY with a JSON object:
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <float 0-1>,
  "reasoning": "<brief>"
}"""

GENERATOR_SYSTEM = """\
You are a financial sentiment analyst. Provide a thorough, well-argued \
analysis of the text's sentiment.

Consider multiple angles: market implications, language tone, factual claims, \
potential biases.

Respond ONLY with a JSON object:
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <float 0-1>,
  "arguments": ["arg1", "arg2", "arg3"],
  "reasoning": "<detailed analysis>"
}"""

DISCRIMINATOR_SYSTEM = """\
You are a critical reviewer of sentiment analysis. Your job is to challenge \
the initial analysis and find weaknesses.

Given the original text and an analyst's assessment, identify:
- Potential misinterpretations
- Missing context or nuance
- Arguments for an alternative sentiment classification

Respond ONLY with a JSON object:
{
  "agrees": true | false,
  "counter_sentiment": "positive" | "negative" | "neutral",
  "confidence": <float 0-1>,
  "challenges": ["challenge1", "challenge2"],
  "reasoning": "<critical review>"
}"""

ARBITRATOR_SYSTEM = """\
You are a senior arbiter resolving a sentiment analysis debate. Two analysts \
disagree on the sentiment of a financial text.

Consider both positions carefully and make a final, well-justified determination.

Respond ONLY with a JSON object:
{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": <float 0-1>,
  "resolution": "<how you resolved the disagreement>",
  "reasoning": "<final reasoning>"
}"""


class DebateState(TypedDict):
    text: str
    initial_sentiment: str
    initial_confidence: float
    generator_output: str
    discriminator_output: str
    final_sentiment: str
    final_confidence: float
    final_reasoning: str
    route: str
    total_tokens: int


class DebateMultiAgentSystem(SentimentAgent):
    """Tier 3: confidence-routed debate protocol between LLM agents."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        confidence_threshold: float = 0.85,
    ):
        self.config = config or LLMConfig()
        self.name = "MultiAgent-Debate"
        self.confidence_threshold = confidence_threshold
        self._llm = ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        self._total_tokens = 0
        self._total_cost = 0.0
        self._fast_count = 0
        self._debate_count = 0
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

    def _initial_assess_node(self, state: DebateState) -> dict:
        result, tokens = self._call_llm(INITIAL_ASSESS_SYSTEM, state["text"])
        sentiment = result.get("sentiment", "neutral").lower().strip()
        confidence = float(result.get("confidence", 0.5))
        return {
            "initial_sentiment": sentiment if sentiment in LABELS else "neutral",
            "initial_confidence": confidence,
            "total_tokens": state.get("total_tokens", 0) + tokens,
        }

    def _route(self, state: DebateState) -> str:
        if state["initial_confidence"] >= self.confidence_threshold:
            return "fast_path"
        return "debate"

    def _fast_path_node(self, state: DebateState) -> dict:
        self._fast_count += 1
        return {
            "final_sentiment": state["initial_sentiment"],
            "final_confidence": state["initial_confidence"],
            "final_reasoning": "High-confidence fast path (no debate needed)",
            "route": "fast",
        }

    def _generator_node(self, state: DebateState) -> dict:
        result, tokens = self._call_llm(GENERATOR_SYSTEM, state["text"])
        return {
            "generator_output": json.dumps(result),
            "total_tokens": state.get("total_tokens", 0) + tokens,
        }

    def _discriminator_node(self, state: DebateState) -> dict:
        user_msg = (
            f"Original text: {state['text']}\n\n"
            f"Analyst's assessment: {state['generator_output']}"
        )
        result, tokens = self._call_llm(DISCRIMINATOR_SYSTEM, user_msg)
        return {
            "discriminator_output": json.dumps(result),
            "total_tokens": state.get("total_tokens", 0) + tokens,
        }

    def _arbitrator_node(self, state: DebateState) -> dict:
        self._debate_count += 1
        user_msg = (
            f"Original text: {state['text']}\n\n"
            f"Analyst's position: {state['generator_output']}\n\n"
            f"Critic's position: {state['discriminator_output']}"
        )
        result, tokens = self._call_llm(ARBITRATOR_SYSTEM, user_msg)
        sentiment = result.get("sentiment", "neutral").lower().strip()
        return {
            "final_sentiment": sentiment if sentiment in LABELS else "neutral",
            "final_confidence": float(result.get("confidence", 0.5)),
            "final_reasoning": result.get("reasoning", ""),
            "route": "debate",
            "total_tokens": state.get("total_tokens", 0) + tokens,
        }

    def _build_graph(self):
        graph = StateGraph(DebateState)

        graph.add_node("initial_assess", self._initial_assess_node)
        graph.add_node("fast_path", self._fast_path_node)
        graph.add_node("generator", self._generator_node)
        graph.add_node("discriminator", self._discriminator_node)
        graph.add_node("arbitrator", self._arbitrator_node)

        graph.add_edge(START, "initial_assess")
        graph.add_conditional_edges(
            "initial_assess",
            self._route,
            {"fast_path": "fast_path", "debate": "generator"},
        )
        graph.add_edge("fast_path", END)
        graph.add_edge("generator", "discriminator")
        graph.add_edge("discriminator", "arbitrator")
        graph.add_edge("arbitrator", END)

        return graph.compile()

    def analyze(self, text: str) -> AgentResult:
        initial: DebateState = {
            "text": text,
            "initial_sentiment": "",
            "initial_confidence": 0.0,
            "generator_output": "",
            "discriminator_output": "",
            "final_sentiment": "",
            "final_confidence": 0.0,
            "final_reasoning": "",
            "route": "",
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
                    "route": result.get("route", "unknown"),
                    "generator": result.get("generator_output", ""),
                    "discriminator": result.get("discriminator_output", ""),
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

    @property
    def routing_stats(self) -> dict:
        total = self._fast_count + self._debate_count
        return {
            "fast_path": self._fast_count,
            "debate": self._debate_count,
            "total": total,
            "fast_pct": self._fast_count / total * 100 if total > 0 else 0,
        }
