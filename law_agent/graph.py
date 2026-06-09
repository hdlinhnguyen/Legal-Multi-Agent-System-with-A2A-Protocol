"""Law Agent LangGraph StateGraph definition — optimised for low latency.

Graph topology:
    analyze_law → check_routing → (parallel) call_tax + call_compliance → aggregate → END

Key performance improvements vs original:
1. `get_fast_llm()` for the routing/analysis node (capped at 300 tokens)
2. `aggregate()` no longer makes an LLM call — it formats results via a
   string template.  This removes 1 full LLM round-trip from every request.
3. LLM instances are module-level singletons (via lru_cache in llm.py).
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send
from langgraph.graph import END, StateGraph

from common.llm import get_fast_llm, get_llm

logger = logging.getLogger(__name__)

MAX_DELEGATION_DEPTH = 3


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

def _last_wins(a: str, b: str) -> str:
    """Reducer: keep the most recently written value."""
    return b if b else a


class LawState(TypedDict):
    question: str
    context_id: str
    trace_id: str
    delegation_depth: int
    law_analysis: str
    needs_tax: bool
    needs_compliance: bool
    # Annotated so parallel branches can both write without conflict
    tax_result: Annotated[str, _last_wins]
    compliance_result: Annotated[str, _last_wins]
    final_answer: str


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

async def analyze_and_route(state: LawState) -> dict:
    """LLM analysis + routing decision in a single call using fast LLM."""
    # Use fast LLM for routing — deterministic JSON output, short response
    llm = get_fast_llm()
    depth = state.get("delegation_depth", 0)

    if depth >= MAX_DELEGATION_DEPTH:
        logger.info("Max delegation depth reached (%d); skipping sub-agents", depth)
        # Use full LLM only when we need a real legal analysis with no sub-agents
        full_llm = get_llm()
        messages = [
            SystemMessage(
                content=(
                    "You are a senior corporate litigation attorney. Analyse the legal aspects "
                    "of the question in under 150 words. Be direct and concise."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
        result = await full_llm.ainvoke(messages)
        return {"law_analysis": result.content, "needs_tax": False, "needs_compliance": False}

    messages = [
        SystemMessage(
            content=(
                "You are a legal routing expert. Briefly analyse the legal aspects (under 150 words) "
                "and decide if specialist tax or compliance sub-agents are needed.\n\n"
                "Reply with ONLY valid JSON (no markdown):\n"
                "{\n"
                '  "law_analysis": "<your analysis under 150 words>",\n'
                '  "needs_tax": <true|false>,\n'
                '  "needs_compliance": <true|false>\n'
                "}\n\n"
                "needs_tax = true → question involves tax law, IRS, tax evasion, penalties\n"
                "needs_compliance = true → question involves regulatory compliance, SEC, SOX, AML, FCPA"
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    result = await llm.ainvoke(messages)
    raw = result.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Routing LLM returned non-JSON: %r — defaulting to both=True", raw)
        parsed = {
            "law_analysis": raw,
            "needs_tax": True,
            "needs_compliance": True
        }

    return {
        "law_analysis": parsed.get("law_analysis", ""),
        "needs_tax": bool(parsed.get("needs_tax", True)),
        "needs_compliance": bool(parsed.get("needs_compliance", True)),
    }


def route_to_subagents(state: LawState) -> list[Send]:
    """Routing function: dispatch parallel Send objects based on routing flags."""
    sends: list[Send] = []
    if state.get("needs_tax"):
        sends.append(Send("call_tax", state))
    if state.get("needs_compliance"):
        sends.append(Send("call_compliance", state))
    if not sends:
        # No sub-agents needed — go straight to aggregation
        sends.append(Send("aggregate", state))
    return sends


async def call_tax(state: LawState) -> dict:
    """Delegate to the Tax Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    try:
        endpoint = await discover("tax_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Tax Agent returned %d chars", len(result))
        return {"tax_result": result}
    except Exception as exc:
        logger.exception("call_tax failed: %s", exc)
        return {"tax_result": f"[Tax analysis unavailable: {exc}]"}


async def call_compliance(state: LawState) -> dict:
    """Delegate to the Compliance Agent via A2A."""
    from common.a2a_client import delegate
    from common.registry_client import discover

    try:
        endpoint = await discover("compliance_question")
        result = await delegate(
            endpoint=endpoint,
            question=state["question"],
            context_id=state["context_id"],
            trace_id=state["trace_id"],
            depth=state.get("delegation_depth", 0) + 1,
        )
        logger.info("Compliance Agent returned %d chars", len(result))
        return {"compliance_result": result}
    except Exception as exc:
        logger.exception("call_compliance failed: %s", exc)
        return {"compliance_result": f"[Compliance analysis unavailable: {exc}]"}


def aggregate(state: LawState) -> dict:
    """Combine results using a string template — NO LLM call needed.

    Previously this made a full LLM round-trip just to format text.
    Replacing it with a deterministic template removes ~8–15 s of latency.
    """
    sections: list[str] = []

    if state.get("law_analysis"):
        sections.append(f"## ⚖️ Legal Analysis\n\n{state['law_analysis']}")

    if state.get("tax_result"):
        sections.append(f"## 💰 Tax Analysis\n\n{state['tax_result']}")

    if state.get("compliance_result"):
        sections.append(f"## 📋 Regulatory Compliance Analysis\n\n{state['compliance_result']}")

    combined = "\n\n---\n\n".join(sections)

    disclaimer = (
        "\n\n---\n\n"
        "> **Disclaimer:** This analysis is for educational purposes only. "
        "Please consult a licensed attorney for advice specific to your situation."
    )

    return {"final_answer": combined + disclaimer}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_graph():
    """Build and compile the Law Agent StateGraph."""
    graph = StateGraph(LawState)

    graph.add_node("analyze_and_route", analyze_and_route)
    graph.add_node("call_tax", call_tax)
    graph.add_node("call_compliance", call_compliance)
    graph.add_node("aggregate", aggregate)

    graph.set_entry_point("analyze_and_route")

    graph.add_conditional_edges(
        "analyze_and_route",
        route_to_subagents,
        ["call_tax", "call_compliance", "aggregate"],
    )

    graph.add_edge("call_tax", "aggregate")
    graph.add_edge("call_compliance", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()