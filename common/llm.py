"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.

Performance optimization: LLM instance is cached as a module-level singleton
to avoid re-instantiation on every agent node call.
"""

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a cached ChatOpenAI client pointed at OpenRouter.

    The instance is created once and reused for all subsequent calls,
    eliminating per-call overhead of object creation and configuration parsing.
    """
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.1,       # Lower temp = faster token generation
        max_tokens=800,         # Cap output tokens to reduce tail latency
    )


@lru_cache(maxsize=1)
def get_fast_llm() -> ChatOpenAI:
    """Return a lightweight LLM for routing/classification tasks only.

    Uses a faster/cheaper model to handle JSON routing decisions
    without spending time on a full-power model.
    """
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_FAST_MODEL", "openai/gpt-4o-mini"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.0,       # Deterministic for routing
        max_tokens=300,        # Routing JSON is short
    )