"""A2A delegation helper — optimised for low latency.

Key optimizations vs original:
1. **Persistent httpx.AsyncClient** — reuses TCP connections (connection pool).
   Previously, each call created and immediately closed an httpx client,
   incurring a full TCP+TLS handshake on every delegation (~100–200 ms).

2. **Agent card cache** — the /.well-known/agent.json fetch is done at most
   once per unique endpoint.  Previously fetched on every delegation (~200 ms).

3. Registry discover results are cached separately in registry_client.py.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import httpx

from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    TextPart,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level persistent HTTP client
# Reused across all delegation calls — avoids per-call TCP handshake overhead.
# ---------------------------------------------------------------------------
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=120.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _http_client


# ---------------------------------------------------------------------------
# Agent card cache — keyed by endpoint URL
# Saves ~200–400 ms per delegation by not re-fetching the agent manifest.
# ---------------------------------------------------------------------------
_agent_card_cache: dict[str, AgentCard] = {}


async def _get_agent_card(endpoint: str) -> AgentCard:
    if endpoint not in _agent_card_cache:
        client = _get_http_client()
        card_url = f"{endpoint}/.well-known/agent.json"
        card_resp = await client.get(card_url)
        card_resp.raise_for_status()
        _agent_card_cache[endpoint] = AgentCard.model_validate(card_resp.json())
        logger.debug("Agent card fetched and cached for %s", endpoint)
    return _agent_card_cache[endpoint]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def delegate(
    endpoint: str,
    question: str,
    context_id: str,
    trace_id: str,
    depth: int,
) -> str:
    """Send a question to an A2A agent and return the text response.

    Args:
        endpoint: Base URL of the target agent (e.g. "http://localhost:10101").
        question: The question to ask.
        context_id: Current A2A context ID to propagate.
        trace_id: Trace ID generated at the Customer Agent; propagated throughout.
        depth: Current delegation depth (used to enforce MAX_DELEGATION_DEPTH).

    Returns:
        The agent's text response, or an empty string if none could be extracted.
    """
    http_client = _get_http_client()

    # Fetch (or reuse cached) agent card
    agent_card = await _get_agent_card(endpoint)

    # Build A2AClient using the shared persistent http client
    client = A2AClient(httpx_client=http_client, agent_card=agent_card)

    # Build message with trace metadata
    message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=question))],
        message_id=str(uuid4()),
        context_id=context_id,
        metadata={
            "trace_id": trace_id,
            "context_id": context_id,
            "delegation_depth": depth,
        },
    )

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(message=message),
    )

    logger.debug(
        "Delegating to %s (depth=%d, trace=%s)", endpoint, depth, trace_id
    )

    response = await client.send_message(request)

    # Extract text from SendMessageResponse
    return _extract_text(response)


def _extract_text(response: object) -> str:
    """Walk the response tree and collect all TextPart.text values."""
    text = ""

    # Unwrap root if it's a RootModel
    if hasattr(response, "root"):
        response = response.root

    # SendMessageSuccessResponse has a .result (Task | Message)
    result = getattr(response, "result", None)
    if result is None:
        return text

    # Task — text lives in artifacts
    artifacts = getattr(result, "artifacts", None)
    if artifacts:
        for artifact in artifacts:
            parts = getattr(artifact, "parts", []) or []
            for part in parts:
                text += _part_text(part)
        if text:
            return text

    # Message — text lives in parts directly
    parts = getattr(result, "parts", None)
    if parts:
        for part in parts:
            text += _part_text(part)

    # Task history messages as fallback
    if not text:
        history = getattr(result, "history", None)
        if history:
            for msg in history:
                msg_parts = getattr(msg, "parts", []) or []
                for part in msg_parts:
                    text += _part_text(part)

    return text


def _part_text(part: object) -> str:
    """Extract text from a Part object (handling both Part(root=TextPart) and raw TextPart)."""
    inner = getattr(part, "root", part)
    return getattr(inner, "text", "") or ""