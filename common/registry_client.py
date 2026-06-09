"""Registry client helpers — optimised for low latency.

Key optimization: `discover()` results are cached in memory with a TTL
(default 60 s).  This eliminates the HTTP round-trip to the registry on
every delegation (~50–100 ms per call) while still auto-refreshing when
agents restart.

Provides `discover(task)` to look up an agent endpoint from the registry,
and `register(agent_info)` for agents to self-register on startup.
"""

import os
import time

import httpx

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:10000")

# ---------------------------------------------------------------------------
# In-memory discover cache
# Structure: {task: (endpoint_str, expires_at_monotonic_float)}
# ---------------------------------------------------------------------------
_DISCOVER_TTL = 60.0  # seconds — refresh if agent restarts
_discover_cache: dict[str, tuple[str, float]] = {}


async def discover(task: str) -> str:
    """Return the endpoint URL of the agent that handles the given task.

    Results are cached for _DISCOVER_TTL seconds to avoid a registry
    round-trip on every delegation.

    Args:
        task: The task identifier (e.g. "legal_question", "tax_question").

    Returns:
        The HTTP endpoint base URL of the matching agent.

    Raises:
        httpx.HTTPStatusError: If no agent is found or the registry is unreachable.
    """
    now = time.monotonic()
    cached = _discover_cache.get(task)
    if cached and cached[1] > now:
        return cached[0]  # return cached endpoint

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{REGISTRY_URL}/discover/{task}")
        resp.raise_for_status()
        endpoint = resp.json()["endpoint"]

    _discover_cache[task] = (endpoint, now + _DISCOVER_TTL)
    return endpoint


async def register(agent_info: dict) -> None:
    """Register an agent with the registry.

    Args:
        agent_info: Dict with keys: agent_name, version, description,
                    tasks, endpoint, tags.

    Raises:
        httpx.HTTPStatusError: If registration fails.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{REGISTRY_URL}/register", json=agent_info)
        resp.raise_for_status()