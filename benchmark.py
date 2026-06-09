"""Latency benchmark script.

Sends a test question through the full agent chain and measures end-to-end
response time. Run AFTER all agents are started.

Usage:
    python benchmark.py
"""

import asyncio
import time

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

CUSTOMER_ENDPOINT = "http://localhost:10100"
TEST_QUESTION = (
    "Our company has been hiding $2M in offshore accounts via shell companies "
    "in the Cayman Islands for 3 years to avoid corporate taxes. "
    "What are the legal and tax consequences we face?"
)


async def run_benchmark(label: str = "Test"):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Question: {TEST_QUESTION[:80]}...")

    async with httpx.AsyncClient(timeout=300.0) as http:
        # Fetch agent card
        card_resp = await http.get(f"{CUSTOMER_ENDPOINT}/.well-known/agent.json")
        card_resp.raise_for_status()
        agent_card = AgentCard.model_validate(card_resp.json())

        client = A2AClient(httpx_client=http, agent_card=agent_card)

        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=TEST_QUESTION))],
            message_id="bench-001",
            context_id="bench-ctx-001",
        )
        request = SendMessageRequest(
            id="bench-req-001",
            params=MessageSendParams(message=message),
        )

        print(f"\n⏱️  Sending request...")
        t_start = time.perf_counter()
        response = await client.send_message(request)
        t_end = time.perf_counter()

    elapsed = t_end - t_start
    print(f"\n✅ Response received in: {elapsed:.2f}s")

    # Extract answer
    answer = ""
    resp_root = getattr(response, "root", response)
    result = getattr(resp_root, "result", None)
    if result:
        artifacts = getattr(result, "artifacts", None)
        if artifacts:
            for artifact in artifacts:
                for part in getattr(artifact, "parts", []) or []:
                    inner = getattr(part, "root", part)
                    answer += getattr(inner, "text", "") or ""

    if answer:
        print(f"\n📝 Answer preview:\n{answer[:300]}...")
    else:
        print("(no text extracted from response)")

    return elapsed


async def main():
    print("\n🚀 Agent Latency Benchmark")
    print("Make sure all agents are running (start_all.sh)")

    # Warm-up: first call may have cold-start overhead (registry lookup, model load)
    print("\n--- Warm-up run (first call, cold caches) ---")
    t1 = await run_benchmark("Run 1 — Cold start (registry cache miss, no TCP keep-alive)")

    # Second call: caches warm (registry + agent card cached)
    print("\n--- Second run (warm caches) ---")
    t2 = await run_benchmark("Run 2 — Warm (registry cached, TCP connection reused)")

    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Cold start (Run 1):  {t1:.2f}s")
    print(f"  Warm cache (Run 2):  {t2:.2f}s")
    print(f"  Cache speedup:       {t1 - t2:.2f}s saved")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
