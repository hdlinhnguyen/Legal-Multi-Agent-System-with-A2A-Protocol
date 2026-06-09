"""
RAG Supervisor-Worker Agent — FastAPI Entry Point.

Chạy như một standalone web service, expose endpoint:
  POST /ask   → { "query": "câu hỏi" }
  GET  /health → { "status": "ok" }

Usage:
    cd Checklist_Assignment/Lab_Assignment/rag_supervisor_agent
    uv run python app.py
    # hoặc
    python app.py

Then test:
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"query": "Hình phạt cho tội tàng trữ ma tuý?"}'
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from rag_supervisor_graph import RAGState, create_rag_supervisor_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [rag_supervisor] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Supervisor-Worker Agent",
    description="LangGraph Supervisor-Worker pattern applied to Day08 RAG pipeline",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = create_rag_supervisor_graph()
    return _graph


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class AskResponse(BaseModel):
    query: str
    answer: str
    retrieval_strategy: str
    num_sources: int
    sources_preview: list[str]


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "RAG Supervisor-Worker"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Gửi câu hỏi qua Supervisor-Worker RAG pipeline.

    Flow:
    1. Supervisor phân tích query → quyết định workers
    2. Workers chạy song song (semantic + lexical ± pageindex)
    3. Aggregator: RRF merge → rerank → LLM generation
    4. Trả về answer có citation
    """
    graph = get_graph()

    initial_state: RAGState = {
        "query": request.query,
        "use_semantic": True,
        "use_lexical": True,
        "use_pageindex": False,
        "top_k": request.top_k,
        "semantic_results": [],
        "lexical_results": [],
        "pageindex_results": [],
        "merged_results": [],
        "final_answer": "",
        "sources_used": [],
        "retrieval_strategy": "",
    }

    result = await graph.ainvoke(initial_state)

    sources_preview = [
        src.get("content", "")[:100] + "..."
        for src in result.get("sources_used", [])[:3]
    ]

    return AskResponse(
        query=request.query,
        answer=result["final_answer"],
        retrieval_strategy=result["retrieval_strategy"],
        num_sources=len(result.get("sources_used", [])),
        sources_preview=sources_preview,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting RAG Supervisor-Worker Agent on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
