"""
RAG Supervisor-Worker Agent — Assignment Day08 Improved.

Pattern: Supervisor-Worker (Supervisor điều phối 3 Workers song song)

Architecture:
    User Query
      └→ Supervisor (analyze_query node)
           ├→ [PARALLEL via Send()]
           │    ├→ SemanticWorker  — Dense vector search
           │    ├→ LexicalWorker   — BM25 keyword search
           │    └→ PageIndexWorker — Vectorless fallback search
           └→ Aggregator (RRF merge + rerank + LLM generation)

Why Supervisor-Worker over monolithic RAG?
- Workers có thể chạy song song → giảm latency
- Supervisor có thể bỏ qua worker không cần thiết (e.g. skip PageIndex nếu đủ results)
- Dễ thêm worker mới (e.g. WebSearchWorker) mà không sửa logic cũ
- Mỗi worker có thể scale độc lập
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated, TypedDict

# Add Day08 RAG pipeline to path
RAG_SRC = Path(__file__).parent.parent.parent.parent / "Day08-RAGpipeline" / "src"
sys.path.insert(0, str(RAG_SRC.parent))

from langgraph.types import Send
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

# =============================================================================
# STATE DEFINITION
# =============================================================================

def _list_merge(a: list, b: list) -> list:
    """Reducer: merge lists from parallel workers."""
    return a + b


class RAGState(TypedDict):
    """Shared state across all Supervisor-Worker nodes."""
    query: str
    use_semantic: bool
    use_lexical: bool
    use_pageindex: bool
    top_k: int
    # Parallel worker results (reducer để merge từ nhiều workers)
    semantic_results: Annotated[list, _list_merge]
    lexical_results: Annotated[list, _list_merge]
    pageindex_results: Annotated[list, _list_merge]
    # Aggregated final results
    merged_results: list
    final_answer: str
    sources_used: list
    retrieval_strategy: str


# =============================================================================
# SUPERVISOR NODE — Phân tích query và quyết định workers
# =============================================================================

def analyze_query(state: RAGState) -> dict:
    """
    Supervisor node: Phân tích query để quyết định strategy.

    Logic:
    - Query ngắn (< 5 từ) → dùng cả 3 workers
    - Query có từ khóa pháp lý cụ thể → ưu tiên lexical + semantic
    - Query mơ hồ → ưu tiên semantic + pageindex fallback
    """
    query = state["query"]
    words = query.split()

    # Luôn dùng semantic + lexical
    use_semantic = True
    use_lexical = True

    # PageIndex: chỉ dùng khi query ngắn hoặc có thể cần fallback
    use_pageindex = len(words) <= 4 or any(
        kw in query.lower()
        for kw in ["điều", "khoản", "nghị định", "thông tư", "luật số"]
    )

    logger.info(
        "Supervisor analyzed query=%r | semantic=%s lexical=%s pageindex=%s",
        query[:60], use_semantic, use_lexical, use_pageindex,
    )

    return {
        "use_semantic": use_semantic,
        "use_lexical": use_lexical,
        "use_pageindex": use_pageindex,
    }


# =============================================================================
# ROUTING FUNCTION — Dispatch workers theo quyết định của Supervisor
# =============================================================================

def route_to_workers(state: RAGState) -> list[Send]:
    """
    Routing function trả về list[Send] để LangGraph dispatch song song.

    Pattern giống Stage 4 law_agent: route_to_subagents()
    """
    sends: list[Send] = []

    if state.get("use_semantic", True):
        sends.append(Send("semantic_worker", state))

    if state.get("use_lexical", True):
        sends.append(Send("lexical_worker", state))

    if state.get("use_pageindex", False):
        sends.append(Send("pageindex_worker", state))

    if not sends:
        sends.append(Send("aggregator", state))

    return sends


# =============================================================================
# WORKER NODES — 3 workers chạy song song
# =============================================================================

def semantic_worker(state: RAGState) -> dict:
    """
    Worker 1: Dense vector search (semantic similarity).

    Dùng sentence-transformers để embed query và tìm cosine similarity
    với vector store từ Day08 Task5.
    """
    query = state["query"]
    top_k = state.get("top_k", 5)

    try:
        from src.task5_semantic_search import semantic_search
        results = semantic_search(query, top_k=top_k * 2)
        for r in results:
            r["worker"] = "semantic"
        logger.info("SemanticWorker: returned %d results for query=%r", len(results), query[:40])
        return {"semantic_results": results}
    except Exception as exc:
        logger.warning("SemanticWorker failed: %s", exc)
        return {"semantic_results": []}


def lexical_worker(state: RAGState) -> dict:
    """
    Worker 2: BM25 keyword/lexical search.

    Tìm kiếm dựa trên từ khóa, tốt cho các truy vấn chứa thuật ngữ pháp lý cụ thể
    như số điều luật, tên văn bản, v.v.
    """
    query = state["query"]
    top_k = state.get("top_k", 5)

    try:
        from src.task6_lexical_search import lexical_search
        results = lexical_search(query, top_k=top_k * 2)
        for r in results:
            r["worker"] = "lexical"
        logger.info("LexicalWorker: returned %d results for query=%r", len(results), query[:40])
        return {"lexical_results": results}
    except Exception as exc:
        logger.warning("LexicalWorker failed: %s", exc)
        return {"lexical_results": []}


def pageindex_worker(state: RAGState) -> dict:
    """
    Worker 3: PageIndex vectorless search (fallback).

    Không cần vector store, tìm kiếm dựa trên page-level indexing.
    Dùng khi query liên quan đến điều khoản cụ thể hoặc khi 2 workers kia không đủ.
    """
    query = state["query"]
    top_k = state.get("top_k", 5)

    try:
        from src.task8_pageindex_vectorless import pageindex_search
        results = pageindex_search(query, top_k=top_k)
        for r in results:
            r["worker"] = "pageindex"
        logger.info("PageIndexWorker: returned %d results for query=%r", len(results), query[:40])
        return {"pageindex_results": results}
    except Exception as exc:
        logger.warning("PageIndexWorker failed: %s", exc)
        return {"pageindex_results": []}


# =============================================================================
# AGGREGATOR NODE — Merge, rerank, và generate
# =============================================================================

def aggregator(state: RAGState) -> dict:
    """
    Aggregator: Merge results từ tất cả workers, rerank, và generate với citation.

    Steps:
    1. Collect all worker results
    2. RRF merge (Reciprocal Rank Fusion)
    3. Rerank để lọc top_k tốt nhất
    4. Generate với citation qua LLM
    """
    query = state["query"]
    top_k = state.get("top_k", 5)

    # Collect từ tất cả workers
    all_result_lists = []
    workers_used = []

    if state.get("semantic_results"):
        all_result_lists.append(state["semantic_results"])
        workers_used.append("semantic")

    if state.get("lexical_results"):
        all_result_lists.append(state["lexical_results"])
        workers_used.append("lexical")

    if state.get("pageindex_results"):
        all_result_lists.append(state["pageindex_results"])
        workers_used.append("pageindex")

    logger.info("Aggregator: merging from workers %s", workers_used)

    # Merge bằng RRF
    merged = []
    if all_result_lists:
        try:
            from src.task7_reranking import rerank_rrf, rerank
            merged = rerank_rrf(all_result_lists, top_k=top_k * 2)

            # Rerank nếu có đủ kết quả
            if len(merged) >= 2:
                merged = rerank(query, merged, top_k=top_k, method="cross_encoder")
        except Exception as exc:
            logger.warning("Reranking failed: %s — using simple merge", exc)
            # Fallback: simple concat + dedup by content
            seen = set()
            for lst in all_result_lists:
                for item in lst:
                    key = item.get("content", "")[:100]
                    if key not in seen:
                        seen.add(key)
                        merged.append(item)
            merged = merged[:top_k]

    # Generate với citation
    answer = _generate_answer(query, merged)

    return {
        "merged_results": merged[:top_k],
        "final_answer": answer,
        "sources_used": merged[:top_k],
        "retrieval_strategy": "+".join(workers_used) if workers_used else "none",
    }


def _generate_answer(query: str, chunks: list[dict]) -> str:
    """Generate answer với citation từ chunks."""
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    try:
        from src.task10_generation import (
            generate_with_citation,
            reorder_for_llm,
            format_context,
            SYSTEM_PROMPT,
            OPENROUTER_API_KEY,
            OPENROUTER_ENDPOINT,
            LLM_MODEL,
            TEMPERATURE,
            TOP_P,
        )
        import requests

        reordered = reorder_for_llm(chunks)
        context = format_context(reordered)
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        if not OPENROUTER_API_KEY:
            return "Chưa cấu hình OPENROUTER_API_KEY. Vui lòng thêm vào .env file."

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": 512,
        }
        response = requests.post(
            OPENROUTER_ENDPOINT,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        if "choices" in result and result["choices"]:
            return result["choices"][0].get("message", {}).get("content", "")
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        return f"Lỗi khi generate: {exc}"


# =============================================================================
# GRAPH CONSTRUCTION — Build Supervisor-Worker StateGraph
# =============================================================================

def create_rag_supervisor_graph():
    """
    Build và compile LangGraph StateGraph với pattern Supervisor-Worker.

    Graph topology:
        [START]
          └→ analyze_query (Supervisor)
               └→ [conditional parallel edges via route_to_workers]
                    ├→ semantic_worker
                    ├→ lexical_worker
                    └→ pageindex_worker
                         └→ aggregator → [END]
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("analyze_query", analyze_query)      # Supervisor
    graph.add_node("semantic_worker", semantic_worker)   # Worker 1
    graph.add_node("lexical_worker", lexical_worker)     # Worker 2
    graph.add_node("pageindex_worker", pageindex_worker) # Worker 3
    graph.add_node("aggregator", aggregator)             # Aggregator

    # Entry point
    graph.set_entry_point("analyze_query")

    # Supervisor → parallel workers via conditional edges
    graph.add_conditional_edges(
        "analyze_query",
        route_to_workers,
        ["semantic_worker", "lexical_worker", "pageindex_worker", "aggregator"],
    )

    # All workers → aggregator
    graph.add_edge("semantic_worker", "aggregator")
    graph.add_edge("lexical_worker", "aggregator")
    graph.add_edge("pageindex_worker", "aggregator")

    # Aggregator → END
    graph.add_edge("aggregator", END)

    return graph.compile()


# =============================================================================
# MAIN — Demo chạy thử
# =============================================================================

async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    graph = create_rag_supervisor_graph()

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Điều 249 Bộ luật Hình sự quy định gì?",
        "Những nghệ sĩ nào bị bắt vì liên quan tới ma tuý năm 2024?",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print("=" * 70)

        initial_state: RAGState = {
            "query": query,
            "use_semantic": True,
            "use_lexical": True,
            "use_pageindex": False,
            "top_k": 5,
            "semantic_results": [],
            "lexical_results": [],
            "pageindex_results": [],
            "merged_results": [],
            "final_answer": "",
            "sources_used": [],
            "retrieval_strategy": "",
        }

        result = await graph.ainvoke(initial_state)

        print(f"\nStrategy: {result['retrieval_strategy']}")
        print(f"Sources: {len(result['sources_used'])} chunks")
        print(f"\nAnswer:\n{result['final_answer'][:500]}...")


if __name__ == "__main__":
    asyncio.run(main())
