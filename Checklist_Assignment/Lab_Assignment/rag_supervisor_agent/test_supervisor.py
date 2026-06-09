"""
Tests cho RAG Supervisor-Worker Agent.

Chạy:
    cd Checklist_Assignment/Lab_Assignment/rag_supervisor_agent
    python test_supervisor.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_supervisor_graph import (
    RAGState,
    analyze_query,
    route_to_workers,
    create_rag_supervisor_graph,
)
from langgraph.types import Send


# =============================================================================
# Unit Tests
# =============================================================================

def test_supervisor_short_query():
    """Short query nên kích hoạt pageindex worker."""
    state: RAGState = {
        "query": "ma tuý", "use_semantic": False, "use_lexical": False,
        "use_pageindex": False, "top_k": 5,
        "semantic_results": [], "lexical_results": [], "pageindex_results": [],
        "merged_results": [], "final_answer": "", "sources_used": [], "retrieval_strategy": "",
    }
    result = analyze_query(state)
    assert result["use_semantic"] is True
    assert result["use_lexical"] is True
    assert result["use_pageindex"] is True, "Short query (2 words) should use pageindex"
    print("✅ test_supervisor_short_query passed")


def test_supervisor_law_article_query():
    """Query có 'điều' nên kích hoạt pageindex."""
    state: RAGState = {
        "query": "Điều 249 Bộ luật Hình sự quy định gì về tội tàng trữ",
        "use_semantic": False, "use_lexical": False, "use_pageindex": False,
        "top_k": 5,
        "semantic_results": [], "lexical_results": [], "pageindex_results": [],
        "merged_results": [], "final_answer": "", "sources_used": [], "retrieval_strategy": "",
    }
    result = analyze_query(state)
    assert result["use_pageindex"] is True, "Query with 'điều' should use pageindex"
    print("✅ test_supervisor_law_article_query passed")


def test_routing_returns_sends():
    """route_to_workers nên trả về list[Send]."""
    state: RAGState = {
        "query": "test", "use_semantic": True, "use_lexical": True, "use_pageindex": True,
        "top_k": 5,
        "semantic_results": [], "lexical_results": [], "pageindex_results": [],
        "merged_results": [], "final_answer": "", "sources_used": [], "retrieval_strategy": "",
    }
    sends = route_to_workers(state)
    assert len(sends) == 3, f"Expected 3 Send objects, got {len(sends)}"
    node_names = [s.node for s in sends]
    assert "semantic_worker" in node_names
    assert "lexical_worker" in node_names
    assert "pageindex_worker" in node_names
    print("✅ test_routing_returns_sends passed")


def test_routing_skip_pageindex():
    """Khi use_pageindex=False, chỉ dispatch 2 workers."""
    state: RAGState = {
        "query": "test", "use_semantic": True, "use_lexical": True, "use_pageindex": False,
        "top_k": 5,
        "semantic_results": [], "lexical_results": [], "pageindex_results": [],
        "merged_results": [], "final_answer": "", "sources_used": [], "retrieval_strategy": "",
    }
    sends = route_to_workers(state)
    assert len(sends) == 2, f"Expected 2 Send objects, got {len(sends)}"
    node_names = [s.node for s in sends]
    assert "pageindex_worker" not in node_names
    print("✅ test_routing_skip_pageindex passed")


async def test_full_graph_no_data():
    """Graph chạy thành công dù không có data (graceful empty results)."""
    graph = create_rag_supervisor_graph()
    initial_state: RAGState = {
        "query": "hình phạt tàng trữ ma tuý",
        "use_semantic": True, "use_lexical": True, "use_pageindex": False,
        "top_k": 3,
        "semantic_results": [], "lexical_results": [], "pageindex_results": [],
        "merged_results": [], "final_answer": "", "sources_used": [], "retrieval_strategy": "",
    }
    result = await graph.ainvoke(initial_state)
    assert "final_answer" in result
    assert "retrieval_strategy" in result
    print(f"✅ test_full_graph_no_data passed | strategy={result['retrieval_strategy']}")
    print(f"   Answer preview: {result['final_answer'][:100]}...")


# =============================================================================
# Run all tests
# =============================================================================

async def main():
    print("\n🧪 RAG Supervisor-Worker Agent Tests\n" + "="*50)

    # Sync tests
    test_supervisor_short_query()
    test_supervisor_law_article_query()
    test_routing_returns_sends()
    test_routing_skip_pageindex()

    # Async tests
    await test_full_graph_no_data()

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
