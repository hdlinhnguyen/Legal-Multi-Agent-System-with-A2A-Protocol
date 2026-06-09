# RAG Supervisor-Worker Agent

> **Assignment:** Cải thiện Day08 RAG Pipeline bằng LangGraph Supervisor-Worker Pattern  
> **Topic:** Multi-Agent Systems — Day 9

---

## 🏗️ Kiến trúc: Supervisor-Worker Pattern

```
User Query
  └→ SupervisorAgent: analyze_query()
       │  (Phân tích query, quyết định strategy)
       │
       ├→ [PARALLEL via LangGraph Send() API]
       │    │
       │    ├→ Worker 1: SemanticWorker
       │    │   Dense vector search (sentence-transformers)
       │    │   → semantic_results[]
       │    │
       │    ├→ Worker 2: LexicalWorker
       │    │   BM25 keyword search
       │    │   → lexical_results[]
       │    │
       │    └→ Worker 3: PageIndexWorker (optional)
       │        Vectorless fallback search
       │        → pageindex_results[]
       │
       └→ Aggregator: aggregator()
            RRF Merge + Rerank + LLM Generation
            → final_answer (với citation)
```

---

## 🆚 So sánh với Day08 Monolithic RAG

| Aspect | Day08 (Monolithic) | Assignment (Supervisor-Worker) |
|---|---|---|
| **Search strategy** | Sequential (semantic → lexical → rerank) | Parallel (tất cả chạy đồng thời) |
| **Latency** | ~N×T (tổng tuần tự) | ~max(T) (parallel) |
| **Extensibility** | Sửa pipeline trực tiếp | Thêm Worker mới không ảnh hưởng logic cũ |
| **Fault tolerance** | 1 bước fail → toàn bộ fail | Worker fail → graceful fallback |
| **Supervisor intelligence** | Không có | Supervisor quyết định skip worker không cần thiết |
| **Pattern** | Function pipeline | LangGraph StateGraph + Send() |

---

## 📂 File Structure

```
rag_supervisor_agent/
├── rag_supervisor_graph.py   # Core: StateGraph với Supervisor + 3 Workers
├── app.py                    # FastAPI server (POST /ask)
├── test_supervisor.py        # Unit + integration tests
└── README.md                 # This file
```

---

## 🚀 Chạy thử

### Option 1: Run trực tiếp (demo)
```bash
cd Checklist_Assignment/Lab_Assignment/rag_supervisor_agent
python rag_supervisor_graph.py
```

### Option 2: FastAPI server
```bash
python app.py
# Server runs on http://localhost:8000

# Test với curl:
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "Hình phạt cho tội tàng trữ trái phép chất ma tuý?"}'
```

### Option 3: Run tests
```bash
python test_supervisor.py
```

---

## 🔑 Key Implementation Details

### 1. RAGState — Shared State
```python
class RAGState(TypedDict):
    query: str
    use_semantic: bool          # Supervisor decides which workers to use
    use_lexical: bool
    use_pageindex: bool
    semantic_results: Annotated[list, _list_merge]   # Reducer cho parallel
    lexical_results: Annotated[list, _list_merge]
    pageindex_results: Annotated[list, _list_merge]
    final_answer: str
    retrieval_strategy: str     # e.g. "semantic+lexical"
```

### 2. Parallel Dispatch với LangGraph Send()
```python
def route_to_workers(state: RAGState) -> list[Send]:
    sends = []
    if state["use_semantic"]:  sends.append(Send("semantic_worker", state))
    if state["use_lexical"]:   sends.append(Send("lexical_worker", state))
    if state["use_pageindex"]: sends.append(Send("pageindex_worker", state))
    return sends
```

### 3. Annotated Reducer — Merge parallel results
```python
def _list_merge(a: list, b: list) -> list:
    return a + b

semantic_results: Annotated[list, _list_merge]
```

---

## 🔗 Dependency

Dùng code từ Day08 RAG pipeline:
- `Day08-RAGpipeline/src/task5_semantic_search.py` → `semantic_search()`
- `Day08-RAGpipeline/src/task6_lexical_search.py` → `lexical_search()`
- `Day08-RAGpipeline/src/task7_reranking.py` → `rerank_rrf()`, `rerank()`
- `Day08-RAGpipeline/src/task8_pageindex_vectorless.py` → `pageindex_search()`
- `Day08-RAGpipeline/src/task10_generation.py` → `generate_with_citation()`
