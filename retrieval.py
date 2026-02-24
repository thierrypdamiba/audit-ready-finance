"""4-stage retrieval pipeline: plan, graph search, vector search, generate."""

import asyncio
import os
import time
from datetime import datetime, timezone
from functools import lru_cache

os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from audit_trail import log_query
from claude_client import generate_answer
from config import (
    COLLECTION_NAME,
    CLAUDE_MODEL,
    COMPANY_ALIASES,
    COMPANY_NAMES,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    TICKERS,
)

# Reuse clients instead of creating new ones per request
_qdrant_client = None
_openai_client = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


@lru_cache(maxsize=128)
def embed_query(query: str) -> tuple:
    """Embed query text. Cached to avoid re-embedding repeated queries."""
    client = get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    return tuple(response.data[0].embedding)


# Cache cognee graph search results (cognee's internal vector retrieval is slow)
_graph_cache: dict[str, tuple[float, list[dict]]] = {}
_GRAPH_CACHE_TTL = 300  # 5 minutes


# -- Stage 1: Build Retrieval Plan --

def build_retrieval_plan(query: str) -> dict:
    """Analyze query to determine retrieval strategy before executing anything."""
    query_lower = query.lower()

    # Detect mentioned companies using aliases
    target_companies = []
    for ticker in TICKERS:
        aliases = COMPANY_ALIASES.get(ticker, [ticker.lower()])
        if any(alias in query_lower for alias in aliases):
            target_companies.append(ticker)

    # If no specific company mentioned, search all
    if not target_companies:
        target_companies = list(TICKERS)

    # Detect relevant sections
    target_sections = []
    section_keywords = {
        "risk_factors": ["risk", "risks", "threat", "challenge", "exposure", "vulnerability"],
        "md_and_a": ["revenue", "growth", "financial", "performance", "income", "profit", "loss", "strategy", "cash", "margin"],
        "business": ["business", "product", "service", "operation", "segment", "market"],
        "controls": ["control", "audit", "compliance", "procedure", "governance", "regulation", "regulatory"],
    }

    for section, keywords in section_keywords.items():
        if any(kw in query_lower for kw in keywords):
            target_sections.append(section)

    # If no specific section detected, search all
    if not target_sections:
        target_sections = list(section_keywords.keys())

    # Detect comparison queries
    is_comparison = any(w in query_lower for w in ["compare", "versus", "vs", "differ", "between", "lower", "higher", "more", "less"])

    plan = {
        "query": query,
        "target_companies": target_companies,
        "target_sections": target_sections,
        "is_comparison": is_comparison,
        "search_strategy": "multi_company_comparison" if is_comparison and len(target_companies) > 1 else "focused_search",
        "planned_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    plan["steps"].append({
        "stage": 1,
        "action": "build_retrieval_plan",
        "description": f"Identified {len(target_companies)} companies, {len(target_sections)} sections",
    })
    plan["steps"].append({
        "stage": 2,
        "action": "graph_search",
        "description": "Search cognee knowledge graph for entity relationships",
    })
    plan["steps"].append({
        "stage": 3,
        "action": "vector_search",
        "description": f"Filtered Qdrant search on companies={target_companies}, sections={target_sections}",
    })
    plan["steps"].append({
        "stage": 4,
        "action": "generate_answer",
        "description": "Generate answer with Claude Citations API",
    })

    return plan


# -- Stage 2: Graph Search (Cognee) --

async def graph_search(query: str) -> list[dict]:
    """Search cognee knowledge graph for entity relationships."""
    # Check cache first
    if query in _graph_cache:
        cached_time, cached_result = _graph_cache[query]
        if time.perf_counter() - cached_time < _GRAPH_CACHE_TTL:
            return cached_result

    try:
        import cognee_community_vector_adapter_qdrant.register  # noqa: F401
        import cognee
        from cognee.api.v1.search import SearchType

        # Point cognee to shipped databases (Kuzu graph + SQLite metadata)
        cognee_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cognee_data")
        if os.path.isdir(cognee_data_dir):
            cognee.config.system_root_directory(cognee_data_dir)

        cognee.config.set_llm_config({
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "llm_api_key": OPENAI_API_KEY,
        })
        cognee.config.set_vector_db_config({
            "vector_db_provider": "qdrant",
            "vector_db_url": QDRANT_URL,
            "vector_db_key": QDRANT_API_KEY,
        })

        # Timeout graph search at 15 seconds (cognee vector retrieval ~7s + graph ~3s)
        results = await asyncio.wait_for(
            cognee.search(
                query_text=query,
                query_type=SearchType.GRAPH_COMPLETION,
            ),
            timeout=15.0,
        )

        entities = []
        if results:
            for item in results:
                if isinstance(item, dict):
                    entities.append(item)
                elif isinstance(item, (list, tuple)):
                    for triple in item:
                        if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                            entities.append({
                                "subject": str(triple[0]),
                                "predicate": str(triple[1]),
                                "object": str(triple[2]),
                            })
                        elif hasattr(triple, "__dict__"):
                            entities.append({
                                "subject": getattr(triple, "name", str(triple)),
                                "type": getattr(triple, "type", "unknown"),
                            })
                else:
                    if hasattr(item, "__dict__"):
                        entities.append({k: str(v)[:200] for k, v in vars(item).items() if not k.startswith("_")})
                    else:
                        entities.append({"raw": str(item)[:200]})

        _graph_cache[query] = (time.perf_counter(), entities)
        return entities

    except asyncio.TimeoutError:
        print("cognee search timed out after 15s")
        return [{"error": "graph search timed out", "fallback": True}]
    except Exception as ex:
        print(f"cognee search error: {ex}")
        return [{"error": str(ex), "fallback": True}]


# -- Stage 3: Filtered Vector Search (Qdrant) --

def _vector_search_single(
    client: QdrantClient,
    query_vector: list[float],
    ticker: str,
    target_sections: list[str] | None,
    tenant_id: str | None,
    top_k: int,
) -> list[dict]:
    """Search for a single company. Used by vector_search for diversity."""
    must = [FieldCondition(key="ticker", match=MatchValue(value=ticker))]

    if target_sections and len(target_sections) < 4:
        if len(target_sections) == 1:
            must.append(FieldCondition(key="section", match=MatchValue(value=target_sections[0])))

    if tenant_id:
        must.append(FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)))

    should = []
    if target_sections and len(target_sections) > 1 and len(target_sections) < 4:
        should = [FieldCondition(key="section", match=MatchValue(value=s)) for s in target_sections]

    filter_kwargs = {"must": must}
    if should:
        filter_kwargs["should"] = should

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(**filter_kwargs),
        limit=top_k,
        with_payload=True,
    )

    chunks = []
    for hit in response.points:
        payload = hit.payload or {}
        chunks.append({
            "text": payload.get("text", ""),
            "company": payload.get("company", ""),
            "ticker": payload.get("ticker", ""),
            "section": payload.get("section", ""),
            "item_number": payload.get("item_number", ""),
            "fiscal_year": payload.get("fiscal_year", ""),
            "chunk_index": payload.get("chunk_index", 0),
            "tenant_id": payload.get("tenant_id", ""),
            "score": hit.score,
            "point_id": str(hit.id),
        })
    return chunks


def vector_search(
    query: str,
    target_companies: list[str] | None = None,
    target_sections: list[str] | None = None,
    tenant_id: str | None = None,
    top_k: int = 10,
    ensure_diversity: bool = True,
) -> list[dict]:
    """Search Qdrant with metadata filters and update access counts."""
    client = get_qdrant_client()
    query_vector = list(embed_query(query))

    # For multi-company queries, search per-company to ensure diversity
    if ensure_diversity and target_companies and len(target_companies) > 1:
        per_company_k = max(2, top_k // len(target_companies))
        all_results = []
        for ticker in target_companies:
            company_results = _vector_search_single(
                client, query_vector, ticker, target_sections, tenant_id, per_company_k,
            )
            all_results.extend(company_results)
        # Sort by score and trim to top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    # Single company or no company filter
    must_conditions = []

    if target_companies and len(target_companies) == 1:
        must_conditions.append(
            FieldCondition(key="ticker", match=MatchValue(value=target_companies[0]))
        )

    if target_sections and len(target_sections) < 4:
        if len(target_sections) == 1:
            must_conditions.append(
                FieldCondition(key="section", match=MatchValue(value=target_sections[0]))
            )
        # For multiple sections, use should filter
        # (handled below via should_conditions)

    if tenant_id:
        must_conditions.append(
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
        )

    should_conditions = []
    if target_sections and len(target_sections) > 1 and len(target_sections) < 4:
        should_conditions = [
            FieldCondition(key="section", match=MatchValue(value=s))
            for s in target_sections
        ]

    search_filter = None
    if must_conditions or should_conditions:
        filter_kwargs = {}
        if must_conditions:
            filter_kwargs["must"] = must_conditions
        if should_conditions:
            filter_kwargs["should"] = should_conditions
        search_filter = Filter(**filter_kwargs)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )

    # Update access counts
    chunks = []
    for hit in response.points:
        payload = hit.payload or {}
        current_count = payload.get("access_count", 0)
        try:
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "access_count": current_count + 1,
                    "last_accessed": datetime.now(timezone.utc).isoformat(),
                },
                points=[hit.id],
            )
        except Exception:
            pass

        chunks.append({
            "text": payload.get("text", ""),
            "company": payload.get("company", ""),
            "ticker": payload.get("ticker", ""),
            "section": payload.get("section", ""),
            "item_number": payload.get("item_number", ""),
            "fiscal_year": payload.get("fiscal_year", ""),
            "chunk_index": payload.get("chunk_index", 0),
            "tenant_id": payload.get("tenant_id", ""),
            "score": hit.score,
            "point_id": str(hit.id),
        })

    return chunks


# -- Stage 4: Full Pipeline --

async def query_pipeline_stream(
    query: str,
    tenant_id: str | None = None,
    top_k: int = 8,
):
    """Streaming version of query_pipeline. Yields (event, data) tuples for SSE."""
    import json as _json
    total_start = time.perf_counter()

    # Stage 1
    yield ("step", {"stage": 1, "status": "running", "title": "Building retrieval plan", "detail": "Analyzing query intent..."})
    plan = build_retrieval_plan(query)
    companies = plan["target_companies"]
    sections = plan["target_sections"]
    company_names = {t: COMPANY_NAMES.get(t, t) for t in companies}

    # Thinking: show reasoning
    yield ("thinking", {"stage": 1, "text": f"Detected companies: {', '.join(company_names.values())}"})
    yield ("thinking", {"stage": 1, "text": f"Relevant sections: {', '.join(sections)}"})
    if plan["is_comparison"]:
        yield ("thinking", {"stage": 1, "text": f"This is a comparison query, will use multi-company retrieval"})
    yield ("thinking", {"stage": 1, "text": f"Strategy: {plan['search_strategy']}"})

    yield ("step", {
        "stage": 1, "status": "done", "title": "Building retrieval plan",
        "detail": f"Targeting {', '.join(company_names.values())} across {len(sections)} section{'s' if len(sections) != 1 else ''}",
    })

    # Stage 2 + 3 in parallel
    yield ("step", {"stage": 2, "status": "running", "title": "Searching knowledge graph", "detail": "Querying Cognee for entity relationships..."})
    yield ("thinking", {"stage": 2, "text": "Connecting to Cognee GraphRAG engine..."})
    yield ("step", {"stage": 3, "status": "running", "title": "Searching vector database", "detail": f"Querying Qdrant with filters..."})
    yield ("thinking", {"stage": 3, "text": f"Collection: {COLLECTION_NAME} ({len(companies)} company filter{'s' if len(companies) != 1 else ''})"})
    if len(companies) > 1:
        yield ("thinking", {"stage": 3, "text": f"Using diversified search: {top_k // len(companies)} chunks per company"})

    async def _vector_search_async():
        return vector_search(query=query, target_companies=companies, target_sections=sections, tenant_id=tenant_id, top_k=top_k)

    graph_task = asyncio.create_task(graph_search(query))
    vector_task = asyncio.create_task(_vector_search_async())
    graph_entities, vector_results = await asyncio.gather(graph_task, vector_task)

    has_fallback = any(e.get("fallback") for e in graph_entities if isinstance(e, dict))
    if not has_fallback and graph_entities:
        for ent in graph_entities[:3]:
            if isinstance(ent, dict):
                preview = ent.get("raw", ent.get("subject", str(ent)))[:100]
                yield ("thinking", {"stage": 2, "text": preview})
    graph_detail = f"Found {len(graph_entities)} entity relationship{'s' if len(graph_entities) != 1 else ''}" if not has_fallback else "Graph unavailable, continuing with vector results"
    yield ("step", {"stage": 2, "status": "done", "title": "Searching knowledge graph", "detail": graph_detail})

    vec_companies = set(r["ticker"] for r in vector_results)
    # Show what was retrieved
    for ticker in vec_companies:
        count = sum(1 for r in vector_results if r["ticker"] == ticker)
        top_section = max(set(r["section"] for r in vector_results if r["ticker"] == ticker), key=lambda s: sum(1 for r in vector_results if r["ticker"] == ticker and r["section"] == s))
        yield ("thinking", {"stage": 3, "text": f"{COMPANY_NAMES.get(ticker, ticker)}: {count} chunks (top section: {top_section})"})

    yield ("step", {"stage": 3, "status": "done", "title": "Searching vector database", "detail": f"Retrieved {len(vector_results)} chunks from {', '.join(vec_companies)}"})

    plan["steps"][1]["result_count"] = len(graph_entities)
    plan["steps"][2]["result_count"] = len(vector_results)
    retrieval_latency_ms = (time.perf_counter() - total_start) * 1000

    # Stage 4
    yield ("step", {"stage": 4, "status": "running", "title": "Generating answer", "detail": "Sending documents to Claude..."})
    yield ("thinking", {"stage": 4, "text": f"Building {len(vector_results)} document blocks with citation metadata"})
    yield ("thinking", {"stage": 4, "text": f"Model: {CLAUDE_MODEL} with native citations API"})
    yield ("thinking", {"stage": 4, "text": "Waiting for Claude to analyze and cite sources..."})

    generation_start = time.perf_counter()
    generation_result = generate_answer(query, vector_results)
    generation_latency_ms = (time.perf_counter() - generation_start) * 1000
    total_latency_ms = (time.perf_counter() - total_start) * 1000

    n_cites = len(generation_result["citations"])
    yield ("thinking", {"stage": 4, "text": f"Received {generation_result['output_tokens']} output tokens with {n_cites} inline citations"})
    yield ("step", {"stage": 4, "status": "done", "title": "Generating answer", "detail": f"Generated answer with {n_cites} citation{'s' if n_cites != 1 else ''}"})

    plan["steps"][3]["citation_count"] = n_cites

    provenance_path = {
        "query": query,
        "plan": {"companies": plan["target_companies"], "sections": plan["target_sections"], "strategy": plan["search_strategy"]},
        "graph_entities": graph_entities[:10],
        "vector_documents": [{"company": r["company"], "section": r["section"], "score": r["score"], "chunk_preview": r["text"][:100]} for r in vector_results],
        "answer_preview": generation_result["answer"][:200],
        "citation_count": n_cites,
    }

    vector_results_for_log = [{k: v for k, v in r.items() if k != "text"} for r in vector_results]
    audit_id = log_query(
        query=query, retrieval_plan=plan, graph_entities=graph_entities,
        vector_results=vector_results_for_log, answer=generation_result["answer"],
        citations=generation_result["citations"], provenance_path=provenance_path,
        model_used=generation_result.get("model", "claude-haiku-4-5-20251001"),
        input_tokens=generation_result["input_tokens"], output_tokens=generation_result["output_tokens"],
        retrieval_latency_ms=retrieval_latency_ms, generation_latency_ms=generation_latency_ms,
        total_latency_ms=total_latency_ms,
    )

    result = {
        "audit_id": audit_id,
        "answer": generation_result["answer"],
        "citations": generation_result["citations"],
        "citation_mode": generation_result["citation_mode"],
        "retrieval_plan": plan,
        "graph_entities": graph_entities,
        "vector_results": [{"company": r["company"], "ticker": r["ticker"], "section": r["section"], "fiscal_year": r["fiscal_year"], "score": r["score"], "text_preview": r["text"][:200], "chunk_index": r["chunk_index"]} for r in vector_results],
        "provenance_path": provenance_path,
        "latency": {"retrieval_ms": round(retrieval_latency_ms, 1), "generation_ms": round(generation_latency_ms, 1), "total_ms": round(total_latency_ms, 1)},
        "tokens": {"input": generation_result["input_tokens"], "output": generation_result["output_tokens"]},
    }
    yield ("result", result)


async def query_pipeline(
    query: str,
    tenant_id: str | None = None,
    top_k: int = 8,
) -> dict:
    """Execute the full 4-stage retrieval pipeline."""
    total_start = time.perf_counter()

    # Stage 1: Build retrieval plan
    plan = build_retrieval_plan(query)

    # Stage 2 + 3: Graph search and vector search in parallel
    retrieval_start = time.perf_counter()

    async def _vector_search_async():
        return vector_search(
            query=query,
            target_companies=plan["target_companies"],
            target_sections=plan["target_sections"],
            tenant_id=tenant_id,
            top_k=top_k,
        )

    graph_task = asyncio.create_task(graph_search(query))
    vector_task = asyncio.create_task(_vector_search_async())
    graph_entities, vector_results = await asyncio.gather(graph_task, vector_task)

    plan["steps"][1]["result_count"] = len(graph_entities)
    plan["steps"][2]["result_count"] = len(vector_results)
    retrieval_end = time.perf_counter()
    retrieval_latency_ms = (retrieval_end - retrieval_start) * 1000

    # Stage 4: Generate answer with citations
    generation_start = time.perf_counter()
    generation_result = generate_answer(query, vector_results)
    generation_end = time.perf_counter()
    generation_latency_ms = (generation_end - generation_start) * 1000

    total_latency_ms = (generation_end - total_start) * 1000

    plan["steps"][3]["citation_count"] = len(generation_result["citations"])

    # Build provenance path
    provenance_path = {
        "query": query,
        "plan": {
            "companies": plan["target_companies"],
            "sections": plan["target_sections"],
            "strategy": plan["search_strategy"],
        },
        "graph_entities": graph_entities[:10],
        "vector_documents": [
            {
                "company": r["company"],
                "section": r["section"],
                "score": r["score"],
                "chunk_preview": r["text"][:100],
            }
            for r in vector_results
        ],
        "answer_preview": generation_result["answer"][:200],
        "citation_count": len(generation_result["citations"]),
    }

    # Log to audit trail
    vector_results_for_log = [
        {k: v for k, v in r.items() if k != "text"}
        for r in vector_results
    ]

    audit_id = log_query(
        query=query,
        retrieval_plan=plan,
        graph_entities=graph_entities,
        vector_results=vector_results_for_log,
        answer=generation_result["answer"],
        citations=generation_result["citations"],
        provenance_path=provenance_path,
        model_used=generation_result.get("model", "claude-haiku-4-5-20251001"),
        input_tokens=generation_result["input_tokens"],
        output_tokens=generation_result["output_tokens"],
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        total_latency_ms=total_latency_ms,
    )

    return {
        "audit_id": audit_id,
        "answer": generation_result["answer"],
        "citations": generation_result["citations"],
        "citation_mode": generation_result["citation_mode"],
        "retrieval_plan": plan,
        "graph_entities": graph_entities,
        "vector_results": [
            {
                "company": r["company"],
                "ticker": r["ticker"],
                "section": r["section"],
                "fiscal_year": r["fiscal_year"],
                "score": r["score"],
                "text_preview": r["text"][:200],
                "chunk_index": r["chunk_index"],
            }
            for r in vector_results
        ],
        "provenance_path": provenance_path,
        "latency": {
            "retrieval_ms": round(retrieval_latency_ms, 1),
            "generation_ms": round(generation_latency_ms, 1),
            "total_ms": round(total_latency_ms, 1),
        },
        "tokens": {
            "input": generation_result["input_tokens"],
            "output": generation_result["output_tokens"],
        },
    }
