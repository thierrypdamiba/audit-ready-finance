"""FastAPI application for audit-ready finance demo."""

import asyncio
import os

os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient

from audit_trail import get_log_by_id, get_recent_logs, init_db
from config import COLLECTION_NAME, PORT, QDRANT_API_KEY, QDRANT_URL
from retrieval import graph_search, query_pipeline, vector_search

app = FastAPI(title="Audit-Ready Finance", version="1.0.0")


# -- Request/Response Models --

class QueryRequest(BaseModel):
    question: str
    tenant_id: str | None = None
    top_k: int = 8


class VectorSearchRequest(BaseModel):
    query: str
    company: str | None = None
    section: str | None = None
    tenant_id: str | None = None
    top_k: int = 10


class GraphSearchRequest(BaseModel):
    query: str


# -- Endpoints --

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if not os.path.exists(frontend_path):
        return HTMLResponse("<h1>Frontend not built yet</h1>", status_code=200)
    with open(frontend_path, "r") as f:
        return HTMLResponse(f.read())


@app.post("/api/query")
async def handle_query(req: QueryRequest):
    """Main query endpoint. Returns answer + citations + full provenance."""
    result = await query_pipeline(
        query=req.question,
        tenant_id=req.tenant_id,
        top_k=req.top_k,
    )
    return result


@app.get("/api/audit-log")
async def list_audit_logs(limit: int = 50):
    """List recent audit log entries."""
    return get_recent_logs(limit=limit)


@app.get("/api/audit-log/{log_id}")
async def get_audit_log(log_id: int):
    """Get full audit trail for a specific query."""
    entry = get_log_by_id(log_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Audit log entry not found")
    return entry


@app.get("/api/collections")
async def list_collections():
    """Show Qdrant collection stats."""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collections = client.get_collections().collections

    result = []
    for col in collections:
        try:
            info = client.get_collection(col.name)
            result.append({
                "name": col.name,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
            })
        except Exception:
            result.append({"name": col.name, "error": "could not fetch info"})

    return result


@app.get("/api/graph-explore")
async def explore_graph(query: str = "financial risk"):
    """Explore cognee entity relationships."""
    entities = await graph_search(query)
    return {"query": query, "entities": entities}


@app.post("/api/search/vector")
async def direct_vector_search(req: VectorSearchRequest):
    """Direct Qdrant vector search for comparison."""
    companies = [req.company] if req.company else None
    sections = [req.section] if req.section else None

    results = vector_search(
        query=req.query,
        target_companies=companies,
        target_sections=sections,
        tenant_id=req.tenant_id,
        top_k=req.top_k,
    )

    return {
        "query": req.query,
        "filters": {
            "company": req.company,
            "section": req.section,
            "tenant_id": req.tenant_id,
        },
        "results": results,
    }


@app.post("/api/search/graph")
async def direct_graph_search(req: GraphSearchRequest):
    """Direct cognee graph search for comparison."""
    entities = await graph_search(req.query)
    return {"query": req.query, "entities": entities}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    checks = {"status": "ok", "qdrant": False, "audit_db": False}

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.get_collections()
        checks["qdrant"] = True
    except Exception as ex:
        checks["qdrant_error"] = str(ex)

    try:
        init_db()
        checks["audit_db"] = True
    except Exception as ex:
        checks["audit_db_error"] = str(ex)

    if not all([checks["qdrant"], checks["audit_db"]]):
        checks["status"] = "degraded"

    return checks


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
