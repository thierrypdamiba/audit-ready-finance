"""SQLite audit trail for query provenance tracking."""

import json
import sqlite3
import time
from datetime import datetime, timezone

from config import AUDIT_DB_PATH


def get_db() -> sqlite3.Connection:
    """Get SQLite connection with row factory."""
    conn = sqlite3.connect(AUDIT_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create audit trail table if it doesn't exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT DEFAULT 'demo_user',
            query TEXT NOT NULL,
            retrieval_plan TEXT,
            graph_entities TEXT,
            vector_results TEXT,
            answer TEXT,
            citations TEXT,
            provenance_path TEXT,
            model_used TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            retrieval_latency_ms REAL DEFAULT 0,
            generation_latency_ms REAL DEFAULT 0,
            total_latency_ms REAL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def log_query(
    query: str,
    retrieval_plan: dict,
    graph_entities: list,
    vector_results: list,
    answer: str,
    citations: list,
    provenance_path: dict,
    model_used: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    retrieval_latency_ms: float = 0,
    generation_latency_ms: float = 0,
    total_latency_ms: float = 0,
    user_id: str = "demo_user",
) -> int:
    """Log a query and its full provenance to the audit trail. Returns the log ID."""
    conn = get_db()
    cursor = conn.execute(
        """
        INSERT INTO audit_log (
            timestamp, user_id, query, retrieval_plan, graph_entities,
            vector_results, answer, citations, provenance_path,
            model_used, input_tokens, output_tokens,
            retrieval_latency_ms, generation_latency_ms, total_latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            user_id,
            query,
            json.dumps(retrieval_plan),
            json.dumps(graph_entities),
            json.dumps(vector_results),
            answer,
            json.dumps(citations),
            json.dumps(provenance_path),
            model_used,
            input_tokens,
            output_tokens,
            retrieval_latency_ms,
            generation_latency_ms,
            total_latency_ms,
        ),
    )
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    return log_id


def get_recent_logs(limit: int = 50) -> list[dict]:
    """Get recent audit log entries."""
    conn = get_db()
    rows = conn.execute(
        """
        SELECT id, timestamp, user_id, query, model_used,
               input_tokens, output_tokens, total_latency_ms,
               citations
        FROM audit_log
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        citations = json.loads(row["citations"]) if row["citations"] else []
        results.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "user_id": row["user_id"],
            "query": row["query"],
            "model_used": row["model_used"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "total_latency_ms": row["total_latency_ms"],
            "citations_count": len(citations),
        })

    return results


def get_log_by_id(log_id: int) -> dict | None:
    """Get full audit log entry by ID."""
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM audit_log WHERE id = ?", (log_id,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "user_id": row["user_id"],
        "query": row["query"],
        "retrieval_plan": json.loads(row["retrieval_plan"]) if row["retrieval_plan"] else {},
        "graph_entities": json.loads(row["graph_entities"]) if row["graph_entities"] else [],
        "vector_results": json.loads(row["vector_results"]) if row["vector_results"] else [],
        "answer": row["answer"],
        "citations": json.loads(row["citations"]) if row["citations"] else [],
        "provenance_path": json.loads(row["provenance_path"]) if row["provenance_path"] else {},
        "model_used": row["model_used"],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "retrieval_latency_ms": row["retrieval_latency_ms"],
        "generation_latency_ms": row["generation_latency_ms"],
        "total_latency_ms": row["total_latency_ms"],
    }


# Initialize DB on import
init_db()
