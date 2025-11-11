from fastapi import APIRouter, Query
from ..core.db import query

router = APIRouter(prefix="/datasets", tags=["datasets"])

def _chunks_table_exists() -> bool:
    try:
        r = query("SELECT to_regclass('public.chunks');")
        return bool(r and r[0][0])
    except Exception:
        return False

@router.get("/options")
def document_options(search: str = "", collection: str = "", limit: int = 20):
    like = f"%{search.lower()}%"
    if _chunks_table_exists():
        try:
            rows = query("""
                SELECT id,
                       filename,
                       size,
                       created_at,
                       COALESCE((SELECT COUNT(*) FROM chunks c WHERE c.document_id = documents.id),0) AS chunk_count
                FROM documents
                WHERE lower(filename) LIKE %s
                ORDER BY created_at DESC
                LIMIT %s;
            """, (like, limit))
            return {"options": [{
                "doc_id": r[0],
                "name": r[1],
                "collection": "default",
                "created_at": r[3].isoformat(),
                "chunks": r[4],
                "size_mb": round((r[2] or 0)/1024/1024, 3)
            } for r in rows]}
        except Exception:
            pass
    # Fallback without touching chunks
    rows = query("""
        SELECT id, filename, size, created_at
        FROM documents
        WHERE lower(filename) LIKE %s
        ORDER BY created_at DESC
        LIMIT %s;
    """, (like, limit))
    return {"options": [{
        "doc_id": r[0],
        "name": r[1],
        "collection": "default",
        "created_at": r[3].isoformat(),
        "chunks": 0,
        "size_mb": round((r[2] or 0)/1024/1024, 3)
    } for r in rows]}
