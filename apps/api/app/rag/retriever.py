from ..core.db import query
from ..core.logging import log_event, hash_vector
import time, math

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b: return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def _embedding_col_is_vector() -> bool:
    try:
        rows = query("""
            SELECT udt_name
            FROM information_schema.columns
            WHERE table_name='chunks' AND column_name='embedding';
        """)
        return bool(rows and rows[0][0] == 'vector')
    except Exception:
        return False

def _vec_literal(vec: list[float]) -> str:  # NEW
    return "[" + ",".join(f"{float(x):.6f}" for x in (vec or [])) + "]"

def topk_by_cosine(query_vec: list[float], k: int = 8):
    start = time.time()
    degraded = False
    items = []
    if _embedding_col_is_vector():
        # pgvector path (distance first, convert to similarity in Python)
        sql = """
            SELECT id, document_id, text,
                   (embedding <=> %s::vector) AS distance
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        try:
            vlit = _vec_literal(query_vec)
            rows = query(sql, (vlit, vlit, k))
            for r in rows:
                dist = float(r[3] or 0.0)
                sim = 1.0 - dist            # convert distance -> similarity
                if sim < 0: sim = 0.0
                if sim > 1: sim = 1.0
                items.append({"id": str(r[0]), "document_id": str(r[1]), "text": r[2], "score": sim})
        except Exception:
            degraded = True
    if not items and not _embedding_col_is_vector():
        degraded = True
    if degraded:
        # Fallback: array embeddings stored as DOUBLE PRECISION[]
        # Pull more than k to get meaningful ranking (cap at 2000 or 5*k)
        sample_n = min(2000, max(k * 5, k))
        try:
            rows = query("SELECT id, document_id, text, embedding FROM chunks LIMIT %s;", (sample_n,))
            scored = []
            for r in rows:
                emb = r[3]
                sim = _cosine(query_vec, emb) if isinstance(emb, list) else 0.0
                scored.append({"id": str(r[0]), "document_id": str(r[1]), "text": r[2], "score": float(sim)})
            scored.sort(key=lambda x: x["score"], reverse=True)
            items = scored[:k]
        except Exception:
            items = []
    latency = int((time.time() - start) * 1000)
    log_event(
        "rag.retrieve",
        query_vec_hash=hash_vector(query_vec),
        top_k=k,
        timing_ms=latency,
        degraded=degraded,
        candidates=[{"chunk_id": it["id"], "sim": it["score"], "doc_id": it["document_id"]} for it in items],
    )
    return items
