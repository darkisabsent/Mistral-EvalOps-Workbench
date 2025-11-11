from typing import List, Dict, Optional
import time
from ..core.mistral_client import get_client
from ..core.settings import settings
from ..core.logging import log_event, tracer, model_event_sampled, hash_text

def rerank(query_text: str, candidates: List[Dict], config: Optional[Dict] = None) -> List[Dict]:
    """
    candidates: [{chunk_id, text, sim}]
    config: {strategy: "none"|"llm"|"cross_encoder", k?: int, timeout_ms?: int}
    """
    cfg = config or {}
    strategy = (cfg.get("strategy") or ("llm" if cfg.get("strategy") is True else "none"))
    k = int(cfg.get("k", len(candidates)))
    timeout_ms = int(cfg.get("timeout_ms", 8000))

    start = time.time()
    degraded = False
    if strategy in (False, None, "none"):
        out = [{"chunk_id": c["id"] if "id" in c else c.get("chunk_id"), "score": c.get("sim", 0), "reason": None} for c in candidates]
        log_event("rag.rerank", strategy="none", input_k=len(candidates), output_k=min(k, len(out)), timing_ms=int((time.time()-start)*1000))
        return out[:k]

    if strategy == "cross_encoder":
        # Placeholder: fall back to ANN order until a scorer is plugged in.
        out = [{"chunk_id": c["id"] if "id" in c else c.get("chunk_id"), "score": c.get("sim", 0), "reason": None} for c in candidates]
        log_event("rag.rerank", strategy="cross_encoder", input_k=len(candidates), output_k=min(k, len(out)), timing_ms=int((time.time()-start)*1000), degraded=True)
        return out[:k]

    if strategy == "llm":
        # Build a compact, schema-enforced scoring request
        schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "score": {"type": "number", "minimum": 0, "maximum": 5},
                            "reason": {"type": "string"},
                        },
                        "required": ["chunk_id", "score"]
                    }
                }
            },
            "required": ["results"]
        }
        def short(txt: str, n: int = 240) -> str:
            txt = txt or ""
            return txt[:n] + ("..." if len(txt) > n else "")

        items = [{"chunk_id": c.get("id") or c.get("chunk_id"), "text": short(c.get("text", ""))} for c in candidates]
        listing = "\n".join(f"- id={it['chunk_id']}: {it['text']}" for it in items)
        system = "You are a helpful reranker. Score each chunk for how helpful it is to answer the query. Higher is better."
        user = f"Query: {query_text}\nChunks:\n{listing}\nReturn JSON with array 'results' entries {{chunk_id, score, reason}}. Do not include chunk text."

        client = get_client()
        sampled = model_event_sampled()
        if sampled:
            log_event("llm.request", provider=settings.backend, model=settings.model_chat, mode="json",
                      prompt_hash=hash_text(user), tokens_estimate=len(user)//4)
        try:
            with tracer.start_as_current_span("rag.rerank.llm") as span:
                span.set_attribute("llm.provider", settings.backend)
                span.set_attribute("llm.model", settings.model_chat)
                resp = client.chat.complete(
                    model=settings.model_chat,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}],
                    temperature=0.1,
                    max_tokens=300,
                    response_format={"type":"json_object","schema":schema},
                )
        except Exception:
            degraded = True
            out = [{"chunk_id": c.get("id") or c.get("chunk_id"), "score": c.get("sim", 0), "reason": None} for c in candidates]
            log_event("rag.rerank", strategy="llm", input_k=len(candidates), output_k=min(k, len(out)),
                      timing_ms=int((time.time()-start)*1000), degraded=True)
            return out[:k]

        timing = int((time.time()-start)*1000)
        parsed = getattr(resp, "output_parsed", {}) or {}
        results = parsed.get("results") or []
        # map scores back; fall back to ANN if missing
        score_map = {str(r["chunk_id"]): float(r["score"]) for r in results if "chunk_id" in r and "score" in r}
        out = []
        for c in candidates:
            cid = str(c.get("id") or c.get("chunk_id"))
            out.append({"chunk_id": cid, "score": score_map.get(cid, c.get("sim", 0)), "reason": None})
        out.sort(key=lambda x: x["score"], reverse=True)
        log_event("rag.rerank", strategy="llm", input_k=len(candidates), output_k=min(k, len(out)), timing_ms=timing, degraded=degraded)
        return out[:k]

    # Unknown strategy -> none
    out = [{"chunk_id": c.get("id") or c.get("chunk_id"), "score": c.get("sim", 0), "reason": None} for c in candidates]
    log_event("rag.rerank", strategy="none", input_k=len(candidates), output_k=min(k, len(out)), timing_ms=int((time.time()-start)*1000))
    return out[:k]
