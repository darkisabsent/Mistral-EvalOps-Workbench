from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from ..core.db import query
import statistics, json, datetime

router = APIRouter(prefix="/runs", tags=["runs"])

@router.get("")
def list_runs(
    kind: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    prompt_name: Optional[str] = Query(None),
    prompt_version: Optional[int] = Query(None),
    start: Optional[str] = Query(None),
    end: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    conditions = []
    params = []
    if kind:
        conditions.append("r.kind=%s"); params.append(kind)
    if model:
        conditions.append("r.model=%s"); params.append(model)
    if prompt_name:
        conditions.append("(r.config->>'prompt_name')=%s"); params.append(prompt_name)
    if prompt_version is not None:
        conditions.append("(r.config->>'prompt_version')=%s"); params.append(str(prompt_version))
    if start:
        conditions.append("r.started_at >= %s"); params.append(start)
    if end:
        conditions.append("r.started_at <= %s"); params.append(end)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"""
        SELECT r.id,r.kind,r.started_at,r.finished_at,r.latency_ms,r.model,r.backend,
               r.input_tokens,r.output_tokens,r.cost_cents,r.config,
               AVG(j.relevance)::float AS avg_relevance,
               AVG(j.groundedness)::float AS avg_groundedness
        FROM runs r
        LEFT JOIN judgements j ON j.run_id=r.id
        {where}
        GROUP BY r.id
        ORDER BY r.started_at DESC
        LIMIT %s OFFSET %s;
    """
    rows = query(sql, (*params, limit, offset))
    runs = []
    for r in rows:
        cfg = r[10]
        runs.append({
            "id": r[0], "kind": r[1],
            "started_at": r[2].isoformat(),
            "finished_at": r[3].isoformat() if r[3] else None,
            "latency_ms": r[4],
            "model": r[5], "backend": r[6],
            "input_tokens": r[7], "output_tokens": r[8], "cost_cents": float(r[9] or 0),
            "config": cfg,
            "avg_relevance": r[11], "avg_groundedness": r[12],
            "retrieval_latency_ms": cfg.get("retrieval_latency_ms"),
            "generation_latency_ms": cfg.get("generation_latency_ms"),
            "total_latency_ms": cfg.get("total_latency_ms"),
            "context": cfg.get("context"),
            "prompt_name": cfg.get("prompt_name"), "prompt_version": cfg.get("prompt_version"),
            "dataset_id": cfg.get("dataset_id")
        })
    return {"runs": runs}

@router.get("/{run_id}")
def get_run(run_id: str):
    rows = query("SELECT id, kind, started_at, finished_at, latency_ms, model, backend, config, input_tokens, output_tokens, cost_cents, meta FROM runs WHERE id=%s;", (run_id,))
    if not rows:
        raise HTTPException(404, "run not found")
    r = rows[0]
    judg = query("SELECT relevance, groundedness, rationale, refs, query, answer FROM judgements WHERE run_id=%s;", (run_id,))
    rels = [j[0] for j in judg]
    grs = [j[1] for j in judg]
    def _avg(xs): return (sum(xs)/len(xs)) if xs else 0
    def _st(xs): return statistics.pstdev(xs) if len(xs) > 1 else 0
    cfg = r[7]
    detail = {
        "id": r[0],
        "kind": r[1],
        "started_at": r[2].isoformat(),
        "finished_at": r[3].isoformat() if r[3] else None,
        "latency_ms": r[4],
        "model": r[5],
        "backend": r[6],
        "input_tokens": r[8],
        "output_tokens": r[9],
        "cost_cents": float(r[10] or 0),
        "config": cfg,
        "meta": r[11] or {},  # NEW
        "count": len(judg),
        "avg_relevance": _avg(rels),
        "avg_groundedness": _avg(grs),
        "stdev_relevance": _st(rels),
        "stdev_groundedness": _st(grs),
        "answer": cfg.get("answer"),
        "citations": cfg.get("citations"),
        "retrieval_latency_ms": cfg.get("retrieval_latency_ms"),     # NEW
        "generation_latency_ms": cfg.get("generation_latency_ms"),   # NEW
        "total_latency_ms": cfg.get("total_latency_ms"),             # NEW
        "context": cfg.get("context"),                               # NEW
        "prompt_name": cfg.get("prompt_name"),                       # ensure exposed
        "prompt_version": cfg.get("prompt_version"),
        "retriever_top_k": (cfg.get("retriever") or {}).get("top_k") # NEW
    }
    items = [{
        "relevance": j[0],
        "groundedness": j[1],
        "rationale": j[2],
        "refs": j[3],
        "query": j[4],
        "answer": j[5]
    } for j in judg]
    return {"run": detail, "judgements": items}

@router.get("/group/{group_id}")  # NEW
def get_group_runs(group_id: str):
    """
    Return both runs for an A/B group with their judgements for side-by-side compare.
    """
    rows = query("SELECT id, kind, config, meta, latency_ms, input_tokens, output_tokens, cost_cents FROM runs WHERE meta->>'group_id'=%s ORDER BY started_at;", (group_id,))
    if not rows:
        raise HTTPException(404, detail="group not found")
    out = []
    for r in rows:
        rid, kind, cfg, meta, lat, tin, tout, cost = r
        judg = query("SELECT relevance, groundedness, rationale, refs, query, answer FROM judgements WHERE run_id=%s ORDER BY ordinal;", (rid,))
        out.append({
            "id": rid,
            "kind": kind,
            "meta": meta or {},
            "config": cfg or {},
            "latency_ms": lat,
            "input_tokens": tin,
            "output_tokens": tout,
            "cost_cents": float(cost or 0),
            "judgements": [{
                "relevance": j[0],
                "groundedness": j[1],
                "rationale": j[2],
                "refs": j[3],
                "query": j[4],
                "answer": j[5]
            } for j in judg]
        })
    return {"group_id": group_id, "runs": out}
