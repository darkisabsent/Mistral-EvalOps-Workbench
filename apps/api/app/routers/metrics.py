from fastapi import APIRouter
from ..core.db import query
import math
from datetime import datetime, timedelta  # NEW

router = APIRouter(prefix="/metrics", tags=["metrics"])

def _pct(xs, p):
    if not xs: return 0
    xs = sorted(xs)
    i = (p/100.0)*(len(xs)-1)
    lo = math.floor(i); hi = math.ceil(i)
    if lo == hi: return xs[int(i)]
    return xs[lo] + (xs[hi]-xs[lo])*(i-lo)

@router.get("/summary")
def summary(window: str = "24h"):
    # NEW: parse window like "24h" or "7d"
    now = datetime.utcnow()
    try:
        n = int(window[:-1]); unit = window[-1].lower()
        delta = timedelta(hours=n) if unit == 'h' else timedelta(days=n if unit == 'd' else 1)
    except Exception:
        delta = timedelta(hours=24)
    since = now - delta
    rows = query(
        "SELECT latency_ms, input_tokens, output_tokens, cost_cents FROM runs WHERE finished_at IS NOT NULL AND started_at >= %s ORDER BY started_at DESC LIMIT 200;",
        (since,)
    )
    latencies = [r[0] for r in rows if r[0] is not None]
    in_tok = [r[1] or 0 for r in rows]
    out_tok = [r[2] or 0 for r in rows]
    costs = [float(r[3] or 0) for r in rows]
    judg = query("""
        SELECT AVG(relevance)::float, AVG(groundedness)::float
        FROM judgements
        WHERE run_id IN (SELECT id FROM runs WHERE finished_at IS NOT NULL AND started_at >= %s ORDER BY started_at DESC LIMIT 200)
    """, (since,))
    rel_avg = float(judg[0][0]) if judg and judg[0][0] is not None else 0.0
    grd_avg = float(judg[0][1]) if judg and judg[0][1] is not None else 0.0
    return {
        "p50_latency_ms": int(_pct(latencies,50)) if latencies else 0,
        "p95_latency_ms": int(_pct(latencies,95)) if latencies else 0,
        "avg_tokens_in": int(sum(in_tok)/len(in_tok)) if in_tok else 0,
        "avg_tokens_out": int(sum(out_tok)/len(out_tok)) if out_tok else 0,
        "avg_cost_cents": round(sum(costs)/len(costs),4) if costs else 0,
        "relevance_avg": rel_avg,
        "groundedness_avg": grd_avg,
        "runs_count": len(rows)
    }
