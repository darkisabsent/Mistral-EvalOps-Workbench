from fastapi import APIRouter, HTTPException, Request, Response
from uuid import uuid4
import json, time, statistics
from ..evals.judge import judge_one
from ..core.db import query
from ..core.settings import settings
from ..core.mistral_client import get_client
from ..rag.embeddings import embed_texts
from ..rag.retriever import topk_by_cosine
from ..rag.rerank import rerank
import threading  # NEW
from datetime import datetime  # NEW
import uuid  # NEW

router = APIRouter(prefix="/eval", tags=["eval"])

def _load_prompt(name: str, version: int):
    rows = query("SELECT system, user_template, params FROM prompts WHERE name=%s AND version=%s;", (name, version))
    if not rows:
        raise HTTPException(404, detail=f"Prompt not found: {name}@{version}")
    return rows[0][0], rows[0][1], rows[0][2]

def _apply_template(system: str, user_template: str, context: str, question: str):
    return system, user_template.replace("{{context}}", context).replace("{{question}}", question)

def _normalize_dataset_id(dataset_id: str) -> str:  # NEW
    """
    Accept:
      - full UUID
      - unique UUID prefix (>= 6 chars) present in datasets.id
    Raise 400 if not resolvable.
    """
    ds = (dataset_id or "").strip()
    if not ds:
        raise HTTPException(400, detail="dataset_id required")
    # Try full UUID
    try:
        uuid.UUID(ds)
        return ds
    except Exception:
        pass
    # Try prefix match (minimum length guard)
    if len(ds) < 6:
        raise HTTPException(400, detail="dataset_id prefix too short")
    rows = query("SELECT id FROM datasets WHERE id::text LIKE %s LIMIT 2;", (ds + "%",))
    if not rows:
        raise HTTPException(400, detail="dataset_id not found")
    if len(rows) > 1:
        raise HTTPException(400, detail="dataset_id prefix not unique")
    return rows[0][0]

def _fetch_ds_items(dataset_id: str):
    ds_id = _normalize_dataset_id(dataset_id)  # CHANGED
    return query(
        "SELECT id, question, reference_answer, doc_ids, resolved_doc_ids "
        "FROM dataset_items WHERE dataset_id=%s ORDER BY ordinal;",
        (ds_id,)
    )

def _token_count(text: str) -> int:
    # Simple heuristic: whitespace split (~4 chars/token average ignored deliberately). Documented.
    return len([t for t in text.split() if t])

def _budget_context_tokens(snippets: list[dict], max_tokens: int = 2000):
    """
    Heuristic token budgeting (not char length). Adjust max_tokens as needed.
    Each snippet counted as len(whitespace_split).
    """
    used, out = 0, []
    for c in snippets:
        tc = _token_count(c["text"])
        if used + tc > max_tokens:
            break
        out.append(c)
        used += tc
    return out

def _context_from_doc_ids(doc_ids: list[str], k: int):
    if not doc_ids:
        return []
    placeholders = ",".join(["%s"] * len(doc_ids))
    rows = query(f"""
        SELECT id, document_id, text FROM chunks
        WHERE document_id IN ({placeholders})
        ORDER BY document_id, ordinal
        LIMIT %s;
    """, (*doc_ids, k))
    # score unknown here; set to 1.0 for ordering purposes
    return [{"id": r[0], "document_id": r[1], "text": r[2], "score": 1.0} for r in rows]

def _retrieve(question: str, top_k: int, rerank_cfg, doc_ids: list[str]):
    # If doc_ids provided, prefer deterministic selection
    if doc_ids:
        base = _context_from_doc_ids(doc_ids, top_k)
    else:
        [qvec] = embed_texts([question])
        base = topk_by_cosine(qvec, k=top_k)
    # optional rerank
    strategy = "none"
    k_after = top_k
    if isinstance(rerank_cfg, dict):
        strategy = rerank_cfg.get("strategy", "none")
        k_after = int(rerank_cfg.get("k", top_k))
    elif isinstance(rerank_cfg, (str, bool)):
        strategy = "llm" if rerank_cfg is True else (rerank_cfg if isinstance(rerank_cfg, str) else "none")
    if strategy != "none" and not doc_ids:
        scored = rerank(
            question,
            [{"id": c["id"], "text": c["text"], "sim": c.get("score", 0)} for c in base],
            {"strategy": strategy, "k": k_after}
        )
        order = [r["chunk_id"] for r in scored]
        by_id = {c["id"]: c for c in base}
        base = [by_id[i] for i in order if i in by_id]
    return base[:k_after]

def _answer_variant(client, question: str, variant: dict, doc_ids: list[str]):
    pr = variant.get("prompt", {})
    pn, pv = pr.get("name"), pr.get("version")
    if not pn or pv is None:
        raise HTTPException(400, detail="variant.prompt {name,version} required")
    retr = variant.get("retriever", {}) or {}
    top_k = int(retr.get("top_k", 8))
    rerank_cfg = retr.get("rerank", "none")

    ctx_rows = _retrieve(question, top_k, rerank_cfg, doc_ids)
    ctx_rows = _budget_context_tokens(ctx_rows, 2000)
    context_text = "\n\n".join([c["text"] for c in ctx_rows])
    low_conf = (len(ctx_rows) == 0) or (ctx_rows and float(ctx_rows[0].get("score", 0)) < 0.30)  # heuristic; score <0.30 treated low confidence

    sys, tmpl, params = _load_prompt(pn, int(pv))
    sys_final, user_final = _apply_template(sys, tmpl, context_text, question)
    messages = [{"role":"system","content":sys_final},{"role":"user","content":user_final}]

    start = time.time()
    resp = client.chat.complete(
        model=settings.model_chat,
        messages=messages,
        **{k: v for k, v in (params or {}).items() if k in ("temperature","top_p","max_tokens")}
    )
    latency = int((time.time() - start) * 1000)
    answer = getattr(resp, "output_text", None) or ""
    usage = getattr(resp, "usage", None) or {}
    in_tok = getattr(usage, "input_tokens", None) or usage.get("input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", None) or usage.get("output_tokens", 0) or 0

    citations = [{
        "chunk_id": c["id"], "doc_id": c["document_id"], "rank": i+1,
        "score": float(c.get("score", 0)), 
        "snippet": (c["text"][:80] + ("..." if len(c["text"]) > 160 else "") + c["text"][-80:] if len(c["text"]) > 160 else c["text"])
    } for i, c in enumerate(ctx_rows)]
    return {
        "answer": answer,
        "citations": citations,
        "stats": {"latency_ms": latency, "input_tokens": in_tok, "output_tokens": out_tok, "low_confidence": low_conf},
        "retrieved_chunk_ids": [c["chunk_id"] if "chunk_id" in c else c["id"] for c in citations],
    }

# NEW: Judge helper function (completed)
def judge_answer(question: str, answer: str, context: str, judge_cfg: dict) -> dict:
    """
    LLM-as-Judge: score relevance and groundedness.
    Returns: {"relevance":1-5,"groundedness":1-5,"rationale":"..."}
    """
    # delegate to shared judge_one (supports repeats, structured JSON)
    return judge_one(question, answer, context, judge_cfg)

# NEW: /eval/judge endpoint
@router.post("/judge")
def judge_run(body: dict):
    """
    Batch judge a run's answers.
    Input: {"run_id":"uuid","judge":{"model":"mistral-small-latest","repeats":1}}
    Output: {"run_id":"uuid","judged":N,"relevance_avg":3.8,"groundedness_avg":4.6}
    """
    run_id = body.get("run_id")
    if not run_id:
        raise HTTPException(400, detail="run_id required")
    judge_cfg = body.get("judge", {}) or {}
    judge_cfg.setdefault("model", settings.judge_model or settings.model_chat)
    judge_cfg.setdefault("repeats", 1)
    judge_cfg.setdefault("temperature", 0.1)

    # Load run
    rows = query("SELECT id, kind, config FROM runs WHERE id=%s;", (run_id,))
    if not rows:
        raise HTTPException(404, detail="run not found")
    run_kind = rows[0][1]
    run_cfg = rows[0][2] or {}

    # Fetch existing judgements
    judg_rows = query(
        "SELECT id, query, answer, refs FROM judgements WHERE run_id=%s ORDER BY created_at;",
        (run_id,)
    )

    new_mode = False
    work_items = []

    if judg_rows:
        # Rejudge existing judgements
        for (_jid, q, a, refs) in judg_rows:
            citations = (refs or {}).get("citations") or []
            ctx = "\n".join([c.get("snippet","") for c in citations])
            work_items.append({"query": q, "answer": a, "context": ctx})
    else:
        # No judgements yet: attempt to build from run config (chat run)
        if run_kind == "chat" and run_cfg.get("answer"):
            citations = run_cfg.get("citations") or []
            ctx = "\n".join([c.get("snippet","") for c in citations])
            work_items.append({"query": run_cfg.get("query","(unknown query)") , "answer": run_cfg["answer"], "context": ctx})
            new_mode = True
        else:
            raise HTTPException(400, detail="No judgements present and run kind unsupported for initial judging")

    relevance_scores, grounded_scores = [], []
    ordinal_base = len(judg_rows)

    for idx, item in enumerate(work_items):
        j = judge_answer(item["query"], item["answer"], item["context"], judge_cfg)
        relevance_scores.append(j["relevance"])
        grounded_scores.append(j["groundedness"])

        if new_mode:
            # Insert new judgement row
            query(
                """INSERT INTO judgements(run_id, ordinal, query, answer, refs, relevance, groundedness, rationale)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s);""",
                (
                    run_id,
                    ordinal_base + idx,
                    item["query"],
                    item["answer"],
                    json.dumps({
                        "citations": run_cfg.get("citations") or [],
                        "judge": {
                            "model": j["model"],
                            "repeats": j.get("repeats", 1),
                            "latency_ms": j["latency_ms"]
                        }
                    }),
                    j["relevance"],
                    j["groundedness"],
                    j["rationale"]
                )
            )
        else:
            # Update existing judgement (matched by query+answer)
            query(
                """UPDATE judgements
                   SET relevance=%s, groundedness=%s, rationale=%s
                   WHERE run_id=%s AND query=%s AND answer=%s;""",
                (
                    j["relevance"],
                    j["groundedness"],
                    j["rationale"],
                    run_id,
                    item["query"],
                    item["answer"]
                )
            )

    def _avg(xs): return round(sum(xs) / len(xs), 2) if xs else 0.0

    return {
        "run_id": run_id,
        "judged": len(work_items),
        "relevance_avg": _avg(relevance_scores),
        "groundedness_avg": _avg(grounded_scores)
    }

# NEW: A/B eval runner
@router.post("/run")
def start_eval_run(body: dict):
    """
    Start A/B eval over a dataset.
    Input:
      {
        "dataset_id":"uuid",
        "variant_a":{"prompt":{"name":"rag","version":2},"retriever":{"top_k":8}},
        "variant_b":{"prompt":{"name":"rag","version":3},"retriever":{"top_k":8}}
      }
    Output: { "group_id":"uuid", "run_a_id":"uuid", "run_b_id":"uuid" }
    """
    dataset_id = body.get("dataset_id") or body.get("dataset")
    if not dataset_id:
        raise HTTPException(400, detail="dataset_id required")
    variant_a = body.get("variant_a")
    variant_b = body.get("variant_b")
    if not isinstance(variant_a, dict) or not isinstance(variant_b, dict):
        raise HTTPException(400, detail="variant_a and variant_b required")

    items = _fetch_ds_items(dataset_id)
    if not items:
        raise HTTPException(404, detail="dataset empty")

    client = get_client()
    group_id = str(uuid4())

    # Insert two runs (A and B)
    meta_a = {"group_id": group_id, "variant": "A"}
    meta_b = {"group_id": group_id, "variant": "B"}
    cfg_a = {"dataset_id": dataset_id, "variant": "A", "prompt": variant_a.get("prompt"), "retriever": variant_a.get("retriever")}
    cfg_b = {"dataset_id": dataset_id, "variant": "B", "prompt": variant_b.get("prompt"), "retriever": variant_b.get("retriever")}
    run_a = query(
        "INSERT INTO runs(kind,config,model,backend,meta) VALUES (%s,%s,%s,%s,%s) RETURNING id;",
        ("ab_eval", json.dumps(cfg_a), settings.model_chat, settings.backend, json.dumps(meta_a))
    )[0][0]
    run_b = query(
        "INSERT INTO runs(kind,config,model,backend,meta) VALUES (%s,%s,%s,%s,%s) RETURNING id;",
        ("ab_eval", json.dumps(cfg_b), settings.model_chat, settings.backend, json.dumps(meta_b))
    )[0][0]

    total_a_in = total_a_out = total_b_in = total_b_out = 0
    count = 0
    # naive wins tracker (requires judge to be meaningful; default ties)
    wins = {"a": 0, "b": 0, "ties": 0}

    for (item_id, question, reference_answer, doc_ids_json, resolved_array) in items:
        count += 1
        # pick resolved ids if present else parse json list
        doc_ids = resolved_array if resolved_array else (json.loads(doc_ids_json) if doc_ids_json else [])
        # Variant A
        va = _answer_variant(client, question, variant_a, doc_ids)
        total_a_in += va["stats"]["input_tokens"]; total_a_out += va["stats"]["output_tokens"]
        query(
            "INSERT INTO judgements(run_id, ordinal, query, answer, refs, relevance, groundedness, rationale) VALUES (%s,%s,%s,%s,%s,%s,%s,%s);",
            (run_a, count-1, question, va["answer"], json.dumps({
                "variant": "A",
                "citations": va["citations"],
                "stats": va["stats"],
                "prompt": variant_a.get("prompt", {}),
                "retriever": variant_a.get("retriever", {}),
            }), None, None, None)
        )
        # Variant B
        vb = _answer_variant(client, question, variant_b, doc_ids)
        total_b_in += vb["stats"]["input_tokens"]; total_b_out += vb["stats"]["output_tokens"]
        query(
            "INSERT INTO judgements(run_id, ordinal, query, answer, refs, relevance, groundedness, rationale) VALUES (%s,%s,%s,%s,%s,%s,%s,%s);",
            (run_b, count-1, question, vb["answer"], json.dumps({
                "variant": "B",
                "citations": vb["citations"],
                "stats": vb["stats"],
                "prompt": variant_b.get("prompt", {}),
                "retriever": variant_b.get("retriever", {}),
            }), None, None, None)
        )
        # Without judge yet, mark ties (UI can re-judge later)
        wins["ties"] += 1

    # Build summary in config for both runs (placeholders for rel/grd until judged)
    def _summary(in_tok, out_tok):
        return {
            "A": {"relevance": 0.0, "groundedness": 0.0, "in_tokens_avg": (total_a_in / max(1, count)), "out_tokens_avg": (total_a_out / max(1, count))},
            "B": {"relevance": 0.0, "groundedness": 0.0, "in_tokens_avg": (total_b_in / max(1, count)), "out_tokens_avg": (total_b_out / max(1, count))},
            "delta": {"relevance": 0.0, "groundedness": 0.0},
            "wins": wins,
        }

    cost_a = round(
        (total_a_in / 1_000_000.0) * settings.input_tokens_cost_per_million +
        (total_a_out / 1_000_000.0) * settings.output_tokens_cost_per_million, 4
    )
    cost_b = round(
        (total_b_in / 1_000_000.0) * settings.input_tokens_cost_per_million +
        (total_b_out / 1_000_000.0) * settings.output_tokens_cost_per_million, 4
    )

    # Update runs with counts, tokens, costs, summary
    row_a = query("SELECT config FROM runs WHERE id=%s;", (run_a,))
    row_b = query("SELECT config FROM runs WHERE id=%s;", (run_b,))
    cfg_a_final = row_a[0][0] if row_a else cfg_a
    cfg_b_final = row_b[0][0] if row_b else cfg_b
    cfg_a_final.update({"summary": _summary(total_a_in, total_a_out)})
    cfg_b_final.update({"summary": _summary(total_b_in, total_b_out)})

    query(
        "UPDATE runs SET finished_at=now(), count=%s, input_tokens=%s, output_tokens=%s, cost_cents=%s, config=%s WHERE id=%s;",
        (count, total_a_in, total_a_out, cost_a, json.dumps(cfg_a_final), run_a)
    )
    query(
        "UPDATE runs SET finished_at=now(), count=%s, input_tokens=%s, output_tokens=%s, cost_cents=%s, config=%s WHERE id=%s;",
        (count, total_b_in, total_b_out, cost_b, json.dumps(cfg_b_final), run_b)
    )

    return {"group_id": group_id, "run_a_id": str(run_a), "run_b_id": str(run_b)}

# NEW: Batch API — enqueue a batch judging job for an A/B group
@router.post("/batch")
def start_batch_eval(body: dict):
    """
    Input: { "ab_group_id": "uuid", "judge_model": "..." }
    Output: { "batch_id":"…", "status":"queued", "requests_count": N, "eta_seconds": int }
    """
    group_id = body.get("ab_group_id")
    if not group_id:
        raise HTTPException(400, detail="ab_group_id required")

    judge_model = body.get("judge_model", settings.judge_model or settings.model_chat)

    # Find runs in this A/B group
    run_rows = query("SELECT id, meta, config FROM runs WHERE meta->>'group_id' = %s ORDER BY started_at;", (group_id,))
    if not run_rows:
        raise HTTPException(404, detail=f"No runs for group {group_id}")

    # Build work items: each judgement-less QA pair to score
    items = []
    for run_id, meta, cfg in run_rows:
        jrows = query("SELECT id, query, answer, refs FROM judgements WHERE run_id=%s;", (run_id,))
        for jid, q, a, refs in jrows:
            refs = refs or {}
            ctx = "\n".join([c.get("snippet", "") for c in (refs.get("citations") or [])])
            items.append({"jid": jid, "run_id": run_id, "query": q, "answer": a, "context": ctx})

    requests_count = len(items)
    if requests_count == 0:
        raise HTTPException(400, detail="No judgements to score for this group")

    batch_id = str(uuid4())
    query(
        "INSERT INTO batch_jobs(id, group_id, model, status, requests_count, completed_count) VALUES (%s,%s,%s,%s,%s,%s);",
        (batch_id, group_id, judge_model, "queued", requests_count, 0)
    )

    def _process():
        # mark processing
        query("UPDATE batch_jobs SET status='processing' WHERE id=%s;", (batch_id,))
        completed = 0
        # Process sequentially; provider Batch API can replace this loop
        for it in items:
            try:
                j = judge_one(it["query"], it["answer"], it["context"], {"model": judge_model, "temperature": 0.1})
                query(
                    "UPDATE judgements SET relevance=%s, groundedness=%s, rationale=%s WHERE id=%s;",
                    (int(j["relevance"]), int(j["groundedness"]), j.get("rationale", ""), it["jid"])
                )
                # update per-run summary (A or B)
                rrow = query("SELECT meta, config FROM runs WHERE id=%s;", (it["run_id"],))
                if rrow:
                    meta, cfg = rrow[0]
                    variant = (meta or {}).get("variant", "")
                    avg = query("SELECT AVG(relevance)::float, AVG(groundedness)::float FROM judgements WHERE run_id=%s AND relevance IS NOT NULL;", (it["run_id"],))
                    rel_avg = float(avg[0][0] or 0.0)
                    grd_avg = float(avg[0][1] or 0.0)
                    cfg = cfg or {}
                    summary = cfg.get("summary") or {}
                    if variant in ("A", "B"):
                        summary.setdefault("A", {"relevance": 0.0, "groundedness": 0.0})
                        summary.setdefault("B", {"relevance": 0.0, "groundedness": 0.0})
                        summary[variant]["relevance"] = rel_avg
                        summary[variant]["groundedness"] = grd_avg
                        cfg["summary"] = summary
                        query("UPDATE runs SET config=%s WHERE id=%s;", (json.dumps(cfg), it["run_id"]))
            except Exception:
                # continue on error
                pass
            completed += 1
            query("UPDATE batch_jobs SET completed_count=%s WHERE id=%s;", (completed, batch_id))
        # done
        query("UPDATE batch_jobs SET status='completed', completed_at=now() WHERE id=%s;", (batch_id,))

    threading.Thread(target=_process, daemon=True).start()

    # rough ETA: 2s per item (tunable)
    return {"batch_id": batch_id, "status": "queued", "requests_count": requests_count, "eta_seconds": requests_count * 2}

# NEW: Batch status polling
@router.get("/batch/{batch_id}")
def get_batch_status(batch_id: str):
    row = query("SELECT status, requests_count, COALESCE(completed_count,0) FROM batch_jobs WHERE id=%s;", (batch_id,))
    if not row:
        raise HTTPException(404, detail="Batch job not found")
    status, total, done = row[0]
    return {"batch_id": batch_id, "status": status, "completed": int(done or 0), "total": int(total or 0)}