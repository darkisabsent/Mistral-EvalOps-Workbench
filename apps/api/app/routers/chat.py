import json, time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ..rag.embeddings import embed_texts_with_retry, embed_texts  # ensure retry available
from ..rag.retriever import topk_by_cosine
from ..core.mistral_client import get_client
from ..core.settings import settings
from ..core.logging import log_event, tracer, hash_text, token_estimate_from_texts, model_event_sampled
from ..core.db import query
import logging
from contextlib import suppress  # NEW
import re  # NEW
logger = logging.getLogger("api")

router = APIRouter(prefix="/chat", tags=["chat"])

def _load_prompt(name: str, version: int):
    rows = query("SELECT system, user_template, params FROM prompts WHERE name=%s AND version=%s;", (name, version))
    if not rows:
        raise HTTPException(404, detail="Prompt version not found")
    return rows[0][0], rows[0][1], rows[0][2]

def _apply_template(system: str, user_template: str, context: str, question: str):
    return system, user_template.replace("{{context}}", context).replace("{{question}}", question)

def _doc_count():
    rows = query("SELECT COUNT(*) FROM documents WHERE status='ok';")
    return rows[0][0] if rows else 0

# Fallback loader (new)
def _load_prompt_or_default(name: str = "rag", version: int = 1):
    try:
        return _load_prompt(name, version)
    except Exception:
        system = "You are a helpful assistant. Answer strictly using the provided context. If insufficient, say you don't know."
        user_tmpl = "Context:\n{{context}}\n\nQuestion:\n{{question}}\n\nAnswer factually and concisely."
        return system, user_tmpl, {}

def _extract_usage(usage_obj, messages, answer: str):
    """
    Safely extract input/output token counts from either a pydantic UsageInfo or dict.
    Falls back to heuristic estimates if missing or zero.
    """
    in_tok = None
    out_tok = None
    try:
        if usage_obj:
            if hasattr(usage_obj, "input_tokens"):
                in_tok = usage_obj.input_tokens
            elif isinstance(usage_obj, dict):
                in_tok = usage_obj.get("input_tokens")
            if hasattr(usage_obj, "output_tokens"):
                out_tok = usage_obj.output_tokens
            elif isinstance(usage_obj, dict):
                out_tok = usage_obj.get("output_tokens")
    except Exception:
        pass
    # Fallbacks
    if not in_tok:
        in_tok = token_estimate_from_texts([m["content"] for m in messages])
    if not out_tok:
        out_tok = max(1, len(answer) // 4)
    return int(in_tok), int(out_tok)

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b: return 0.0
    n = min(len(a), len(b))
    dot = sum((a[i] or 0.0) * (b[i] or 0.0) for i in range(n))
    na = sum((a[i] or 0.0) ** 2 for i in range(n)) ** 0.5
    nb = sum((b[i] or 0.0) ** 2 for i in range(n)) ** 0.5
    if na == 0.0 or nb == 0.0: return 0.0
    return dot / (na * nb)

def _parse_vec(v):
    if v is None: return []
    if isinstance(v, list): return [float(x) for x in v]
    if isinstance(v, tuple): return [float(x) for x in v]
    if isinstance(v, str) and v.startswith('[') and v.endswith(']'):
        with suppress(Exception):
            return [float(x) for x in v.strip('[]').split(',') if x.strip()]
    return []

def _synth_answer(question: str, ctx: list[dict]) -> str:
    # Minimal fallback summary if model returns nothing
    base = "Hybrid search combines keyword BM25 scoring with vector semantic similarity to balance exact term matches and conceptual meaning, improving both recall and precision."
    if not ctx:
        return base
    bits = []
    for i, c in enumerate(ctx[:3], 1):
        bits.append(f"[{i}] {c['snippet'][:140].rstrip()}")
    return base + " Sources: " + " ".join(bits)

# Explicit answer extractor (keeps if output_text empty)
def _extract_full_answer(resp):
    # NEW: fallback extraction if output_text too short (joins all choice contents)
    ans = getattr(resp, "output_text", "") or ""
    if ans.strip() and len(ans.split()) > 3:
        return ans
    out = []
    with suppress(Exception):
        for ch in getattr(resp, "choices", []):
            part = ""
            with suppress(Exception):
                part = getattr(getattr(ch, "message", None), "content", "") or ""
            if not part:
                with suppress(Exception):
                    part = getattr(ch, "delta", "") or ""
            if part:
                out.append(part)
    joined = "".join(out).strip()
    return joined or ans

def _build_numbered_context(chunks: list[dict], question: str) -> str:  # NEW
    lines = []
    for i, c in enumerate(chunks, 1):
        snippet = c.get("text", "")[:300].replace("\n", " ")
        fname = c.get("filename") or ""
        prefix = f'file="{fname}" ' if fname else ""
        lines.append(f"[{i}] {prefix}snippet=\"{snippet}\"")
    return "\n".join(lines)

def _citation_instructions(k: int) -> str:  # NEW
    return (
        "Use only the context items [1]...[{k}]. "
        "Every factual claim must end with one or more bracketed citations, e.g., [2] or [1][3]. "
        "If a claim isn’t supported by the context, write: Not in context. "
        "Answer in one paragraph."
    ).replace("{k}", str(k))

def _validate_citations(answer: str, k: int) -> bool:  # NEW
    import re
    if not re.search(r"\[\d+\]", answer):
        return False
    # invalid marker > k
    nums = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    return all(1 <= n <= k for n in nums)

def _sanitize_citations(answer: str, k: int) -> str:  # NEW
    if not answer or k <= 0:
        return answer or ""
    def _keep(m: re.Match) -> str:
        try:
            n = int(m.group(1))
            return f"[{n}]" if 1 <= n <= k else ""
        except Exception:
            return ""
    return re.sub(r"\[(\d+)\]", _keep, answer)

# Milestone 3: Non-streaming basic chat
@router.post("/complete")
def chat_complete(req: dict):
    """
    Non-streaming chat completion.
    Input: { "query": "string", "retriever": {"top_k": 8}, "prompt": {"name": "rag", "version": 1} }
    Output: { "run_id": "uuid", "answer": "...", "context": [{document_id,snippet,score}], "tokens": {"in": n, "out": m}, "latency_ms": t }
    """
    if _doc_count() == 0:
        raise HTTPException(400, detail="no_documents_ingested")

    question = req.get("query")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(400, detail="query required")

    retr = req.get("retriever") or {}
    try:
        top_k = int(retr.get("top_k", settings.retriever_top_k))
    except Exception:
        top_k = settings.retriever_top_k
    # NEW: optional min_score threshold
    try:
        min_score = float(retr.get("min_score")) if retr.get("min_score") is not None else None
    except Exception:
        min_score = None

    pr = req.get("prompt") or {}
    p_name = pr.get("name") or "rag"
    try:
        p_ver = int(pr.get("version", 1))
    except Exception:
        p_ver = 1

    client = get_client()

    # Create run record early (minimal config)
    run_cfg = {"query": question, "retriever": {"top_k": top_k}, "prompt_name": p_name, "prompt_version": p_ver}
    run_rows = query("INSERT INTO runs(kind,config,model,backend) VALUES (%s,%s,%s,%s) RETURNING id;",
                     ("chat", json.dumps(run_cfg), settings.model_chat, settings.backend))
    run_id = run_rows[0][0]

    # Embed + retrieve (time separately)
    degraded_context = False
    retrieval_start = time.time()  # NEW
    try:
        [qvec] = embed_texts_with_retry([question])
        ctx_rows = topk_by_cosine(qvec, k=top_k)
    except Exception as ex:
        logger.warning(f"embedding degraded (chat_complete): {ex}")
        degraded_context = True
        ctx_rows = []
    retrieval_latency_ms = int((time.time() - retrieval_start) * 1000)  # NEW

    # RECOMPUTE cosine similarity using stored embeddings to avoid 0.0 scores
    if ctx_rows:
        chunk_ids = [r["id"] for r in ctx_rows if r.get("id")]
        if chunk_ids:
            placeholders = ",".join(["%s"] * len(chunk_ids))
            with suppress(Exception):
                emb_rows = query(f"SELECT id, embedding FROM chunks WHERE id IN ({placeholders});", tuple(chunk_ids))
                emb_map = {str(r[0]): _parse_vec(r[1]) for r in emb_rows}
                for r in ctx_rows:
                    vid = str(r.get("id"))
                    vec = emb_map.get(vid, [])
                    r["score"] = _cosine(qvec, vec)

    # Ensure best-first order by score (descending) before assembling
    with suppress(Exception):
        ctx_rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    # Build context & attach filenames
    max_context_chars = 8000
    assembled, total = [], 0
    for c in ctx_rows:
        t = c["text"]
        if total + len(t) > max_context_chars:
            break
        assembled.append(c)
        total += len(t)
    # Attach filenames for UI/tooltips
    with suppress(Exception):
        doc_ids = list({str(c["document_id"]) for c in assembled})
        if doc_ids:
            placeholders = ",".join(["%s"] * len(doc_ids))
            doc_rows = query(f"SELECT id, filename FROM documents WHERE id IN ({placeholders});", tuple(doc_ids))
            doc_map = {str(r[0]): r[1] for r in doc_rows}
            for c in assembled:
                c["filename"] = doc_map.get(str(c["document_id"]))

    # Build ranked context with chunk_id + rank
    for idx, c in enumerate(assembled):
        c["rank"] = idx + 1
        c["chunk_id"] = c.get("id")

    # Build numbered context and k count for prompt/citation rules
    numbered_context = _build_numbered_context(assembled, question)
    k_ctx = len(assembled)

    # Low-confidence gate (min_score)
    low_confidence = False  # NEW
    if min_score is not None and k_ctx:
        try:
            low_confidence = not any(float(c.get("score", 0.0)) >= float(min_score) for c in assembled)
        except Exception:
            low_confidence = False

    # Prepare UI context list (snippet + filename)
    def _snippet(text: str, q: str, max_len: int = 240) -> str:
        if not text: return ""
        lt, lq = text.lower(), q.lower()
        pos = -1
        for tok in [t for t in lq.split() if len(t) > 2]:
            i = lt.find(tok)
            if i != -1:
                pos = i; break
        start = max(0, (pos if pos >= 0 else 0) - 80)
        end = min(len(text), start + max_len)
        return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")

    context = [{
        "chunk_id": c.get("chunk_id"),          # NEW
        "rank": c.get("rank"),                  # NEW
        "document_id": c["document_id"],
        "snippet": _snippet(c["text"], question),
        "filename": c.get("filename"),
        "score": round(float(c.get("score", 0.0)), 4)
    } for c in assembled]

    # Load prompt + add citation rules
    system_prompt, user_template, params = _load_prompt_or_default(p_name, p_ver)
    citation_rules = "\n\nCITATION RULES:\n" + _citation_instructions(k_ctx) if k_ctx else ""
    system_final, user_final = _apply_template(system_prompt + citation_rules, user_template, numbered_context, question)  # CHANGED (context replaced by numbered list)
    # Append explicit numbered context block for clarity
    user_final = user_final + "\n\nNumbered Context:\n" + numbered_context  # NEW
    messages = [{"role": "system", "content": system_final}, {"role": "user", "content": user_final}]

    # Explicit generation parameters (override prompt params if present)
    gen_defaults = {
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 256,          # SDK uses max_tokens (aka max_output_tokens)
        "safe_prompt": False,
        "random_seed": 42
    }
    # Allow v3/v4 params to pass through
    user_params = {k: v for k, v in (params or {}).items()
                   if k in ("temperature", "top_p", "max_tokens", "response_format", "min_sentences", "max_sentences", "target_tokens")}  # CHANGED
    gen_payload = {**gen_defaults, **user_params}

    def _do_call(tag: str):
        gen_start = time.time()
        resp = client.chat.complete(model=settings.model_chat, messages=messages, **gen_payload)
        generation_latency_ms = int((time.time() - gen_start) * 1000)
        answer_local = _extract_full_answer(resp)
        answer_local = _sanitize_citations(answer_local, k_ctx)  # NEW
        usage = getattr(resp, "usage", None)
        in_tok, out_tok = _extract_usage(usage, messages, answer_local)
        logger.info(f"[chat.complete/{tag}] run_id={run_id} retrieval={retrieval_latency_ms}ms generation={generation_latency_ms}ms total={retrieval_latency_ms+generation_latency_ms}ms")
        return answer_local, generation_latency_ms, in_tok, out_tok

    answer, generation_latency_ms, in_tok, out_tok = _do_call("primary")
    if low_confidence:
        answer = "Not in context"  # NEW

    # Skip retries if explicitly "Not in context"
    if answer.strip().lower() == "not in context":
        needs_retry = False
    else:
        needs_retry = (not answer.strip()) or out_tok <= 1 or not _validate_citations(answer, k_ctx)

    if needs_retry:
        logger.warning(f"[chat.complete] retry (citations_missing_or_short={not _validate_citations(answer,k_ctx)} out={out_tok}) run_id={run_id}")
        retry_system = system_final + "\n\nREMINDER: You omitted or mis-numbered citations. Add bracketed markers [n] tied to provided context only."
        retry_messages = [
            {"role": "system", "content": retry_system},
            {"role": "user", "content": user_final}
        ]
        messages = retry_messages
        answer, generation_latency_ms, in_tok, out_tok = _do_call("retry")
        if (not answer.strip()) or out_tok <= 1 or not _validate_citations(answer, k_ctx):
            logger.error(f"[chat.complete] second attempt invalid citations; synthesizing fallback run_id={run_id}")
            answer = _synth_answer(question, context)
            if not _validate_citations(answer, k_ctx):
                # append minimal citation if still missing
                if k_ctx:
                    answer = answer.rstrip(".") + f" [{1}]"
            out_tok = max(out_tok, len(answer)//4)

    total_latency_ms = retrieval_latency_ms + generation_latency_ms  # NEW

    # Persist run completion (include timings + context)
    cost_cents = (
        (in_tok / 1_000_000.0) * settings.input_tokens_cost_per_million +
        (out_tok / 1_000_000.0) * settings.output_tokens_cost_per_million
    )
    row = query("SELECT config FROM runs WHERE id=%s;", (run_id,))
    cfg = row[0][0] if row else run_cfg
    cfg.update({
        "answer": answer,
        "context": context,
        "model": settings.model_chat,
        "retrieval_latency_ms": retrieval_latency_ms,     # NEW
        "generation_latency_ms": generation_latency_ms,   # NEW
        "total_latency_ms": total_latency_ms,             # NEW
        "prompt_name": p_name,                            # ensure explicit
        "prompt_version": p_ver
    })
    query(
        "UPDATE runs SET finished_at=now(), latency_ms=%s, input_tokens=%s, output_tokens=%s, cost_cents=%s, config=%s WHERE id=%s;",
        (total_latency_ms, in_tok, out_tok, round(cost_cents, 4), json.dumps(cfg), run_id)
    )

    return {
        "run_id": str(run_id),
        "answer": answer,
        "context": context,
        "tokens": {"in": in_tok, "out": out_tok},
        "latency_ms": total_latency_ms,              # CHANGED (total)
        "retrieval_latency_ms": retrieval_latency_ms,  # NEW
        "generation_latency_ms": generation_latency_ms,  # NEW
        "degraded_context": degraded_context,
        "context_count": len(context),
        "model": settings.model_chat
    }

# Keep /chat/stream for existing frontend but make it fallback & single response (no token streaming)
@router.post("/stream")
def chat_stream(req: dict):
    if _doc_count() == 0:
        raise HTTPException(400, detail="no_documents_ingested")
    question = req.get("query")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(400, detail="query required")

    retr = req.get("retriever") or {}
    try:
        top_k = int(retr.get("top_k", settings.retriever_top_k))
    except Exception:
        top_k = settings.retriever_top_k
    try:
        min_score = float(retr.get("min_score")) if retr.get("min_score") is not None else None  # NEW
    except Exception:
        min_score = None

    prompt_obj = req.get("prompt") or {}
    p_name = prompt_obj.get("name") or "rag"
    try:
        p_ver = int(prompt_obj.get("version", 1))
    except Exception:
        p_ver = 1

    client = get_client()
    run_rows = query(
        "INSERT INTO runs(kind,config,model,backend) VALUES (%s,%s,%s,%s) RETURNING id;",
        ("chat", json.dumps({"query": question, "top_k": top_k, "prompt": {"name": p_name, "version": p_ver}}),
         settings.model_chat, settings.backend)
    )
    run_id = run_rows[0][0]

    degraded_context = False
    retrieval_start = time.time()  # NEW
    try:
        [qvec] = embed_texts_with_retry([question])
        ctx_rows = topk_by_cosine(qvec, k=top_k)
    except Exception as ex:
        logger.warning(f"embedding degraded (chat_stream): {ex}")
        degraded_context = True
        ctx_rows = []
    retrieval_latency_ms = int((time.time() - retrieval_start) * 1000)  # NEW

    # RECOMPUTE cosine similarity using stored embeddings to avoid 0.0 scores
    if ctx_rows:
        chunk_ids = [r["id"] for r in ctx_rows if r.get("id")]
        if chunk_ids:
            placeholders = ",".join(["%s"] * len(chunk_ids))
            with suppress(Exception):
                emb_rows = query(f"SELECT id, embedding FROM chunks WHERE id IN ({placeholders});", tuple(chunk_ids))
                emb_map = {str(r[0]): _parse_vec(r[1]) for r in emb_rows}
                for r in ctx_rows:
                    vid = str(r.get("id"))
                    vec = emb_map.get(vid, [])
                    r["score"] = _cosine(qvec, vec)

    # Ensure best-first order by score (descending) before assembling
    with suppress(Exception):
        ctx_rows.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)

    # budget
    max_context_chars = 8000
    assembled, total = [], 0
    for c in ctx_rows:
        t = c["text"]
        if total + len(t) > max_context_chars:
            break
        assembled.append(c)
        total += len(t)
    # Attach filenames for UI/tooltips
    with suppress(Exception):
        doc_ids = list({str(c["document_id"]) for c in assembled})
        if doc_ids:
            placeholders = ",".join(["%s"] * len(doc_ids))
            doc_rows = query(f"SELECT id, filename FROM documents WHERE id IN ({placeholders});", tuple(doc_ids))
            doc_map = {str(r[0]): r[1] for r in doc_rows}
            for c in assembled:
                c["filename"] = doc_map.get(str(c["document_id"]))

    # Build ranked context with chunk_id + rank
    for idx, c in enumerate(assembled):
        c["rank"] = idx + 1        # NEW
        c["chunk_id"] = c.get("id")  # NEW

    # Build numbered context and k count for prompt/citation rules
    numbered_context = _build_numbered_context(assembled, question)
    k_ctx = len(assembled)

    # Build citations for UI (include filename)
    def _snippet(text: str, q: str, max_len: int = 240) -> str:
        if not text: return ""
        lt, lq = text.lower(), q.lower()
        pos = -1
        for tok in [t for t in lq.split() if len(t) > 2]:
            i = lt.find(tok)
            if i != -1:
                pos = i; break
        start = max(0, (pos if pos >= 0 else 0) - 80)
        end = min(len(text), start + max_len)
        return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")

    citations_meta = [
        {
            "rank": i + 1,
            "marker": i + 1,
            "chunk_id": c.get("chunk_id"),            # NEW
            "document_id": c.get("document_id"),
            "filename": c.get("filename"),
            "snippet": _snippet(c["text"], question),
            "score": round(float(c.get("score", 0.0)), 4),
        }
        for i, c in enumerate(assembled)
    ]

    # Prompt (inject instruction for inline citation markers) -> replace with numbered_context and rules
    system_prompt, user_template, params = _load_prompt_or_default(p_name, p_ver)
    citation_rules = "\n\nCITATION RULES:\n" + _citation_instructions(k_ctx) if k_ctx else ""
    system_final, user_final = _apply_template(system_prompt + citation_rules, user_template, numbered_context, question)
    user_final = user_final + "\n\nNumbered Context:\n" + numbered_context
    messages = [{"role": "system", "content": system_final}, {"role": "user", "content": user_final}]

    gen_defaults = {
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 256,
        "safe_prompt": False,
        "random_seed": 42
    }
    # CHANGED: drop unsupported max_output_tokens from overrides
    user_params = {k: v for k, v in (params or {}).items() if k in ("temperature", "top_p", "max_tokens")}
    gen_payload = {**gen_defaults, **user_params}

    def gen():
        gen_start = time.time()
        interrupted = False
        answer = ""
        in_tok = 0
        out_tok = 0
        usage = None
        try:
            resp = client.chat.complete(
                model=settings.model_chat,
                messages=messages,
                **gen_payload
            )
            answer = _extract_full_answer(resp)
            answer = _sanitize_citations(answer, k_ctx)
            usage = getattr(resp, "usage", None)
            in_tok, out_tok = _extract_usage(usage, messages, answer)
            # low-confidence override
            if min_score is not None and assembled and not any(float(c.get("score", 0.0)) >= float(min_score) for c in assembled):
                answer = "Not in context"
            # retry logic (only if not explicit Not in context)
            if (answer.strip().lower() != "not in context" and ((not answer.strip()) or out_tok <= 1 or not _validate_citations(answer, k_ctx))):
                logger.warning(f"[chat.stream] retry path run_id={run_id} out_tok={out_tok}")
                simplified_user = f"{question}\n\nExcerpts:\n" + "\n---\n".join(x["text"] for x in assembled[:3])
                messages_retry = [
                    {"role": "system", "content": "Answer using excerpts. Cite [#]."},
                    {"role": "user", "content": simplified_user}
                ]
                resp = client.chat.complete(
                    model=settings.model_chat,
                    messages=messages_retry,
                    **gen_payload
                )
                answer = _extract_full_answer(resp)
                answer = _sanitize_citations(answer, k_ctx)
                usage = getattr(resp, "usage", None)
                in_tok, out_tok = _extract_usage(usage, messages_retry, answer)
                if (not answer.strip()) or out_tok <= 1 or not _validate_citations(answer, k_ctx):
                    answer = _synth_answer(question, citations_meta)
                    if k_ctx and not _validate_citations(answer, k_ctx):
                        answer = answer.rstrip(".") + " [1]"
                    out_tok = max(out_tok, len(answer) // 4)
        except Exception as ex:
            interrupted = True
            answer = f"[error] {ex}"

        # emit citation events
        for meta in citations_meta:
            yield "data: " + json.dumps({"citation": meta, "run_id": str(run_id)}) + "\n\n"

        # simulate token streaming
        tokens = []
        if answer:
            parts = answer.split(" ")
            for i, p in enumerate(parts):
                tokens.append(p if i == 0 else " " + p)
        sent = 0
        for tok in tokens:
            if interrupted:
                break
            sent += 1
            yield "data: " + json.dumps({
                "delta": tok,
                "run_id": str(run_id),
                "stats": {"tokens_out": sent}
            }) + "\n\n"

        generation_latency_ms = int((time.time() - gen_start) * 1000)
        total_latency_ms = retrieval_latency_ms + generation_latency_ms
        latency_ms = generation_latency_ms  # legacy field for final payload

        final_payload = {
            "final": True,
            "run_id": str(run_id),
            "answer": answer,
            "tokens": {"in": in_tok, "out": out_tok},
            "latency_ms": latency_ms,
            "degraded_context": degraded_context,
            "citations": citations_meta,
            "interrupted": interrupted,
            "model": settings.model_chat,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": total_latency_ms
        }

        # persist run
        cfg_row = query("SELECT config FROM runs WHERE id=%s;", (run_id,))
        cfg = cfg_row[0][0] if cfg_row else {"query": question}
        cfg.update({
            "answer": answer,
            "citations": citations_meta,
            "context": citations_meta,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": total_latency_ms,
            "prompt_name": p_name,
            "prompt_version": p_ver
        })
        cost_cents = round(
            (in_tok / 1_000_000.0) * settings.input_tokens_cost_per_million +
            (out_tok / 1_000_000.0) * settings.output_tokens_cost_per_million,
            4
        )
        query(
            "UPDATE runs SET finished_at=now(), latency_ms=%s, input_tokens=%s, output_tokens=%s, cost_cents=%s, config=%s WHERE id=%s;",
            (total_latency_ms, in_tok, out_tok, cost_cents, json.dumps(cfg), run_id)
        )

        yield "data: " + json.dumps(final_payload) + "\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@router.get("/settings")
def chat_settings():
    """
    Lightweight settings exposure for frontend (backend toggle visibility).
    """
    return {
        "backend": settings.backend,
        "model_chat": settings.model_chat,
        "model_embed": settings.model_embed
    }
