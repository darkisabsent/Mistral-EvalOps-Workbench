from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Response
from ..rag.chunker import chunk_text
from ..rag.embeddings import embed_texts
from ..core.db import query
import io, hashlib
from pypdf import PdfReader
from uuid import uuid4
from pathlib import Path
import os

router = APIRouter(prefix="/ingest", tags=["ingest"])

CHUNK_SIZE = 300        # NEW: smaller chunk size
CHUNK_OVERLAP = 60      # NEW: smaller overlap

def _docs_dirs() -> tuple[Path, Path]:
    """
    Returns (legacy_dir, root_dir)
    legacy_dir: apps/api/datasets/docs
    root_dir:   <repo_root>/datasets/docs (walk up to find repo root that contains 'apps')
    """
    here = Path(__file__).resolve()
    legacy = here.parents[2] / "datasets" / "docs"
    legacy.mkdir(parents=True, exist_ok=True)
    root = None
    for p in here.parents:
        if (p / "apps").exists():
            root = p
            break
    if root is None:
        # fallback a few levels up
        root = here.parents[4] if len(here.parents) > 4 else here.parents[-1]
    root_dir = root / "datasets" / "docs"
    root_dir.mkdir(parents=True, exist_ok=True)
    return legacy, root_dir

def _extract_text(file: UploadFile, data: bytes, ocr: bool = False) -> str:
    name = (file.filename or "").lower()
    if name.endswith(".pdf") or (file.content_type or "") == "application/pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join([p.extract_text() or "" for p in reader.pages])
            if text.strip():
                return text
            if ocr:
                try:
                    from pdf2image import convert_from_bytes  # type: ignore
                    import pytesseract  # type: ignore
                    images = convert_from_bytes(data)
                    ocr_texts = [pytesseract.image_to_string(img) for img in images]
                    return "\n\n".join(ocr_texts)
                except Exception:
                    return ""
            return ""
        except Exception:
            return ""
    try:
        return data.decode(errors="ignore")
    except Exception:
        return ""

def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _has_pgvector() -> bool:
    try:
        r = query("SELECT 1 FROM pg_extension WHERE extname='vector';")
        return bool(r)
    except Exception:
        return False

@router.post("")
async def ingest(
    files: list[UploadFile] = File(...),
    ocr: int = Query(0, description="Enable OCR for scanned PDFs (1=true, 0=false)")
):
    trace_id = str(uuid4())
    if not files:
        raise HTTPException(400, detail="No files provided")
    use_pgvector = _has_pgvector()
    print(f"[ingest] pgvector_enabled={use_pgvector}")

    doc_ids, total_chunks = [], 0
    failed_files = []
    legacy_path, root_path = _docs_dirs()
    use_ocr = bool(ocr)

    print(f"[ingest] trace={trace_id} files={len(files)} legacy_dir={legacy_path} root_dir={root_path} ocr={use_ocr}")

    for f in files:
        raw = await f.read()
        size = len(raw)
        file_hash = _sha256(raw)
        print(f"[ingest] recv filename={f.filename} size={size} sha256={file_hash[:12]}...")

        # Check for duplicates by content hash
        dup_rows = query("SELECT id, status FROM documents WHERE hash=%s;", (file_hash,))
        if dup_rows:
            existing_id, existing_status = dup_rows[0]
            print(f"[ingest] duplicate hash matches doc_id={existing_id} status={existing_status}")
            # Re-run chunk/embed only if previous failed (no chunks or embed_failed)
            if existing_status not in ("embedded", "ok"):
                doc_id = existing_id
                print(f"[ingest] reprocessing failed duplicate doc_id={doc_id}")
            else:
                continue
        else:
            # Insert document record first to get UUID (status will be set to ok at the end)
            rows = query(
                "INSERT INTO documents(filename,size,hash,status) VALUES (%s,%s,%s,%s) RETURNING id;",
                (f.filename, size, file_hash, "new")
            )
            doc_id = rows[0][0]
            doc_ids.append(doc_id)
            print(f"[ingest] inserted document id={doc_id}")

        # Decide on extension and storage filename
        orig_lower = (f.filename or "").lower()
        ext = ".pdf" if (orig_lower.endswith(".pdf") or (f.content_type or "") == "application/pdf") else (Path(orig_lower).suffix or ".bin")
        stored_name = f"{doc_id}{ext}"
        legacy_file = legacy_path / stored_name
        root_file = root_path / stored_name

        # Persist raw file to disk (both locations for compatibility)
        save_ok = False
        try:
            with legacy_file.open("wb") as out:
                out.write(raw)
            save_ok = True
            print(f"[ingest] saved to legacy {legacy_file}")
        except Exception as ex:
            print(f"[ingest] legacy save failed: {ex}")
        try:
            with root_file.open("wb") as out:
                out.write(raw)
            save_ok = True
            print(f"[ingest] saved to root   {root_file}")
        except Exception as ex:
            print(f"[ingest] root save failed: {ex}")

        if not save_ok:
            query("UPDATE documents SET status='file_save_failed' WHERE id=%s;", (doc_id,))
            failed_files.append(f.filename or stored_name)
            print(f"[ingest] ERROR: could not save file for doc_id={doc_id}")
            continue

        # Update stored filename so preview can locate it
        query("UPDATE documents SET filename=%s WHERE id=%s;", (stored_name, doc_id))

        # Extract text (PDF text first, optional OCR if empty and ocr=1)
        text = _extract_text(f, raw, ocr=use_ocr)
        print(f"[ingest] text_len={len(text or '')} for doc_id={doc_id}")
        if not text.strip():
            query("UPDATE documents SET status='unreadable', chunk_count=0 WHERE id=%s;", (doc_id,))
            failed_files.append(f.filename or stored_name)
            print(f"[ingest] WARN: unreadable (no text) doc_id={doc_id}")
            continue
        
        # Chunk and embed (guarantee at least 1 chunk)
        # Debug: report chunker defaults used in UI (visible in Advanced settings)
        print(f"[ingest] chunker params size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}")
        chunks = chunk_text(text)
        if chunks and len(chunks) == 1 and len(text) <= 1200:
            # Force re-chunking smaller for better granularity
            manual = []
            start = 0
            while start < len(text):
                end = min(len(text), start + CHUNK_SIZE)
                manual.append(text[start:end])
                start = end - CHUNK_OVERLAP
                if start < 0: start = 0
                if end == len(text): break
            if len(manual) > 1:
                chunks = manual
        if not chunks and text.strip():
            chunks = [text.strip()[:CHUNK_SIZE]]
        try:
            num_chunks = len(chunks)
        except Exception:
            num_chunks = 0
        print(f"[ingest] chunker returned {num_chunks} chunks for doc_id={doc_id}")
        if num_chunks:
            lens = [len(c) for c in chunks]
            print(f"[ingest] chunk lengths sample: {lens[:6]}")
            print(f"[ingest] first chunk preview (200 chars): {chunks[0][:200].replace(chr(10),' ')}")
        if not chunks and text.strip():
            fallback = text.strip()[:4000]
            chunks = [fallback]
            print(f"[ingest] chunker returned 0; using fallback chunk for doc_id={doc_id}")

        if not chunks:
            query("UPDATE documents SET status='empty', chunk_count=0 WHERE id=%s;", (doc_id,))
            failed_files.append(f.filename or stored_name)
            print(f"[ingest] WARN: empty after chunking doc_id={doc_id}")
            continue
        
        print(f"[ingest] chunks={len(chunks)} (first_chunk_len={len(chunks[0])}) doc_id={doc_id}")

        # Embeddings (must succeed; on failure mark embed_failed and skip vector insert)
        try:
            vecs = embed_texts(chunks)
            embed_error = False
            if vecs and isinstance(vecs[0], (list, tuple)):
                print(f"[ingest] embedded vectors count={len(vecs)} dim={len(vecs[0])}")
        except Exception as ex:
            print(f"[ingest] EMBED CALL FAILED doc_id={doc_id}: {ex}")
            vecs = None
            embed_error = True

        inserted = 0
        if vecs:
            # Always store in embedding_arr (fallback) + optionally pgvector column
            for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
                # text row first (ensures row exists even if both inserts fail)
                try:
                    if use_pgvector:
                        # Try pgvector insert
                        query(
                            "INSERT INTO chunks(document_id, ordinal, text, embedding, embedding_arr) VALUES (%s,%s,%s,%s::vector,%s);",
                            (doc_id, i, chunk, _vec_literal(vec), vec)
                        )
                    else:
                        # No pgvector: store only array column
                        query(
                            "INSERT INTO chunks(document_id, ordinal, text, embedding_arr) VALUES (%s,%s,%s,%s);",
                            (doc_id, i, chunk, vec)
                        )
                    inserted += 1
                except Exception as ex:
                    # Fallback: store text + array only
                    print(f"[ingest] chunk insert degraded ordinal={i} doc_id={doc_id} ex={ex}")
                    try:
                        query(
                            "INSERT INTO chunks(document_id, ordinal, text, embedding_arr) VALUES (%s,%s,%s,%s);",
                            (doc_id, i, chunk, vec)
                        )
                        inserted += 1
                    except Exception as ex2:
                        print(f"[ingest] chunk insert failed final ordinal={i} doc_id={doc_id} ex={ex2}")
            status_final = "embedded" if inserted == len(chunks) else "partial_embedded"
        else:
            # Only when the embedding API failed; still insert plain text chunks so we can backfill later
            for i, chunk in enumerate(chunks):
                try:
                    query("INSERT INTO chunks(document_id, ordinal, text) VALUES (%s,%s,%s);", (doc_id, i, chunk))
                    inserted += 1
                except Exception as ex:
                    print(f"[ingest] plain chunk insert failed ordinal={i} doc_id={doc_id} ex={ex}")
            status_final = "embed_failed" if embed_error else "no_vectors"

        # Determine DB status (normalize embedded variants to 'ok' so summaries work)
        status_db = "ok" if status_final in ("embedded", "partial_embedded") else status_final
        query("UPDATE documents SET status=%s, chunk_count=%s WHERE id=%s;", (status_db, len(chunks), doc_id))
        total_chunks += len(chunks)
        print(f"[ingest] DONE doc_id={doc_id} status_final={status_final} stored_status={status_db} chunks={len(chunks)} pgvector={'yes' if use_pgvector else 'no'} inserted={inserted}")

    print(f"[ingest] complete trace={trace_id} docs={len(doc_ids)} total_chunks={total_chunks} failed={len(failed_files)}")
    return {
        "document_ids": doc_ids,
        "counts": {"docs": len(doc_ids), "chunks": total_chunks},
        "failed_files": failed_files,
        "trace_id": trace_id
    }

@router.post("/retrieve")
def retrieve_chunks(body: dict, response: Response):
    """
    Retrieval-only probe to test ANN quality before generation.
    Input: { "query": "string", "k": 8 }
    Output: { "query": "...", "results": [{chunk_id, document_id, filename, score, similarity_pct, snippet}] }
    """
    # Disable client/proxy caches so UI doesn't show stale results
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    q = (body or {}).get("query", "")
    if not isinstance(q, str):
        raise HTTPException(400, "query required")
    q = q.strip()
    if not q:
        print("[retrieve] empty query after strip")
        raise HTTPException(400, "query required")
    try:
        k = int((body or {}).get("k", 8))
    except Exception:
        k = 8
    k = max(1, min(50, k))

    # Ensure we have chunks to retrieve from
    try:
        ok_docs = int(query("SELECT COUNT(*) FROM documents WHERE chunk_count > 0;")[0][0])
    except Exception:
        ok_docs = 0
    if ok_docs == 0:
        raise HTTPException(400, "no_documents_ingested")

    # Embed query once
    from ..rag.embeddings import embed_texts as _embed
    qvec = _embed([q])[0]
    qdim = len(qvec) if isinstance(qvec, (list, tuple)) else 0
    use_pg = _has_pgvector()
    print(f"[retrieve] q='{q[:60]}...' k={k} dim={qdim} pgvector={'yes' if use_pg else 'no'}")

    results_raw = []

    if use_pg:
        # pgvector path: cosine similarity = 1 - distance; JOIN documents
        vec_lit = _vec_literal(qvec)
        sql = """
          SELECT
            c.id AS chunk_id,
            d.id AS document_id,
            d.filename,
            c.text,
            1 - (c.embedding <=> %s::vector) AS similarity
          FROM chunks c
          JOIN documents d ON d.id = c.document_id
          WHERE c.embedding IS NOT NULL
          ORDER BY c.embedding <=> %s::vector
          LIMIT %s;
        """
        rows = query(sql, (vec_lit, vec_lit, k))
        print(f"[retrieve] pgvector rows={len(rows)}")
        for rid, doc_id, _fname, txt, sim in rows:
            results_raw.append({"id": rid, "document_id": str(doc_id), "text": txt, "similarity": float(sim or 0.0)})
    else:
        # Fallback: use embedding_arr (double precision[]) and compute cosine in Python; JOIN documents
        rows = query("""
            SELECT c.id, d.id, d.filename, c.text, c.embedding_arr
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding_arr IS NOT NULL
            LIMIT 2000;
        """)
        print(f"[retrieve] array fallback rows={len(rows)} (pre-score)")
        import math
        def _cos(a: list, b: list) -> float:
            if not a or not b: return 0.0
            n = min(len(a), len(b))
            dot = sum((a[i] or 0.0) * (b[i] or 0.0) for i in range(n))
            na = math.sqrt(sum((a[i] or 0.0) ** 2 for i in range(n)))
            nb = math.sqrt(sum((b[i] or 0.0) ** 2 for i in range(n)))
            if na == 0.0 or nb == 0.0: return 0.0
            return dot / (na * nb)

        if not rows:
            # BACKFILL: no vector arrays present; embed a small batch of chunks and update embedding_arr
            to_fill = query("""
                SELECT c.id, c.text
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding_arr IS NULL
                ORDER BY c.created_at DESC
                LIMIT 200;
            """)
            print(f"[retrieve] backfill: embedding_arr missing rows={len(to_fill)}")
            if to_fill:
                ids = [r[0] for r in to_fill]
                texts = [r[1] for r in to_fill]
                try:
                    from ..rag.embeddings import embed_texts as _embed_arr
                    vecs = _embed_arr(texts)
                    for cid, vec in zip(ids, vecs):
                        try:
                            query("UPDATE chunks SET embedding_arr=%s WHERE id=%s;", (vec, cid))
                        except Exception as ex:
                            print(f"[retrieve] backfill update failed id={cid}: {ex}")
                except Exception as ex:
                    print(f"[retrieve] backfill embed failed: {ex}")
            # Re-run select after backfill
            rows = query("""
                SELECT c.id, d.id, d.filename, c.text, c.embedding_arr
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding_arr IS NOT NULL
                LIMIT 2000;
            """)
            print(f"[retrieve] array fallback rows={len(rows)} (post-backfill)")

        if rows:
            scored = []
            for rid, doc_id, _fname, txt, arr in rows:
                sim = _cos(arr or [], qvec or [])
                scored.append({"id": rid, "document_id": str(doc_id), "text": txt, "similarity": sim})
            scored.sort(key=lambda r: r["similarity"], reverse=True)
            results_raw = scored[:k]
        else:
            # FINAL FALLBACK: lexical-only ranking to avoid empty results
            print("[retrieve] vector arrays still empty; using lexical-only fallback")
            bulk = query("""
                SELECT c.id, d.id, d.filename, c.text
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.created_at DESC
                LIMIT 1000;
            """)
            def _lex(txt: str) -> float:
                if not txt: return 0.0
                toks = [t for t in q.lower().split() if len(t) > 2]
                tl = txt.lower()
                hits = sum(1 for t in set(toks) if t in tl)
                return hits / max(1, len(set(toks)))
            scored = [{"id": rid, "document_id": str(doc_id), "text": txt, "similarity": _lex(txt)} for rid, doc_id, _fname, txt in bulk]
            scored.sort(key=lambda r: r["similarity"], reverse=True)
            results_raw = scored[:k]

    # Map document filenames for display
    doc_ids = list({r["document_id"] for r in results_raw})
    name_map = {}
    if doc_ids:
        placeholders = ",".join(["%s"] * len(doc_ids))
        r = query(f"SELECT id, filename FROM documents WHERE id IN ({placeholders});", tuple(doc_ids))
        name_map = {str(x[0]): x[1] for x in r}

    def make_snippet(text: str, query_text: str, max_len: int = 240) -> str:
        if not text:
            return ""
        lt, lq = text.lower(), query_text.lower()
        hit = -1
        toks = [t for t in lq.split() if len(t) > 2]
        for t in toks:
            pos = lt.find(t)
            if pos != -1:
                hit = pos
                break
        start = max(0, (hit if hit != -1 else 0) - 80)
        end = min(len(text), start + max_len)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return prefix + text[start:end] + suffix

    results = []
    for i, r in enumerate(results_raw, start=1):
        sim = max(0.0, min(1.0, float(r["similarity"])))
        sim_pct = round(sim * 100.0, 1)
        results.append({
            "chunk_id": r["id"],
            "document_id": r["document_id"],
            "filename": name_map.get(str(r["document_id"]), None),
            "score": sim,
            "similarity_pct": sim_pct,
            "snippet": make_snippet(r["text"], q, 240),
        })

    if results:
        print(f"[retrieve] top1 sim={results[0]['score']:.4f} ({results[0]['similarity_pct']}%) doc={results[0]['document_id']} chunk={results[0]['chunk_id']}")
    print(f"[retrieve] returned={len(results)} for k={k}")

    return {"query": q, "results": results}
