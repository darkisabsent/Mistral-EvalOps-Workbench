from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from ..core.db import query
import os
from pathlib import Path
import uuid
from typing import List
from datetime import datetime, timezone
import logging
from contextlib import suppress
import shutil  # added for file/directory removal

logger = logging.getLogger("api")

# Text extraction: prefer pdfminer, fallback to PyPDF2. No OCR, no image tricks.
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# Optional OCR deps (enabled by default if present)
try:
    from pdf2image import convert_from_path as pdf2image_convert
except Exception:
    pdf2image_convert = None
try:
    import pytesseract as _pytesseract
except Exception:
    _pytesseract = None

router = APIRouter(prefix="/documents", tags=["documents"])

# ------------------------- helpers -------------------------

def is_true_env(name: str, default: str = "1") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "y", "on"}

def embed_dim() -> int:
    with suppress(Exception):
        return int(os.getenv("EMBED_DIM", "1024"))
    return 1024

def docs_dir() -> Path:
    env = os.getenv("DOCS_DIR")
    if env and os.path.isdir(env):
        return Path(env)
    return Path.cwd() / "datasets" / "docs"

def ensure_chunks_schema() -> None:
    # Minimal, deterministic. Don’t assume extensions exist.
    # Try to enable extensions if possible; tolerate failure.
    with suppress(Exception):
        query("CREATE EXTENSION IF NOT EXISTS pgcrypto;")  # gen_random_uuid()

    vector_ok = False
    try:
        # Attempt to ensure pgvector is available
        query("CREATE EXTENSION IF NOT EXISTS vector;")
        vector_ok = True
    except Exception as ex:
        logger.warning(f"pgvector not available or cannot be created (non-fatal): {ex}")
        with suppress(Exception):
            r = query("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector');")
            vector_ok = bool(r and r[0][0])

    # Create table with the appropriate embedding column type.
    # If pgvector is unavailable, fallback to DOUBLE PRECISION[].
    emb_col = f"vector({embed_dim()})" if vector_ok else "DOUBLE PRECISION[]"

    query("""
      CREATE TABLE IF NOT EXISTS chunks (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        ordinal INT NOT NULL,
        text TEXT NOT NULL,
        embedding """ + emb_col + """,
        created_at TIMESTAMPTZ DEFAULT now()
      );
    """)
    query("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);")

    if vector_ok:
        # Only create ivfflat index when vector type exists
        query("""
          CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine
          ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """)
        logger.info(f"chunks schema ready with pgvector, dim={embed_dim()}")
    else:
        logger.info("chunks schema ready without pgvector (embedding stored as DOUBLE PRECISION[])")

def chunks_table_exists() -> bool:
    r = query("SELECT to_regclass('public.chunks');")
    return bool(r and r[0][0])

def normalize_text(s: str) -> str:
    # collapse whitespace; good enough for chunking
    return " ".join((s or "").split())

def ocr_pdf_to_text(path: Path) -> str:
    if not is_true_env("ENABLE_OCR", "1"):
        return ""
    if not (pdf2image_convert and _pytesseract):
        logger.warning("OCR requested but pdf2image/pytesseract not installed; skipping OCR.")
        return ""
    # Settings
    try:
        dpi = int(os.getenv("OCR_DPI", "200"))
    except Exception:
        dpi = 200
    try:
        max_pages = int(os.getenv("OCR_MAX_PAGES", "0"))  # 0 = all pages
    except Exception:
        max_pages = 0
    lang = os.getenv("OCR_LANG", "eng")

    try:
        images = pdf2image_convert(str(path), dpi=dpi)
        if max_pages and len(images) > max_pages:
            images = images[:max_pages]
        texts = []
        for i, img in enumerate(images):
            t = _pytesseract.image_to_string(img, lang=lang) or ""
            if t.strip():
                texts.append(t)
        txt = normalize_text("\n".join(texts))
        logger.info(f"OCR extracted {len(txt)} chars from {path.name} (pages={len(images)})")
        return txt
    except Exception as ex:
        logger.warning(f"OCR failed for {path.name}: {ex}")
        return ""

def extract_text_from_pdf(path: Path) -> str:
    # Text-native PDFs only. OCR fallback if enabled.
    txt = ""
    used = "none"
    if pdfminer_extract_text:
        try:
            txt = pdfminer_extract_text(str(path)) or ""
            if txt.strip():
                used = "pdfminer"
        except Exception:
            txt = ""
    if not txt and PdfReader:
        try:
            reader = PdfReader(str(path))
            buf = []
            for p in reader.pages:
                t = p.extract_text() or ""
                if t: buf.append(t)
            txt = "\n".join(buf)
            if txt.strip():
                used = "pypdf2"
        except Exception:
            txt = ""
    txt = normalize_text(txt)
    if not txt:
        # OCR fallback
        ocr_txt = ocr_pdf_to_text(path)
        if ocr_txt:
            txt = ocr_txt
            used = "ocr"

    logger.info(f"text extraction method={used} length={len(txt)} file={path.name}")
    return txt

def chunk_text(txt: str, size: int = 1000, overlap: int = 150) -> list[str]:
    out, i, n = [], 0, len(txt)
    if n == 0: return out
    while i < n:
        end = min(i + size, n)
        piece = txt[i:end].strip()
        if piece:
            # merge tiny tail fragments into previous chunk
            if out and len(piece) < size // 4:
                out[-1] += " " + piece
            else:
                out.append(piece)
        if end >= n: break
        nxt = end - overlap
        if nxt <= i: nxt = end
        i = nxt
    return out

def embed_or_fail(texts: list[str]) -> list[list[float]]:
    # No silent fallbacks. If embeddings fail, we mark the doc and stop.
    from ..rag.embeddings import embed_texts
    vecs = embed_texts(texts)
    if not isinstance(vecs, list) or len(vecs) != len(texts):
        raise RuntimeError("embedding shape mismatch")
    return vecs

# ------------------------- endpoints -------------------------

@router.get("/summary")
def documents_summary():
    try:
        # CHANGED: treat embedded + partial_embedded as ready
        ok_count = int(query("SELECT COUNT(*) FROM documents WHERE status IN ('ok','embedded','partial_embedded');")[0][0])
    except Exception:
        ok_count = 0
    total_chunks = 0
    if chunks_table_exists():
        try: total_chunks = int(query("SELECT COUNT(*) FROM chunks;")[0][0])
        except Exception: total_chunks = 0
    try:
        r = query("SELECT MAX(created_at) FROM documents;")
        last_ingest = r[0][0].isoformat() if r and r[0][0] else None
    except Exception:
        last_ingest = None
    return {"ok_count": ok_count, "total_chunks": total_chunks, "last_ingest": last_ingest}

@router.get("")
def list_documents():
    if chunks_table_exists():
        rows = query("""
            SELECT d.id, d.filename, d.status, d.size, d.created_at,
                   COALESCE((SELECT COUNT(*) FROM chunks c WHERE c.document_id=d.id),0) AS chunk_count
            FROM documents d
            ORDER BY d.created_at DESC
            LIMIT 500;
        """)
        return {"documents": [{
            "id": r[0],
            "filename": r[1],
            "status": r[2],
            "size": r[3],
            "created_at": r[4].isoformat(),
            "chunks": r[5],
        } for r in rows]}
    rows = query("""
        SELECT id, filename, status, size, created_at
        FROM documents
        ORDER BY created_at DESC
        LIMIT 500;
    """)
    return {"documents": [{
        "id": r[0],
        "filename": r[1],
        "status": r[2],
        "size": r[3],
        "created_at": r[4].isoformat(),
        "chunks": 0
    } for r in rows]}

@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    base = docs_dir()
    base.mkdir(parents=True, exist_ok=True)
    ensure_chunks_schema()

    created, total_size = [], 0

    for up in files:
        name = (up.filename or "").strip()
        if not name.lower().endswith(".pdf"):
            # consume and ignore non-PDFs
            await up.read()
            logger.info(f"skip non-PDF upload: {name}")
            continue

        doc_id = str(uuid.uuid4())
        path = base / f"{doc_id}.pdf"
        raw = await up.read()
        total_size += len(raw)

        logger.info(f"upload received: id={doc_id} file={name} size={len(raw)} bytes")

        # write file
        try:
            with path.open("wb") as f:
                f.write(raw)
        finally:
            await up.close()

        # insert document row as 'new'
        try:
            query(
                "INSERT INTO documents(id, filename, size, status) VALUES (%s,%s,%s,%s);",
                (doc_id, name or "document.pdf", len(raw), "new")
            )
        except Exception as ex:
            with suppress(Exception): path.unlink(missing_ok=True)
            logger.error(f"DB insert failed for id={doc_id}: {ex}")
            raise HTTPException(500, f"DB insert failed: {ex}")

        # extract text (text-native PDFs + OCR fallback if enabled)
        text = extract_text_from_pdf(path)
        if len(text) < 80:  # minimum viable content
            query("UPDATE documents SET status=%s WHERE id=%s;", ("empty", doc_id))
            created.append({"id": doc_id, "filename": name, "status": "empty", "size": len(raw), "chunks": 0})
            logger.info(f"document marked empty (len={len(text)}): id={doc_id} file={name}")
            continue

        # chunk (explicit size/overlap)
        parts = chunk_text(text, size=1000, overlap=150)
        logger.info(f"chunked document id={doc_id} into {len(parts)} chunks (size=1000 overlap=150)")
        if not parts:
            query("UPDATE documents SET status=%s WHERE id=%s;", ("empty", doc_id))
            created.append({"id": doc_id, "filename": name, "status": "empty", "size": len(raw), "chunks": 0})
            continue

        # embed + persist (validate embedding dim = EMBED_DIM)
        try:
            vecs = embed_or_fail(parts)
            # Validate and normalize embedding vectors
            expected_dim = embed_dim()
            for vi, v in enumerate(vecs):
                if not hasattr(v, "__len__") or len(v) != expected_dim:
                    raise RuntimeError(f"embedding dim mismatch for chunk {vi}: expected {expected_dim}, got {getattr(v,'__len__', lambda: 'invalid')() if hasattr(v,'__len__') else 'invalid'}")
            for i, (ct, emb) in enumerate(zip(parts, vecs)):
                # ensure plain Python list[float]
                emb_list = [float(x) for x in emb]
                query(
                    "INSERT INTO chunks(document_id, ordinal, text, embedding) VALUES (%s,%s,%s,%s);",
                    (doc_id, i, ct, emb_list)
                )
            query("UPDATE documents SET status=%s WHERE id=%s;", ("ok", doc_id))
            created.append({"id": doc_id, "filename": name, "status": "ok", "size": len(raw), "chunks": len(parts)})
            logger.info(f"embedded and stored {len(parts)} chunks for id={doc_id}")
        except Exception as ex:
            # rollback partial chunks and mark failure
            with suppress(Exception): query("DELETE FROM chunks WHERE document_id=%s;", (doc_id,))
            query("UPDATE documents SET status=%s WHERE id=%s;", ("embed_failed", doc_id))
            created.append({"id": doc_id, "filename": name, "status": "embed_failed", "size": len(raw), "chunks": 0})
            logger.error(f"embedding failed for id={doc_id}: {ex}")

    if not created:
        raise HTTPException(400, "No PDF files uploaded")

    now = datetime.now(timezone.utc)
    collection_id = now.strftime("docs_%Y%m%d_%H%M%S")

    return {
        "collection": {
            "id": collection_id,
            "name": collection_id,
            "files": len(created),
            "chunks": sum(d["chunks"] for d in created),
            "size_bytes": total_size,
            "status": "uploaded",
            "created_at": now.isoformat()
        },
        "file_ids": [d["id"] for d in created],
        "documents": created
    }

@router.get("/{doc_id}/file")
def get_document_file(doc_id: str):
    base = docs_dir()
    rows = query("SELECT filename FROM documents WHERE id=%s;", (doc_id,))
    if not rows:
        raise HTTPException(404, "Document not found")
    filename = rows[0][0] or ""
    # prefer ID.pdf; fallback to original filename
    candidates = list(base.glob(f"{doc_id}.*")) or list(base.glob(filename))
    if not candidates:
        raise HTTPException(404, "File not found on disk")
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    media = "application/pdf" if path.suffix.lower() == ".pdf" else "text/plain"
    return FileResponse(str(path), media_type=media,
                        headers={"Content-Disposition": f'inline; filename="{path.name}"'})

@router.delete("/all")
def delete_all_documents(confirm: bool = False):
    """
    Delete all documents and associated chunks from the DB and remove files under the docs directory.
    Requires query param ?confirm=true to actually perform deletion.
    """
    if not confirm:
        raise HTTPException(400, "Confirmation required. Call /documents/all?confirm=true to delete all documents and files.")

    base = docs_dir()
    # Count docs before delete (best-effort)
    try:
        r = query("SELECT COUNT(*) FROM documents;")
        n_docs = int(r[0][0]) if r and r[0][0] else 0
    except Exception:
        n_docs = 0

    # Delete DB rows (documents -> chunks cascade)
    try:
        # prefer deleting documents (will cascade); guard if table missing
        query("DELETE FROM documents;")
    except Exception as ex:
        logger.error(f"DB deletion failed: {ex}")
        raise HTTPException(500, f"DB delete failed: {ex}")

    # Remove files from disk under docs_dir()
    removed_files = 0
    try:
        if base.exists() and base.is_dir():
            for p in base.iterdir():
                try:
                    if p.is_file():
                        p.unlink()
                        removed_files += 1
                    elif p.is_dir():
                        shutil.rmtree(p)
                        removed_files += 1
                except Exception as ex:
                    logger.warning(f"Failed to remove {p}: {ex}")
    except Exception as ex:
        logger.error(f"Filesystem cleanup failed: {ex}")
        # don't abort—report what we removed so far
    logger.info(f"Deleted {n_docs} documents and removed {removed_files} files from {base}")
    return {"deleted_documents": n_docs, "removed_files": removed_files}

# --- Retrieval Probe (non-chat) ---

@router.post("/retrieve")
def retrieve_chunks(body: dict):
    """
    Retrieval-only probe to test ANN quality before generation.
    Input: { "query": "string", "k": 8 }
    Output: { "query": "...", "results": [{chunk_id, document_id, filename, score, snippet}] }
    """
    q = (body or {}).get("query", "")
    if not isinstance(q, str) or not q.strip():
        raise HTTPException(400, "query required")
    try:
        k = int((body or {}).get("k", 8))
    except Exception:
        k = 8
    k = max(1, min(50, k))

    # ensure chunks table exists and at least one ok document
    if not chunks_table_exists():
        raise HTTPException(400, "chunks table not initialized (ingest a PDF first)")
    try:
        ok_docs = int(query("SELECT COUNT(*) FROM documents WHERE status='ok';")[0][0])
    except Exception:
        ok_docs = 0
    if ok_docs == 0:
        raise HTTPException(400, "no_documents_ingested")

    # embed + ANN retrieve (clean imports; remove invalid '..app.rag.retriever')
    from ..rag.embeddings import embed_texts
    from ..rag.retriever import topk_by_cosine
    [qvec] = embed_texts([q])
    rows = topk_by_cosine(qvec, k=k)  # [{id, document_id, text, score}]

    # map document_id -> filename
    doc_ids = list({r["document_id"] for r in rows})
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
        # prefer first meaningful token hit
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

    results = [{
        "chunk_id": r["id"],
        "document_id": r["document_id"],
        "filename": name_map.get(str(r["document_id"]), None),
        "score": round(float(r["score"]), 6),
        "snippet": make_snippet(r["text"], q, 240),
    } for r in rows]

    return {"query": q, "results": results}
