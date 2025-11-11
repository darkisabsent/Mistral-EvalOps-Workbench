import json, uuid
from typing import Iterator, Tuple, List, Dict
from ..core.settings import settings
from ..core.db import query
from ..rag.embeddings import embed_texts
from ..rag.retriever import topk_by_cosine
import logging

logger = logging.getLogger("api")

REQUIRED_FIELDS = {"question"}
ALLOWED_FIELDS = {"qa_key", "question", "reference_answer", "doc_ids", "doc_keys", "tags"}

def _is_uuid(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except Exception:
        return False

def _table_has_columns(table: str, *cols: str) -> Dict[str, bool]:
    # detect optional schema columns
    try:
        placeholders = ",".join(["%s"] * len(cols))
        rows = query(
            f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name=%s AND column_name IN ({placeholders})
            """,
            (table, *cols),
        )
        have = {r[0] for r in rows}
        return {c: (c in have) for c in cols}
    except Exception:
        return {c: False for c in cols}

def validate_jsonl_lines(lines: List[str]) -> List[Dict]:
    items = []
    for idx, raw in enumerate(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            raise ValueError(f"Line {idx}: invalid JSON")
        if not isinstance(obj, dict):
            raise ValueError(f"Line {idx}: item not object")
        # support legacy `id` -> `qa_key`
        if "id" in obj and "qa_key" not in obj:
            obj["qa_key"] = obj["id"]
            del obj["id"]
        if not REQUIRED_FIELDS.issubset(obj.keys()):
            missing = REQUIRED_FIELDS - obj.keys()
            raise ValueError(f"Line {idx}: missing required fields {missing}")
        unknown = set(obj.keys()) - ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"Line {idx}: unknown fields {unknown}")
        if not isinstance(obj["question"], str):
            raise ValueError(f"Line {idx}: question must be string")
        ra = obj.get("reference_answer")
        if ra is not None and not isinstance(ra, str):
            raise ValueError(f"Line {idx}: reference_answer must be string")
        doc_ids = obj.get("doc_ids")
        if doc_ids is not None and (not isinstance(doc_ids, list) or any(not isinstance(d, str) for d in doc_ids)):
            raise ValueError(f"Line {idx}: doc_ids must be list[string]")
        doc_keys = obj.get("doc_keys")
        if doc_keys is not None and (not isinstance(doc_keys, list) or any(not isinstance(d, str) for d in doc_keys)):
            raise ValueError(f"Line {idx}: doc_keys must be list[string]")
        tags = obj.get("tags")
        if tags is not None and (not isinstance(tags, list) or any(not isinstance(t, str) for t in tags)):
            raise ValueError(f"Line {idx}: tags must be list[string]")
        items.append({
            "qa_key": str(obj["qa_key"]) if "qa_key" in obj else None,
            "question": obj["question"],
            "reference_answer": ra,
            "doc_ids": doc_ids or [],
            "doc_keys": doc_keys or [],
            "tags": tags or []
        })
        if len(items) > settings.dataset_max_size:
            raise ValueError(f"Dataset exceeds max size {settings.dataset_max_size}")
    if not items:
        raise ValueError("Empty dataset")
    return items

def register_dataset(name: str, file_bytes: bytes) -> Dict:
    # parse and validate dataset JSONL payload
    try:
        text = file_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = ""
    lines = text.splitlines()
    items = validate_jsonl_lines(lines)

    # compute coverage of UUID anchors
    distinct_doc_ids = {d for it in items for d in it["doc_ids"] if _is_uuid(d)}
    coverage = 0
    if distinct_doc_ids:
        placeholders = ",".join(["%s"] * len(distinct_doc_ids))
        rows = query(f"SELECT id FROM documents WHERE id IN ({placeholders})", tuple(distinct_doc_ids))
        coverage = len(rows)

    # insert dataset metadata
    have_ds = _table_has_columns("datasets", "status", "coverage_covered", "coverage_total", "version", "doc_coverage")
    if have_ds.get("doc_coverage"):
        ds_row = query(
            "INSERT INTO datasets(name,size,doc_coverage) VALUES (%s,%s,%s) RETURNING id;",
            (name, len(items), coverage)
        )
    else:
        ds_row = query("INSERT INTO datasets(name,size) VALUES (%s,%s) RETURNING id;", (name, len(items)))
    dataset_id = ds_row[0][0]

    # update optional columns if present
    if have_ds.get("status") and have_ds.get("coverage_covered") and have_ds.get("coverage_total") and have_ds.get("version"):
        status = "partial" if any(it["doc_keys"] for it in items) or (any(it["doc_ids"] for it in items) and coverage < len(items)) else "ready"
        query(
            "UPDATE datasets SET status=%s, coverage_covered=%s, coverage_total=%s, version=%s WHERE id=%s;",
            (status, coverage, len(items), 1, dataset_id)
        )

    # insert dataset_items; include optional columns where available
    have_items = _table_has_columns("dataset_items", "qa_key", "raw_doc_refs", "raw_doc_keys", "tags", "resolved_doc_ids")
    for i, it in enumerate(items):
        doc_ids_json = json.dumps(it["doc_ids"])
        cols = ["dataset_id", "ordinal", "question", "reference_answer", "doc_ids"]
        vals = [dataset_id, i, it["question"], it["reference_answer"], doc_ids_json]
        if have_items.get("qa_key") and it.get("qa_key"):
            cols.insert(2, "qa_key")
            vals.insert(2, it["qa_key"])
        if have_items.get("raw_doc_refs"):
            cols.append("raw_doc_refs"); vals.append(it["doc_ids"])
        if have_items.get("raw_doc_keys"):
            cols.append("raw_doc_keys"); vals.append(it["doc_keys"])
        if have_items.get("tags"):
            cols.append("tags"); vals.append(it["tags"])
        placeholders = ",".join(["%s"] * len(vals))
        query(f"INSERT INTO dataset_items({', '.join(cols)}) VALUES ({placeholders});", tuple(vals))

    logger.info("Registered dataset %s id=%s size=%d coverage=%d", name, dataset_id, len(items), coverage)
    return {"dataset_id": dataset_id, "size": len(items), "doc_coverage": coverage}

def list_datasets() -> List[Dict]:
    rows = query("SELECT id,name,size,doc_coverage,created_at FROM datasets ORDER BY created_at DESC;")
    return [{"id": r[0], "name": r[1], "size": r[2], "doc_coverage": r[3], "created_at": r[4].isoformat()} for r in rows]

def preview_dataset(dataset_id: str, limit: int = 5) -> List[Dict]:
    rows = query("SELECT id, qa_key, question, reference_answer, doc_ids FROM dataset_items WHERE dataset_id=%s ORDER BY ordinal LIMIT %s;",
                 (dataset_id, limit))
    out = []
    for r in rows:
        item_uuid, qa_key, question, reference_answer, doc_ids_raw = r
        if isinstance(doc_ids_raw, list):
            doc_ids = [str(x) for x in doc_ids_raw if isinstance(x, str)]
        elif isinstance(doc_ids_raw, str):
            try:
                tmp = json.loads(doc_ids_raw)
                doc_ids = [str(x) for x in tmp if isinstance(x, str)]
            except Exception:
                doc_ids = []
        else:
            doc_ids = []
        out.append({
            "item_id": item_uuid,
            "qa_key": qa_key,
            "question": question,
            "reference_answer": reference_answer,
            "doc_ids": doc_ids
        })
    return out

def iterate_dataset(dataset_id: str, context_k: int = 8) -> Iterator[Tuple[str, str, str, List[str]]]:
    rows = query("SELECT id, question, reference_answer, doc_ids FROM dataset_items WHERE dataset_id=%s ORDER BY ordinal;",
                 (dataset_id,))
    for r in rows:
        item_id, question, reference_answer, doc_ids_json = r
        try:
            doc_ids = json.loads(doc_ids_json) if doc_ids_json else []
        except Exception:
            doc_ids = []
        if doc_ids:
            placeholders = ",".join(["%s"] * len(doc_ids))
            chunk_rows = query(f"""
                SELECT text FROM chunks 
                WHERE document_id IN ({placeholders})
                ORDER BY document_id, ordinal
                LIMIT %s;
            """, (*doc_ids, context_k))
            context_snippets = [c[0] for c in chunk_rows]
        else:
            [qvec] = embed_texts([question])
            retrieved = topk_by_cosine(qvec, k=context_k)
            context_snippets = [it["text"] for it in retrieved]
        yield (item_id, question, reference_answer, context_snippets)
