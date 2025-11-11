from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import chat, ingest, eval as eval_router, metrics, datasets, runs, documents, prompts
from .core.logging import setup_logging, LoggingMiddleware
from .core.db import query
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
import hashlib
import json

app = FastAPI(title="Mistral EvalOps API")
setup_logging()
app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(eval_router.router)
app.include_router(metrics.router)
app.include_router(datasets.router)
app.include_router(runs.router)
app.include_router(documents.router)
app.include_router(prompts.router)

def _detect_docs_dir() -> str:
    # Prefer env, else search up for "<repo>/datasets/docs"
    env = os.getenv("DOCS_DIR")
    if env and os.path.isdir(env):
        return env
    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "datasets" / "docs"
        if cand.exists():
            return str(cand)
    # fallback: relative from current working dir
    return str(Path.cwd() / "datasets" / "docs")

DOCS_DIR = _detect_docs_dir()
try:
    app.mount("/static/docs", StaticFiles(directory=DOCS_DIR, html=False), name="docs_static")
    print(f"[static] Serving docs from {DOCS_DIR} at /static/docs")
except Exception as ex:
    print("[static] Skipped mounting /static/docs:", ex)

# Replace deprecated @app.on_event with lifespan handler
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed docs from filesystem
    root = Path(DOCS_DIR)
    if root.exists():
        files = [*root.glob("*.pdf"), *root.glob("*.txt"), *root.glob("*.md")]
        seeded = 0
        for p in files:
            try:
                h = hashlib.sha256()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                hashval = h.hexdigest()
                rows = query("SELECT id FROM documents WHERE hash=%s LIMIT 1;", (hashval,))
                if rows:
                    continue
                query(
                    "INSERT INTO documents(filename,size,hash,status) VALUES (%s,%s,%s,%s);",
                    (p.name, p.stat().st_size, hashval, "ok")
                )
                seeded += 1
            except Exception as ex:
                print("[seed] failed for", p.name, ":", ex)
        if seeded:
            print(f"[seed] Registered {seeded} new file(s) from {root}")
    else:
        print("[seed] DOCS_DIR missing:", root)

    # Seed prompt versions
    defs = [
        {
            "name": "rag", "version": 1,
            "system": (
                'You answer ONLY using the <Context> chunks. If the answer is not supported, reply exactly: "Not in context".\n'
                "Do not reveal chain-of-thought.\n"
                "Write a single paragraph of up to 3 sentences (max ~80 tokens total).\n"
                "Cite supporting chunks with bracketed indices like [1] or [2][5]. Use the provided chunk indices only."
            ),
            "user_template": "<Question>\n{{question}}\n</Question>\n\n<Context>\n{{context}}\n</Context>",
            "params": {"temperature": 0.2}
        },
        {
            "name": "rag", "version": 2,
            "system": (
                'You answer ONLY using the <Context> chunks. If the answer is not supported, reply exactly: "Not in context".\n'
                "Do not reveal chain-of-thought.\n"
                "Write a single paragraph of up to 3 sentences (max ~80 tokens total).\n"
                "Cite supporting chunks with bracketed indices like [1] or [2][5]. Use the provided chunk indices only.\n"
                "- Each sentence MUST end with at least one citation.\n"
                "- Do not invent indices that are not present in <Context>.\n"
                "- Prefer quoting short phrases from the context."
            ),
            "user_template": "<Question>\n{{question}}\n</Question>\n\n<Context>\n{{context}}\n</Context>",
            "params": {"temperature": 0.1, "top_p": 1}
        },
        {
            "name": "rag", "version": 3,
            "system": (
                'You answer ONLY using the <Context>. If unsupported, set "answer" to "Not in context".\n'
                "Return a strict JSON object with:\n"
                "{\n"
                '  "answer": string,\n'
                '  "citations": [ /* unique integer indices */ ]\n'
                "}\n"
                "No other fields. Do not reveal chain-of-thought."
            ),
            "user_template": "<Question>\n{{question}}\n</Question>\n\n<Context>\n{{context}}\n</Context>",
            "params": {"temperature": 0.2, "response_format": "json_object"}
        },
        {
            "name": "rag", "version": 4,
            "system": (
                "You answer ONLY using the <Context>.\n"
                "Write between {{min_sentences}} and {{max_sentences}} sentences, one paragraph, ~{{target_tokens}} tokens total.\n"
                "Every sentence must end with citations [i] using provided indices.\n"
                'If unsupported: "Not in context".\n'
                "No extra prose."
            ),
            "user_template": "<Question>\n{{question}}\n</Question>\n\n<Context>\n{{context}}\n</Context>",
            "params": {"temperature": 0.2, "min_sentences": 2, "max_sentences": 4, "target_tokens": 70}
        },
    ]
    for d in defs:
        try:
            row = query("SELECT 1 FROM prompts WHERE name=%s AND version=%s LIMIT 1;", (d["name"], d["version"]))
            if row: continue
            query(
                "INSERT INTO prompts(name, version, system, user_template, params) VALUES (%s,%s,%s,%s,%s);",
                (d["name"], d["version"], d["system"], d["user_template"], json.dumps(d["params"]))
            )
        except Exception as ex:
            print("[seed:prompts] skip/failed for", f'{d["name"]}@{d["version"]}', ":", ex)

    yield  # allow app startup to continue

app.router.lifespan_context = lifespan

@app.get("/healthz")
def healthz(): return {"ok": True}
