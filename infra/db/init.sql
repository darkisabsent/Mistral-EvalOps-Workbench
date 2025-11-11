-- REQUIRE pgvector (no fallback). Installation must succeed before app start.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- documents
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  size BIGINT,
  hash TEXT,
  status TEXT NOT NULL DEFAULT 'ok',
  duplicate_of UUID
);

-- chunks (embedding: 1024 dims for mistral-embed) -- embedding MUST be vector(1024)
CREATE TABLE IF NOT EXISTS chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ordinal INT NOT NULL,
  text TEXT NOT NULL,
  embedding vector(1024),
  meta JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- vector index (cosine)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine
ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- prompts
CREATE TABLE IF NOT EXISTS prompts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  version INT NOT NULL,
  system TEXT,
  user_template TEXT,
  params JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(name, version)
);

CREATE INDEX IF NOT EXISTS idx_prompts_name_version ON prompts(name, version DESC);

-- runs (chat/eval)
CREATE TABLE IF NOT EXISTS runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  kind TEXT NOT NULL,             -- "chat" | "eval"
  config JSONB NOT NULL,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  latency_ms INT,
  input_tokens INT,
  output_tokens INT,
  cost_cents NUMERIC(10,4) DEFAULT 0,
  model TEXT NOT NULL,
  backend TEXT NOT NULL           -- "mistral" | "vllm"
);

-- judgements (LLM-as-judge)
CREATE TABLE IF NOT EXISTS judgements (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  query TEXT NOT NULL,
  answer TEXT NOT NULL,
  refs JSONB NOT NULL,            -- doc_ids/snippets used
  relevance INT NOT NULL,         -- 1..5
  groundedness INT NOT NULL,      -- 1..5
  rationale TEXT NOT NULL
);

-- datasets registry
CREATE TABLE IF NOT EXISTS datasets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  size INT NOT NULL,               -- UI maps this to "items"
  doc_coverage INT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Datasets (versioned + coverage split)
ALTER TABLE datasets
  ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'partial',
  ADD COLUMN IF NOT EXISTS coverage_covered INT DEFAULT 0,
  ADD COLUMN IF NOT EXISTS coverage_total INT DEFAULT 0,
  ADD COLUMN IF NOT EXISTS version INT DEFAULT 1;

-- Legacy columns kept: size, doc_coverage (still raw count of matched doc_ids) for back-compat.

-- Dataset items anchor metadata + qa_key
ALTER TABLE dataset_items
  ADD COLUMN IF NOT EXISTS qa_key TEXT,
  ADD COLUMN IF NOT EXISTS resolved_doc_ids UUID[] NULL,
  ADD COLUMN IF NOT EXISTS raw_doc_refs TEXT[] NULL,
  ADD COLUMN IF NOT EXISTS raw_doc_keys TEXT[] NULL,
  ADD COLUMN IF NOT EXISTS tags TEXT[] NULL;

-- Mapping table (key → doc_ids per dataset version)
CREATE TABLE IF NOT EXISTS dataset_doc_aliases (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  version INT NOT NULL,
  key TEXT NOT NULL,
  doc_ids UUID[] NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_dataset_doc_aliases_dataset ON dataset_doc_aliases(dataset_id);

-- remove uniqueness on documents(hash) to allow duplicate files/content
-- DROP INDEX IF EXISTS idx_documents_hash;
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);

