import { API_BASE } from "./env";

export const API_BASE_URL = API_BASE || "http://localhost:8001";

function genRequestId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return "req-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export interface ApiError extends Error {
  status: number;
  requestId: string;
  data?: any;
}

function makeError(status: number, message: string, requestId: string = genRequestId(), data?: any): ApiError {
  const err = { name: "ApiError", message, status, requestId, data } as ApiError;
  Object.setPrototypeOf(err, Error.prototype);
  if (Error.captureStackTrace) Error.captureStackTrace(err, makeError);
  else err.stack = (new Error(message)).stack;
  return err;
}

function extractErrorMessage(status: number, data: any, text: string): string {
  // FastAPI style
  if (data && Array.isArray(data.detail)) {
    const msgs = data.detail.map((d: any) => d?.msg || JSON.stringify(d)).join("; ");
    return `[${status}] ${msgs}`;
  }
  if (data && typeof data.detail === "string") return `[${status}] ${data.detail}`;
  if (text) return `[${status}] ${text}`;
  return `Request failed ${status}`;
}

async function coreFetch(path: string, init: RequestInit & { cache?: RequestCache } = {}) {
  const requestId = genRequestId();
  const isForm = typeof FormData !== "undefined" && init?.body instanceof FormData;
  const isMutation = init.method === "POST" || init.method === "PUT" || init.method === "PATCH";
  const baseHeaders: Record<string, string> = {
    "x-request-id": requestId,
    ...(isMutation && !isForm ? { "Content-Type": (init.headers as any)?.["Content-Type"] || "application/json" } : {}),
    Accept: "application/json",
  };
  init.headers = { ...(init.headers || {}), ...baseHeaders };

  const ctrl = new AbortController();
  const outerSignal = (init as any).signal as AbortSignal | undefined;
  const timeoutMs = 30000;
  let timeoutId: any;
  if (!outerSignal) {
    (init as any).signal = ctrl.signal;
    timeoutId = setTimeout(() => ctrl.abort(), timeoutMs);
  }

  const maxRetries = 1;
  let attempt = 0;
  try {
    for (;;) {
      try {
        const res = await fetch(`${API_BASE_URL}${path}`, init as RequestInit);
        const respReqId = res.headers.get("x-request-id") || requestId;
        const text = await res.text().catch(() => "");
        let data: any = null;
        try { data = text ? JSON.parse(text) : null; } catch {}
        if (!res.ok) {
          const msg = extractErrorMessage(res.status, data, text);
          throw makeError(res.status, msg, respReqId, data);
        }
        return data;
      } catch (err: any) {
        const isApi = err?.name === "ApiError";
        const isAbort = err?.name === "AbortError";
        if (!isApi && !isAbort && attempt < maxRetries) {
          attempt++;
          await new Promise(r => setTimeout(r, 200));
          continue;
        }
        if (isApi) throw err;
        const hint = "Network error. Ensure the API is reachable at " + API_BASE_URL + path + " and CORS is allowed.";
        throw makeError(0, err?.message ? `${err.message} — ${hint}` : hint, requestId);
      }
    }
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

export async function getJSON<T = any>(path: string): Promise<T> {
  return coreFetch(path, { method: "GET", cache: "no-store" });
}
export async function postJSON<T = any>(path: string, body: any): Promise<T> {
  return coreFetch(path, { method: "POST", body: JSON.stringify(body) });
}
export async function delJSON<T = any>(path: string): Promise<T> {
  return coreFetch(path, { method: "DELETE" });
}
export async function postForm<T = any>(path: string, form: FormData): Promise<T> {
  return coreFetch(path, { method: "POST", body: form });
}

// --- Documents APIs ---
export async function listDocCollections(): Promise<{ id: string; name: string; files: number; chunks: number; size_bytes: number; status: string; created_at: string }[]> {
  try {
    const r = await getJSON<any>("/documents/collections");
    return (r.collections || []).map((c: any) => ({
      id: c.id, name: c.name, files: c.files, chunks: c.chunks, size_bytes: c.size_bytes, status: c.status, created_at: c.created_at,
    }));
  } catch (e) {
    console.warn("listDocCollections failed", e);
    return [];
  }
}

export async function ingestDocs(files: File[]): Promise<{ document_ids: string[]; counts: { docs: number; chunks: number }; failed_files: string[] }> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  return postForm("/ingest", fd);
}

export async function ingestDocsAsCollection(files: File[]): Promise<{ collection: any; failed_files: any[] }> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f, f.name));
  const r = await postForm<any>("/ingest", fd);
  return {
    collection: {
      id: r.collection?.id,
      name: r.collection?.name,
      files: r.collection?.files,
      chunks: r.collection?.chunks,
      size_bytes: r.collection?.size_bytes,
      status: r.collection?.status,
      created_at: r.collection?.created_at,
    },
    failed_files: r.failed_files || [],
  };
}

export async function documentsSummary(): Promise<{ ok_count: number; total_chunks: number; last_ingest: string | null }> {
  try {
    return await getJSON("/documents/summary");
  } catch (e) {
    console.warn("documentsSummary failed (offline fallback)", e);
    return { ok_count: 0, total_chunks: 0, last_ingest: null };
  }
}

// --- Datasets / QA APIs ---
export interface QaCoverage { covered: number; total: number; percent: number }
export interface QaSetRow {
  dataset_id: string; id?: string; name: string; items: number;
  doc_coverage?: QaCoverage; coverage?: QaCoverage;
  status?: string; created_at: string;
}

export async function uploadQaJsonl(file: File, name: string) {
  console.log("[uploadQaJsonl] uploading", { name, filename: file.name, size: file.size, type: file.type });
  const fd = new FormData();
  fd.append("file", file, file.name);
  const res = await postForm(`/datasets/upload?name=${encodeURIComponent(name)}`, fd);
  console.log("[uploadQaJsonl] server response:", res);
  return res;
}

export async function listQaSets(): Promise<QaSetRow[]> {
  try {
    const r = await getJSON<any>("/datasets");
    // Support both { items: [...] } and { datasets: [...] }
    const rows = r.items || r.datasets || [];
    return rows.map((d: any) => {
      const cov = d.coverage || d.doc_coverage || { covered: d.doc_coverage ?? 0, total: d.size ?? d.items ?? 0, percent: d.percent ?? 0 };
      return {
        dataset_id: d.dataset_id || d.id,
        id: d.id,
        name: d.name,
        items: d.items ?? d.size ?? 0,
        coverage: cov,
        doc_coverage: cov,
        status: d.status || "ready",
        created_at: d.created_at,
      } as QaSetRow;
    });
  } catch (e) {
    console.warn("listQaSets failed", e);
    return [];
  }
}

export async function listDatasets(): Promise<{ id: string; name: string; size: number; created_at: string }[]> {
  try {
    const r = await getJSON<any>("/datasets");
    const arr = r.datasets || r.items || [];
    return arr.map((d: any) => ({ id: d.id || d.dataset_id, name: d.name, size: d.size ?? d.items ?? 0, created_at: d.created_at }));
  } catch (e) {
    console.warn("listDatasets failed (offline fallback)", e);
    return [];
  }
}

export async function qaSetPreview(id: string, limit = 20): Promise<{ dataset_id: string; sample: any[]; invalid_rows?: number }> {
  const r = await getJSON(`/datasets/${id}/preview?limit=${limit}`);
  console.log("[qaSetPreview] preview for", id, "=>", r);
  return r;
}

export async function deleteQaSet(id: string): Promise<boolean> {
  try {
    const r = await delJSON<any>(`/datasets/${id}`);
    return !!r.deleted;
  } catch {
    return false;
  }
}

export function qaSetExportUrl(id: string): string {
  return `${API_BASE_URL}/datasets/${encodeURIComponent(id)}/export`;
}

// Optional helpers introduced for coverage/resolve UI (no-op if API lacks them)
export async function datasetCoverage(id: string): Promise<{ covered: number; total: number; percent: number; unresolved_items?: string[] }> {
  try { return await getJSON(`/datasets/${id}/coverage`); } catch { return { covered: 0, total: 0, percent: 0, unresolved_items: [] }; }
}
export async function documentOptions(search = "", limit = 20): Promise<{ options: { doc_id: string; name: string; size_mb: number; created_at: string; collection: string; chunks: number }[] }> {
  const qs = new URLSearchParams(); if (search) qs.set("search", search); qs.set("limit", String(limit));
  try { return await getJSON(`/documents/options?${qs.toString()}`); } catch { return { options: [] }; }
}
export async function resolveDatasetAnchors(dataset_id: string, body: {
  map_keys?: { key: string; doc_ids: string[] }[];
  map_items?: { item_id: string; doc_ids: string[] }[];
  accept_suggestions?: { item_id: string; doc_ids: string[] }[];
}): Promise<{ dataset_id: string; version: number; coverage?: QaCoverage; unresolved_count: number; changed_items: string[] }> {
  try { return await postJSON(`/datasets/${dataset_id}/resolve`, body); } catch { return { dataset_id, version: 1, unresolved_count: 0, changed_items: [] }; }
}
export async function autoResolveSuggestions(dataset_id: string, threshold = 0.75, use_reference_answer = true): Promise<any> {
  try { return await postJSON(`/datasets/${dataset_id}/auto-resolve`, { threshold, use_reference_answer }); } catch { return { dataset_id, suggestions: [] }; }
}

// --- Prompts / Evals / Runs ---
export async function listPrompts(): Promise<{ name: string; version: number; system: string; user_template: string; params: any; created_at: string }[]> {
  try {
    const r = await getJSON<any>("/prompts");
    return (r.prompts || []).map((p: any) => ({
      name: p.name, version: p.version, system: p.system, user_template: p.user_template, params: p.params, created_at: p.created_at,
    }));
  } catch (e) {
    console.warn("listPrompts failed (offline fallback)", e);
    return [];
  }
}

export async function getPrompt(name: string, version: number): Promise<{ name: string; version: number; system: string; user_template: string; params: any; created_at: string }> {
  return await getJSON(`/prompts/${encodeURIComponent(name)}/${version}`);
}

export async function createPromptVersion(input: {
  name: string;
  source_version?: number;
  system?: string;
  user_template?: string;
  params?: any;
}): Promise<{ name: string; version: number }> {
  const body: any = { name: input.name };
  if (input.source_version !== undefined) body.source_version = input.source_version;
  if (input.system !== undefined) body.system = input.system;
  if (input.user_template !== undefined) body.user_template = input.user_template;
  if (input.params !== undefined) body.params = input.params;
  const res = await postJSON<any>("/prompts", body);
  return { name: res.name, version: res.version };
}

export async function deletePromptVersion(name: string, version: number): Promise<boolean> {
  try {
    const r = await delJSON<any>(`/prompts/${encodeURIComponent(name)}/${version}`);
    return !!r.deleted;
  } catch {
    return false;
  }
}

export async function startEvalRun(payload: any): Promise<any> {
  return postJSON("/eval/run", payload);
}

// NEW: A/B eval helper
export async function startABEval(payload: {
  dataset_id: string;
  variant_a: { prompt: { name: string; version: number }; retriever: { top_k: number } };
  variant_b: { prompt: { name: string; version: number }; retriever: { top_k: number } };
}): Promise<{ group_id: string; run_a_id: string; run_b_id: string }> {
  return postJSON("/eval/run", payload);
}

// NEW: Judge a single run
export async function judgeRun(run_id: string, judge_cfg?: { model?: string; repeats?: number }): Promise<{
  judged: number;
  relevance_avg: number;
  groundedness_avg: number;
}> {
  return postJSON("/eval/judge", { run_id, judge: judge_cfg || {} });
}

// NEW: Batch eval
export async function startBatchEval(group_id: string, judge_model?: string): Promise<{
  batch_id: string;
  status: string;
  requests_count: number;
  eta_seconds: number;
}> {
  return postJSON("/eval/batch", { ab_group_id: group_id, judge_model });
}

export async function getBatchStatus(batch_id: string): Promise<{
  batch_id: string;
  status: string;
  completed: number;
  total: number;
}> {
  return getJSON(`/eval/batch/${batch_id}`);
}

// NEW: listRuns
export async function listRuns(
  filters?: Record<string, string | number | boolean | undefined | null>,
  pagination?: { limit?: number; offset?: number }
): Promise<any[]> {
  try {
    const qs = new URLSearchParams();
    if (filters) {
      Object.entries(filters).forEach(([k, v]) => { if (v !== undefined && v !== null) qs.set(k, String(v)); });
    }
    if (pagination?.limit !== undefined) qs.set("limit", String(pagination.limit));
    if (pagination?.offset !== undefined) qs.set("offset", String(pagination.offset));
    const r = await getJSON<any>(`/runs${qs.toString() ? `?${qs.toString()}` : ""}`);
    return r.runs || [];
  } catch {
    return [];
  }
}

// NEW: getRun
export async function getRun(id: string): Promise<any> {
  try {
    return await getJSON(`/runs/${id}`);
  } catch {
    return null;
  }
}

// NEW: metricsSummary
export async function metricsSummary(): Promise<{
  p50_latency_ms: number;
  p95_latency_ms: number;
  avg_tokens_in: number;
  avg_tokens_out: number;
  avg_cost_cents: number;
  relevance_avg: number;
  groundedness_avg: number;
}> {
  try {
    return await getJSON("/metrics/summary");
  } catch {
    return {
      p50_latency_ms: 0,
      p95_latency_ms: 0,
      avg_tokens_in: 0,
      avg_tokens_out: 0,
      avg_cost_cents: 0,
      relevance_avg: 0,
      groundedness_avg: 0,
    };
  }
}

// --- Streaming helpers (SSE-like over fetch) ---
export function sse(
  path: string,
  body: any,
  onEvent: (evt: any) => void,
  onError?: (err: ApiError) => void
) {
  const ctrl = new AbortController();
  const requestId = genRequestId();
  fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-request-id": requestId },
    body: JSON.stringify(body),
    signal: ctrl.signal,
  }).then(async (res) => {
    const respReqId = res.headers.get("x-request-id") || requestId;
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      let data: any = null; try { data = JSON.parse(txt); } catch {}
      onError && onError(makeError(res.status, (data && data.detail) || txt || "SSE failed", respReqId, data));
      return;
    }
    const reader = res.body?.getReader?.();
    if (!reader) return;
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const p of parts) {
        const lines = p.split("\n");
        let dataLine = lines.find((l) => l.startsWith("data: "));
        if (!dataLine && lines.length === 1) dataLine = "data: " + lines[0];
        if (!dataLine) continue;
        const payload = dataLine.slice(6);
        try {
          const evt = JSON.parse(payload);
          onEvent(evt);
        } catch {
          onEvent({ data: payload });
        }
      }
    }
  }).catch((err) => {
    onError && onError(makeError(0, err?.message || "Network error", requestId));
  });
  return { abort: () => ctrl.abort() };
}

export function sseStream(
  path: string,
  body: any,
  onDelta: (token: string, meta: any) => void,
  onError?: (err: ApiError) => void,
  onComplete?: (final: any) => void
) {
  return sse(path, body, (ev) => {
    if (ev.complete) {
      onComplete && onComplete(ev);
    } else if (typeof ev.delta === "string") {
      onDelta(ev.delta, ev);
    } else if (ev.delta !== undefined) {
      onDelta(String(ev.delta ?? ""), ev);
    }
  }, onError);
}

export function streamChat(
  payload: {
    query: string;
    retriever?: { top_k?: number };
    prompt?: { name?: string; version?: number };
  },
  onDelta: (token: string, meta: any) => void,
  onFinal: (final: any) => void,
  onError?: (err: ApiError) => void,
  onCitation?: (c: {
    rank: number; marker: number; document_id: string; snippet: string; score?: number; filename?: string  // CHANGED: add filename
  }) => void
) {
  let done = false;
  return sse("/chat/stream", payload, (ev) => {
    if (ev.citation && !done) {
      onCitation && onCitation(ev.citation);
    } else if (typeof ev.delta === "string" && !done) {
      onDelta(ev.delta, ev);
    } else if (ev.final && !done) {
      done = true;
      onFinal(ev);
    } else if (ev.interrupted && !done) {
      done = true;
      onFinal(ev);
    }
  }, onError);
}

// Basic Chat (non-streaming)
export interface ChatCompleteInput {
  query: string;
  retriever?: { top_k?: number };
  prompt?: { name?: string; version?: number };
}
export interface ChatCompleteOutput {
  run_id: string;
  answer: string;
  context: { document_id: string; snippet: string; score: number; filename?: string }[]; // CHANGED: add filename
  tokens: { in: number; out: number };
  latency_ms: number;
  degraded_context?: boolean;
  context_count?: number;
  model?: string; // NEW: show model in footer if backend returns it
}
export async function chatComplete(body: ChatCompleteInput): Promise<ChatCompleteOutput> {
  return await postJSON("/chat/complete", body);
}

// --- Convenience try* wrappers ---
export async function tryGetJSON<T = any>(path: string): Promise<{ ok: true; data: T } | { ok: false; error: ApiError }> {
  try { const data = await getJSON<T>(path); return { ok: true, data }; }
  catch (e: any) { const err = e || makeError(0, "Network error", genRequestId()); return { ok: false, error: err }; }
}
export async function tryPostJSON<T = any>(path: string, body: any): Promise<{ ok: true; data: T } | { ok: false; error: ApiError }> {
  try { const data = await postJSON<T>(path, body); return { ok: true, data }; }
  catch (e: any) { const err = e || makeError(0, "Network error", genRequestId()); return { ok: false, error: err }; }
}
export async function tryPostForm<T = any>(path: string, form: FormData): Promise<{ ok: true; data: T } | { ok: false; error: ApiError }> {
  try { const data = await postForm<T>(path, form); return { ok: true, data }; }
  catch (e: any) { const err = e || makeError(0, "Network error", genRequestId()); return { ok: false, error: err }; }
}
export async function tryDelJSON<T = any>(path: string): Promise<{ ok: true; data: T } | { ok: false; error: ApiError }> {
  try { const data = await delJSON<T>(path); return { ok: true, data }; }
  catch (e: any) { const err = e || makeError(0, "Network error", genRequestId()); return { ok: false, error: err }; }
}

// --- Legacy/deprecated helpers kept for compatibility ---
let autogenInFlight = false;
export async function autogenDataset(name: string = "docs_auto"): Promise<any> {
  if (autogenInFlight) throw makeError(0, "Autogen already running", genRequestId());
  autogenInFlight = true;
  const path = `/datasets/autogen?name=${encodeURIComponent(name)}`;
  try {
    try { return await postJSON(path, {}); }
    catch { return await getJSON(path); }
  } finally {
    autogenInFlight = false;
  }
}

// New document helpers
export type DocFile = {
  id: string;
  filename: string;
  status: string;
  size: number;
  created_at: string;
  chunks: number; // added
};

// Upload PDFs to /documents/upload (multipart)
export async function uploadDocuments(files: File[]): Promise<{ collection: any; documents: DocFile[] }> {
  const fd = new FormData();
  files.forEach((f) => {
    if (f.name.toLowerCase().endsWith(".pdf")) fd.append("files", f, f.name);
  });
  const res = await postForm("/documents/upload", fd);
  return { collection: res.collection, documents: res.documents || [] };
}

// List documents from /documents
export async function listDocuments(): Promise<DocFile[]> {
  try {
    const r = await getJSON<any>("/documents");
    return (r.documents || []).map((d: any) => ({
      id: d.id,
      filename: d.filename,
      status: d.status,
      size: d.size ?? 0,
      created_at: d.created_at,
      chunks: d.chunks ?? d.chunk_count ?? d.num_chunks ?? 0, // added mapping
    }));
  } catch (e) {
    console.warn("listDocuments failed", e);
    return [];
  }
}

// Build preview URL for a document
export function documentFileUrl(id: string): string {
  return `${API_BASE_URL}/documents/${encodeURIComponent(id)}/file`;
}

// delete all documents/files (requires confirm=true)
export async function deleteAllDocuments(confirm = false): Promise<{ deleted_documents: number; removed_files: number }> {
  const path = `/documents/all?confirm=${confirm ? "1" : "0"}`;
  return await delJSON(path);
}

// NEW: Retrieval probe
export async function retrieveChunks(query: string, k: number = 8): Promise<{
  query: string;
  results: Array<{
    chunk_id: string;
    document_id: string;
    filename: string | null;
    score: number;
    snippet: string;
  }>;
}> {
  return postJSON("/ingest/retrieve", { query, k });
}

export async function ingestDocsWithOcr(
  files: File[],
  { ocr = true }: { ocr?: boolean } = {}
): Promise<{ document_ids: string[]; counts: { docs: number; chunks: number }; failed_files: string[] }> {
  const fd = new FormData();
  files.forEach(f => fd.append("files", f));
  return postForm(`/ingest${ocr ? "?ocr=1" : ""}`, fd);
}

// NEW: diffPrompts
export async function diffPrompts(name: string, a: number, b: number): Promise<{
  name: string; version_a: number; version_b: number;
  system_diff: string[]; user_template_diff: string[];
  params_added: Record<string, any>;
  params_removed: Record<string, any>;
  params_changed: Record<string, [any, any]>;
}> {
  return getJSON(`/prompts/diff?name=${encodeURIComponent(name)}&a=${a}&b=${b}`);
}

// NEW: getRunsByGroup
export async function getRunsByGroup(group_id: string): Promise<{ group_id: string; runs: any[] }> {  // NEW
  return getJSON(`/runs/group/${encodeURIComponent(group_id)}`);
}