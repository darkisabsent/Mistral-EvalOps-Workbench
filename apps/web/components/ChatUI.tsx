"use client";
import React, { useState, useEffect, useRef } from "react";
import { chatComplete, documentsSummary, streamChat, listPrompts, getPrompt, createPromptVersion } from "../lib/api";

const ChatUI: React.FC = () => {
  const [query, setQuery] = useState(
    "In one paragraph, define hybrid search and explain why combining BM25 with vectors improves recall and precision."
  );
  const [answer, setAnswer] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [tokensOut, setTokensOut] = useState(0);
  const [tokensIn, setTokensIn] = useState(0);
  const [citations, setCitations] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [runId, setRunId] = useState<string | undefined>();
  const [latencyMs, setLatencyMs] = useState(0);
  const [docCount, setDocCount] = useState<number>(0);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [promptVersion, setPromptVersion] = useState<number>(1);
  const [availableVersions, setAvailableVersions] = useState<number[]>([1,2]);
  const [promptName, setPromptName] = useState<string>("rag");
  const [availableNames, setAvailableNames] = useState<string[]>(["rag"]);
  const [topK, setTopK] = useState<number>(5); // NEW: default 5
  const [model, setModel] = useState<string | null>(null); // NEW
  const [interrupted, setInterrupted] = useState(false);
  const [promptJsonStr, setPromptJsonStr] = useState<string>("");            // NEW
  const [promptJsonErr, setPromptJsonErr] = useState<string>("");            // NEW
  const [savingPrompt, setSavingPrompt] = useState<boolean>(false);          // NEW
  const pollRef = useRef<NodeJS.Timeout | null>(null);
  const abortRef = useRef<{ abort: () => void } | null>(null);
  const answerEndRef = useRef<HTMLDivElement | null>(null);

  async function refreshDocs() {
    try {
      const r = await documentsSummary();
      setDocCount(r.ok_count);
    } catch {
      setDocCount(0);
    }
  }

  useEffect(() => {
    refreshDocs();
    async function loadPrompts() {            // CHANGED
      try {
        const rows = await listPrompts();
        const names = Array.from(new Set(rows.map(r => r.name))).sort();
        setAvailableNames(names.length ? names : ["rag"]);
        const matching = rows.filter(r => r.name === (promptName || "rag"));
        const versions = matching.map(r => r.version).sort((a,b)=>a-b);
        setAvailableVersions(versions.length ? versions : [1]);
        if (typeof window !== "undefined") {
          const savedVer = parseInt(localStorage.getItem("chat_prompt_version") || "", 10);
          if (versions.includes(savedVer)) setPromptVersion(savedVer);
          else setPromptVersion(versions[versions.length - 1] || 1);
        }
      } catch {
        setAvailableVersions([1]);
      }
    }
    loadPrompts();
    if (typeof window !== "undefined") localStorage.setItem("chat_prompt_name", "rag");
    const poll = setInterval(refreshDocs, 12000);
    pollRef.current = poll;
    return () => { clearInterval(poll); };
  }, []);

  // When user changes version, persist it
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("chat_prompt_version", String(promptVersion));
    }
  }, [promptVersion]);

  // Load current prompt JSON whenever name/version changes (NEW)
  useEffect(() => {
    async function loadPromptJson() {
      setPromptJsonErr("");
      try {
        const p = await getPrompt(promptName || "rag", promptVersion || 1);
        const obj = {
          name: p.name,
          version: p.version,
          system: p.system,
          user_template: p.user_template,
          params: p.params || {}
        };
        setPromptJsonStr(JSON.stringify(obj, null, 2));
      } catch (e: any) {
        setPromptJsonStr("");
        setPromptJsonErr(e?.message || "Failed to load prompt JSON");
      }
    }
    if (promptName && promptVersion) loadPromptJson();
  }, [promptName, promptVersion]);

  // Auto-scroll to bottom when streaming
  useEffect(() => {
    if (streaming && answerEndRef.current) {
      answerEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [answer, streaming]);

  async function ask() {
    if (docCount === 0 || !query.trim()) return;
    reset();
    setLoading(true);
    setErrorMsg("");
    try {
      const res = await chatComplete({
        query,
        retriever: { top_k: topK },
        prompt: { name: "rag", version: promptVersion } // CHANGED: normalized name
      });
      setRunId(res.run_id);
      setAnswer(res.answer || "");
      setLatencyMs(res.latency_ms || 0);
      setTokensIn(res.tokens.in);
      setTokensOut(res.tokens.out);
      setModel(res.model || null); // NEW
      const mapped = (res.context || []).map((c: any) => ({   // CHANGED
        chunk_id: c.chunk_id || c.document_id,                // prefer backend chunk_id
        rank: c.rank || c.score ? c.rank : undefined,         // if provided
        snippet: c.snippet,
        document_id: c.document_id,
        filename: c.filename || undefined
      }));
      setCitations(mapped);
      if (res.degraded_context) {
        setErrorMsg("⚠️ Context temporarily unavailable (rate limited) — answered without document grounding.");
      }
    } catch (e: any) {
      setErrorMsg(e?.message || "Chat failed");
    } finally {
      setLoading(false);
    }
  }

  // Save edited JSON as a new prompt version (NEW)
  async function saveEditedPromptAsNewVersion() {
    setPromptJsonErr("");
    setSavingPrompt(true);
    try {
      const parsed = JSON.parse(promptJsonStr || "{}");
      if (!parsed || typeof parsed !== "object") throw new Error("Invalid JSON");
      const sys = parsed.system;
      const usr = parsed.user_template;
      const params = parsed.params || {};
      if (typeof sys !== "string" || typeof usr !== "string") {
        throw new Error("JSON must include 'system' (string) and 'user_template' (string)");
      }
      const res = await createPromptVersion({
        name: promptName || "rag",
        source_version: promptVersion,
        system: sys,
        user_template: usr,
        params
      });
      // Refresh versions and select the new one
      const rows = await listPrompts();
      const matching = rows.filter(r => r.name === (promptName || "rag"));
      const versions = matching.map(r => r.version).sort((a,b)=>a-b);
      setAvailableVersions(versions.length ? versions : [1]);
      setPromptVersion(res.version);
    } catch (e: any) {
      setPromptJsonErr(e?.message || "Failed to save new version");
    } finally {
      setSavingPrompt(false);
    }
  }

  // Reset editor to current server prompt (NEW)
  async function resetPromptEditor() {
    setPromptJsonErr("");
    try {
      const p = await getPrompt(promptName || "rag", promptVersion || 1);
      const obj = {
        name: p.name,
        version: p.version,
        system: p.system,
        user_template: p.user_template,
        params: p.params || {}
      };
      setPromptJsonStr(JSON.stringify(obj, null, 2));
    } catch (e: any) {
      setPromptJsonErr(e?.message || "Failed to reload prompt");
    }
  }

  // Download current editor JSON (NEW)
  function downloadPromptJson() {
    try {
      const blob = new Blob([promptJsonStr || "{}"], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${promptName || "rag"}_v${promptVersion}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {}
  }

  function reset() {
    setAnswer("");
    setCitations([]);
    setTokensOut(0);
    setTokensIn  (0);
    setRunId(undefined);
    setLatencyMs(0);
    setModel(null); // NEW
    setInterrupted(false);
  }

  function startStream() {
    if (docCount === 0 || !query.trim() || streaming) return;
    reset();
    setErrorMsg("");
    setStreaming(true);
    const startedAt = performance.now();
    
    abortRef.current = streamChat(
      {
        query,
        retriever: { top_k: topK }, // CHANGED
        prompt: { name: "rag", version: promptVersion } // CHANGED
      },
      (tok, meta) => {
        setAnswer(a => a + tok);
        if (meta?.stats?.tokens_out) setTokensOut(meta.stats.tokens_out);
      },
      (final) => {
        setStreaming(false);
        setRunId(final.run_id);
        setLatencyMs(final.latency_ms || Math.round(performance.now() - startedAt));
        setTokensIn(final.tokens?.in || 0);
        setTokensOut(final.tokens?.out || 0);
        setModel(final.model || null); // NEW
        if (final.degraded_context) {
          setErrorMsg("⚠️ Context degraded (rate limited).");
        }
        if (final.interrupted) {
          setInterrupted(true);
          setErrorMsg("⚠️ Generation interrupted.");
        }
        if (final.citations) {
          setCitations(final.citations);
        }
      },
      (err) => {
        setStreaming(false);
        setErrorMsg(err.message || "Stream failed");
      },
      (citation) => {
        setCitations(cs => {
          if (cs.find(x => (x.marker ?? x.rank) === (citation.marker ?? citation.rank))) return cs;
          return [...cs, citation];
        });
      }
    );
  }

  function cancelStream() {
    if (abortRef.current) {
      abortRef.current.abort();
      setStreaming(false);
      setInterrupted(true);
      setErrorMsg("⚠️ Generation interrupted by user.");
    }
  }

  function copyPartialAnswer() {
    if (answer) {
      navigator.clipboard.writeText(answer);
      setErrorMsg("✓ Partial answer copied to clipboard.");
      setTimeout(() => setErrorMsg(""), 2000);
    }
  }

  // Render answer with hoverable inline [#] markers
  function renderAnswerWithCitations(txt: string) {
    if (!txt) return txt;
    const parts: React.ReactNode[] = [];
    const re = /\[(\d+)\]/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = re.exec(txt)) !== null) {
      const idx = m.index;
      if (idx > last) parts.push(txt.slice(last, idx));
      const num = parseInt(m[1], 10);
      const meta = citations.find(c => (c.marker ?? c.rank) === num);
      parts.push(
        <sup
          key={`mark-${idx}-${num}`}
          title={`${meta?.filename ? meta.filename + " — " : ""}${meta?.snippet || "Source"}`}  // CHANGED: add filename
          style={{ cursor: meta ? "help" : "default" }}
          className="text-indigo-400 hover:text-indigo-300"
        >
          [{num}]
        </sup>
      );
      last = re.lastIndex;
    }
    if (last < txt.length) parts.push(txt.slice(last));
    return parts;
  }

  return (
    <div className="grid lg:grid-cols-3 gap-6">
      {/* Left: controls + answer */}
      <div className="lg:col-span-2 space-y-4">
        <div className="rounded border border-neutral-800 p-4 bg-neutral-900 space-y-3">
          <div className="flex flex-wrap gap-3 items-end">
            <div className="flex flex-col">
              <label className="text-neutral-400 text-xs mb-1">Prompt</label>
              <select
                value={promptName}
                onChange={e => setPromptName(e.target.value)}
                className="bg-neutral-800 rounded px-2 py-2 text-xs"
                disabled={loading || streaming}
              >
                {/* Single family 'rag' (could extend later) */}
                {availableNames.map(name => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </div>
            <div className="flex flex-col">
              <label className="text-neutral-400 text-xs mb-1">Version</label>
              <select
                value={promptVersion}
                onChange={e => setPromptVersion(parseInt(e.target.value))}
                className="bg-neutral-800 rounded px-2 py-2 text-xs"
                disabled={loading || streaming}
              >
                {availableVersions.map(v => (
                  <option key={v} value={v}>v{v}</option>
                ))}
              </select>
            </div>
            <div className="flex flex-col">
              <label className="text-neutral-400 text-xs mb-1">Retriever top-k</label>
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={e => setTopK(Math.max(1, Math.min(20, parseInt(e.target.value || "5"))))}
                className="bg-neutral-800 rounded px-2 py-2 text-xs w-20"
                disabled={loading || streaming}
              />
            </div>
            <button
              onClick={ask}
              disabled={!query || loading || docCount === 0 || streaming}
              className="bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-indigo-500 transition px-4 py-2 rounded text-sm font-medium ml-auto"
            >
              {loading ? "Thinking..." : "Ask (Sync)"}
            </button>
            <button
              onClick={startStream}
              disabled={!query || loading || docCount === 0 || streaming}
              className="bg-green-600 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-green-500 transition px-4 py-2 rounded text-sm font-medium"
            >
              {streaming ? "Streaming..." : "Stream"}
            </button>
            {streaming && (
              <button
                onClick={cancelStream}
                className="bg-neutral-700 hover:bg-neutral-600 px-3 py-2 rounded text-sm"
              >
                Cancel
              </button>
            )}
          </div>
          <textarea
            className="w-full bg-neutral-800 rounded px-3 py-2 text-sm outline-none focus:ring focus:ring-indigo-600 min-h-[80px] resize-y"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Ask a grounded question..."
            disabled={loading || streaming}
          />
          {docCount === 0 && (
            <div className="text-xs text-amber-400 space-y-1">
              <p>No ready documents found.</p>
              <p className="text-neutral-400">
                Go to <code>Datasets</code> page to ingest PDFs, then re-check.
              </p>
              <button
                onClick={refreshDocs}
                className="mt-1 px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-[11px]"
              >
                Re-check documents
              </button>
            </div>
          )}
          {errorMsg && (
            <div className="text-xs border border-amber-700 bg-amber-900/30 rounded px-3 py-2 text-amber-200">
              {errorMsg}
            </div>
          )}
          {interrupted && answer && (
            <button
              onClick={copyPartialAnswer}
              className="text-xs px-2 py-1 rounded bg-neutral-700 hover:bg-neutral-600"
            >
              Copy partial answer
            </button>
          )}
          {runId && (
            <div className="text-[11px] text-neutral-400 flex flex-wrap gap-4">
              <span>run: <code className="font-mono">{runId.slice(0, 8)}</code></span>
              <span>latency: {latencyMs}ms</span>
              <span>tokens: {tokensIn} in / {tokensOut} out</span>
              {model && <span>model: {model}</span>}
            </div>
          )}
        </div>

        <div className="rounded border border-neutral-800 bg-neutral-900 p-4">
          <h3 className="text-sm font-semibold mb-2">Answer</h3>
          {streaming && (
            <div className="text-[11px] text-neutral-400 mb-2 flex items-center gap-3">
              <span className="animate-pulse">●</span>
              <span>streaming…</span>
              <span>tokens out: {tokensOut}</span>
            </div>
          )}
          <div className="min-h-[240px] max-h-[60vh] overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed">
            {streaming && !answer && (
              <div className="text-neutral-500 italic">● typing...</div>
            )}
            {answer ? renderAnswerWithCitations(answer) : (!loading && !streaming ? "Ask a question about your ingested documents." : "")}
            <div ref={answerEndRef} />
          </div>
        </div>
      </div>

      {/* Right: Context used (Top-K) */}
      <div className="space-y-4">
        <div className="rounded border border-neutral-800 bg-neutral-900 p-4">
          <h3 className="text-sm font-semibold mb-3">Context used (Top‑K)</h3>
          {/* NEW: subtle banner if no citations emitted */}
          {answer && !/\[\d+\]/.test(answer) && citations.length > 0 && (
            <div className="text-[11px] text-neutral-400 mb-2">
              No citations emitted—showing context used.
            </div>
          )}
          <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
            {
              // Show only cited items ordered by first appearance; fallback to all if no markers.
              (React.useMemo(() => {
                const matches = Array.from(answer.matchAll(/\[(\d+)\]/g)).map(m => parseInt(m[1], 10));
                const uniqueOrder: number[] = [];
                for (const n of matches) if (!uniqueOrder.includes(n)) uniqueOrder.push(n);
                if (uniqueOrder.length === 0) return [...citations];
                const orderIdx = new Map(uniqueOrder.map((n, i) => [n, i]));
                return citations
                  .filter(c => uniqueOrder.includes((c.marker ?? c.rank ?? 0)))
                  .sort((a, b) => (orderIdx.get(a.marker ?? a.rank ?? 0) ?? 9999) - (orderIdx.get(b.marker ?? b.rank ?? 0) ?? 9999));
              }, [citations, answer])).map((c, idx) => (
                <div
                  key={c.chunk_id ?? c.marker ?? c.document_id ?? idx}
                  className="text-xs border border-neutral-700 rounded p-3 bg-neutral-800/40 hover:bg-neutral-800 transition"
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-semibold text-indigo-400">
                      #{c.rank ?? c.marker ?? (idx + 1)}
                    </span>
                    {/* filename chip (optional) */}
                    {c.filename && (
                      <span className="text-neutral-400 text-[10px] truncate max-w-[50%]">{c.filename}</span>
                    )}
                  </div>
                  <div className="text-neutral-300 leading-relaxed">{c.snippet}</div>
                </div>
              ))
            }
            {citations.length === 0 && <div className="text-xs text-neutral-500">No context yet.</div>}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatUI;
