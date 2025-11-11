"use client";
import React, { useEffect, useMemo, useState } from "react";
import { ingestDocs, listDocuments, documentFileUrl, deleteAllDocuments, retrieveChunks } from "../lib/api";

type DocFile = {
  id: string;
  filename: string;
  status: string;
  size: number;
  created_at: string;
  chunks?: number;
};

type RetrievalResult = {
  chunk_id: string;
  document_id: string;
  filename: string | null;
  score: number;
  snippet: string;
};

const fmtSize = (bytes: number) => {
  if (!bytes) return "0 B";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(2)} MB`;
};

const isoParts = (isoString: string) => {
  const d = new Date(isoString);
  return {
    date: d.toLocaleDateString(),
    time: d.toLocaleTimeString([], { hour: '2-digit', minute:'2-digit' }),
  };
};

const DatasetsPage: React.FC = () => {
  const [docs, setDocs] = useState<DocFile[]>([]);
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [toast, setToast] = useState("");
  
  // Retrieval probe state
  const [showRetrievalModal, setShowRetrievalModal] = useState(false);
  const [retrievalQuery, setRetrievalQuery] = useState("");
  const [retrievalK, setRetrievalK] = useState(8);
  const [retrievalResults, setRetrievalResults] = useState<RetrievalResult[]>([]);
  const [retrievalBusy, setRetrievalBusy] = useState(false);
  const [retrievalError, setRetrievalError] = useState("");

  async function load() {
    setErrorMsg("");
    const rows = await listDocuments();
    setDocs(rows);
  }

  useEffect(() => {
    load();
  }, []);

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    setErrorMsg("");
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    setBusy(true);
    try {
      const pdfs = files.filter(f => f.name.toLowerCase().endsWith(".pdf"));
      if (!pdfs.length) throw new Error("Only .pdf files are accepted");
      const res = await ingestDocs(pdfs);
      setToast(`${res.counts.docs} docs ingested, ${res.counts.chunks} chunks.`);
      setTimeout(() => setToast(""), 2500);
    } catch (ex: any) {
      const detail = ex?.data || ex?.detail;
      let msg = ex?.message || "Upload failed";
      if (Array.isArray(detail)) msg = detail.map((d: any) => d?.msg || JSON.stringify(d)).join("; ");
      else if (detail && typeof detail === "object") msg = detail.msg || JSON.stringify(detail);
      setErrorMsg(String(msg));
    } finally {
      await load();
      setBusy(false);
      if (e.target) e.target.value = "";
    }
  }

  async function handleDeleteAll() {
    if (!confirm("Delete ALL documents and their PDF files under datasets/docs? This action is irreversible.")) return;
    setErrorMsg("");
    setBusy(true);
    try {
      const res = await deleteAllDocuments(true);
      setToast(`Deleted ${res.deleted_documents} documents, removed ${res.removed_files} files.`);
      setTimeout(() => setToast(""), 3000);
    } catch (ex: any) {
      setErrorMsg(ex?.message || "Delete all failed");
    } finally {
      await load();
      setBusy(false);
    }
  }

  async function handleTestRetrieval() {
    if (!retrievalQuery.trim()) {
      setRetrievalError("Please enter a query");
      return;
    }
    setRetrievalError("");
    setRetrievalBusy(true);
    try {
      const res = await retrieveChunks(retrievalQuery, retrievalK);
      setRetrievalResults(res.results);
    } catch (ex: any) {
      setRetrievalError(ex?.message || "Retrieval failed");
    } finally {
      setRetrievalBusy(false);
    }
  }

  const rows = useMemo(() => docs, [docs]);

  return (
    <div className="space-y-6">
      <div className="rounded border border-neutral-800 bg-neutral-900 p-4 space-y-3">
        <h3 className="text-sm font-semibold">Ingest PDFs</h3>
        <input
          type="file"
          accept=".pdf"
          multiple
          disabled={busy}
          onChange={onUpload}
          className="text-xs"
        />
        {toast && <div className="text-[11px] text-green-400">{toast}</div>}
        {errorMsg && (
          <div className="text-[11px] text-red-300 border border-red-700 bg-red-900/30 rounded px-2 py-1">
            {errorMsg}
          </div>
        )}
        <div className="flex gap-2">
          <button className="px-3 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-xs" onClick={load} disabled={busy}>Refresh</button>
          <button
            className="px-3 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-xs"
            onClick={() => setShowRetrievalModal(true)}
            disabled={busy || docs.length === 0}
          >
            Try Retrieval
          </button>
          <button
            className="px-3 py-1 rounded bg-red-700 hover:bg-red-600 text-xs"
            onClick={handleDeleteAll}
            disabled={busy}
            title="Deletes all documents and PDF files (irreversible)"
          >
            Delete all docs
          </button>
        </div>
      </div>

      <div className="rounded border border-neutral-800 bg-neutral-900 p-4 space-y-3">
        <h3 className="text-sm font-semibold">Documents</h3>
        <table className="w-full text-[12px]">
          <thead className="text-neutral-400">
            <tr>
              <th className="text-left p-2">File</th>
              <th className="text-left p-2">UUID</th>
              <th className="text-left p-2">Size</th>
              <th className="text-left p-2">Status</th>
              <th className="text-left p-2">Chunks</th>
              <th className="text-left p-2">Date</th>
              <th className="text-left p-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((d) => {
              const parts = isoParts(d.created_at);
              const chunks = (d.chunks ?? 0);
              return (
                <tr key={d.id} className="border-t border-neutral-800">
                  <td className="p-2">{d.filename}</td>
                  <td className="p-2"><code className="text-[10px]">{d.id.slice(0, 8)}</code></td>
                  <td className="p-2">{fmtSize(d.size)}</td>
                  <td className="p-2">{d.status}</td>
                  <td className="p-2">{chunks}</td>
                  <td className="p-2">{parts.date}</td>
                  <td className="p-2">
                    <a
                      href={documentFileUrl(d.id)}
                      target="_blank"
                      rel="noreferrer"
                      className="text-indigo-400 hover:underline"
                    >
                      Preview
                    </a>
                  </td>
                </tr>
              );
            })}
            {!rows.length && (
              <tr>
                <td colSpan={7} className="p-3 text-neutral-500 text-xs">
                  No documents yet. Upload PDFs above.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Retrieval Modal */}
      {showRetrievalModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-neutral-900 border border-neutral-700 rounded-lg p-6 max-w-3xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Test Retrieval</h3>
              <button
                onClick={() => setShowRetrievalModal(false)}
                className="text-neutral-400 hover:text-white"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="text-xs text-neutral-400 block mb-1">Query</label>
                <input
                  type="text"
                  value={retrievalQuery}
                  onChange={(e) => setRetrievalQuery(e.target.value)}
                  className="w-full bg-neutral-800 border border-neutral-700 rounded px-3 py-2 text-sm"
                  placeholder="Enter your search query..."
                />
              </div>
              
              <div>
                <label className="text-xs text-neutral-400 block mb-1">Top K</label>
                <input
                  type="number"
                  value={retrievalK}
                  onChange={(e) => setRetrievalK(Math.max(1, Math.min(50, parseInt(e.target.value) || 8)))}
                  className="w-32 bg-neutral-800 border border-neutral-700 rounded px-3 py-2 text-sm"
                  min="1"
                  max="50"
                />
              </div>

              {retrievalError && (
                <div className="text-xs text-red-300 border border-red-700 bg-red-900/30 rounded px-2 py-1">
                  {retrievalError}
                </div>
              )}

              <button
                onClick={handleTestRetrieval}
                disabled={retrievalBusy}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm disabled:opacity-50"
              >
                {retrievalBusy ? "Searching..." : "Search"}
              </button>

              {retrievalResults.length > 0 && (
                <div className="space-y-3 mt-4">
                  <h4 className="text-sm font-semibold text-neutral-300">Results ({retrievalResults.length})</h4>
                  {retrievalResults.map((r, idx) => (
                    <div key={r.chunk_id} className="bg-neutral-800 border border-neutral-700 rounded p-3 space-y-2">
                      <div className="flex justify-between items-start">
                        <div className="text-xs text-neutral-400">
                          <span className="font-mono">#{idx + 1}</span>
                          <span className="mx-2">•</span>
                          <span>{r.filename || "Unknown"}</span>
                        </div>
                        <span className="text-xs font-mono text-green-400">
                          {(r.score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-sm text-neutral-200">{r.snippet}</p>
                      <div className="text-[10px] text-neutral-500 font-mono">
                        chunk: {r.chunk_id.slice(0, 8)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DatasetsPage;
