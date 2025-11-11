"use client";
import React, { useEffect, useState } from "react";
import { getRun, listRuns, metricsSummary, getRunsByGroup } from "../lib/api";
import { RunsTable } from "./RunsTable";

const PAGE_SIZE = 20; // NEW

const RunsPage: React.FC = () => {
  const [runs, setRuns] = useState<any[]>([]);
  const [sel, setSel] = useState<any|null>(null);
  const [summary, setSummary] = useState<any|null>(null);
  const [abGroup, setAbGroup] = useState<any|null>(null);         // NEW
  const [showCompareModal, setShowCompareModal] = useState(false); // NEW
  const [compareItem, setCompareItem] = useState<any|null>(null);  // NEW
  
  // Filters
  const [filterKind, setFilterKind] = useState<string>("");
  const [filterModel, setFilterModel] = useState<string>("");
  const [filterPrompt, setFilterPrompt] = useState<string>("");
  const [showFilters, setShowFilters] = useState(false);
  const [page, setPage] = useState(1);                     // NEW

  async function load() {
    try {
      const filters: Record<string, any> = {};
      if (filterKind) filters.kind = filterKind;
      if (filterModel) filters.model = filterModel;
      if (filterPrompt) filters.prompt_name = filterPrompt;
      const list = await listRuns(filters, { limit: PAGE_SIZE, offset: (page - 1) * PAGE_SIZE }); // CHANGED
      setRuns(list);
    } catch {
      setRuns([]);
    }
    try {
      const s = await metricsSummary();
      setSummary(s);
    } catch {
      setSummary(null);
    }
  }
  
  // Reset to first page when filters change
  useEffect(() => { setPage(1); }, [filterKind, filterModel, filterPrompt]); // NEW

  useEffect(()=>{ load(); }, [filterKind, filterModel, filterPrompt, page]); // CHANGED (added page)

  async function open(id: string) {
    const detail = await getRun(id);
    setSel(detail);
    // If A/B run, load group runs
    if (detail?.run?.meta?.group_id) {
      try {
        const grp = await getRunsByGroup(detail.run.meta.group_id);
        setAbGroup(grp);
      } catch {
        setAbGroup(null);
      }
    } else {
      setAbGroup(null);
    }
  }

  function clearFilters() {
    setFilterKind("");
    setFilterModel("");
    setFilterPrompt("");
  }

  function snippet(txt: string, n = 160) {  // NEW
    if (!txt) return "";
    return txt.length > n ? txt.slice(0, n) + "…" : txt;
  }

  function openCompare(q: string) {         // NEW
    if (!abGroup) return;
    const runs = abGroup.runs;
    const a = runs.find((r: any) => r.meta?.variant === "A");
    const b = runs.find((r: any) => r.meta?.variant === "B");
    const ja = a?.judgements?.find((j: any) => j.query === q);
    const jb = b?.judgements?.find((j: any) => j.query === q);
    setCompareItem({ query: q, a: ja, b: jb });
    setShowCompareModal(true);
  }

  // Check if run is A/B type
  const isAB = sel?.run?.kind === "ab_eval";
  const abGroupId = sel?.run?.meta?.group_id; // CHANGED

  return (
    <div className="grid lg:grid-cols-3 gap-6">
      <div className="lg:col-span-2 space-y-6">
        <div className="rounded border border-neutral-800 bg-neutral-900 p-4 space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-semibold">Runs</h3>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="text-xs px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700"
            >
              {showFilters ? "Hide Filters" : "Show Filters"}
            </button>
          </div>
          
          {showFilters && (
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div>
                <label className="block text-neutral-400 mb-1">Kind</label>
                <select
                  value={filterKind}
                  onChange={(e) => setFilterKind(e.target.value)}
                  className="w-full bg-neutral-800 rounded px-2 py-1"
                >
                  <option value="">All</option>
                  <option value="chat">Chat</option>
                  <option value="eval">Eval</option>
                  <option value="ab_eval">A/B Eval</option>
                </select>
              </div>
              <div>
                <label className="block text-neutral-400 mb-1">Model</label>
                <input
                  type="text"
                  value={filterModel}
                  onChange={(e) => setFilterModel(e.target.value)}
                  placeholder="e.g. mistral-large"
                  className="w-full bg-neutral-800 rounded px-2 py-1"
                />
              </div>
              <div>
                <label className="block text-neutral-400 mb-1">Prompt</label>
                <input
                  type="text"
                  value={filterPrompt}
                  onChange={(e) => setFilterPrompt(e.target.value)}
                  placeholder="e.g. rag"
                  className="w-full bg-neutral-800 rounded px-2 py-1"
                />
              </div>
              <button
                onClick={clearFilters}
                className="col-span-3 px-2 py-1 rounded bg-neutral-700 hover:bg-neutral-600 text-xs"
              >
                Clear Filters
              </button>
            </div>
          )}
          
          {summary && (
            <div className="text-[11px] text-neutral-400 flex flex-wrap gap-4">
              <span>p50 {summary.p50_latency_ms}ms</span>
              <span>p95 {summary.p95_latency_ms}ms</span>
              <span>avg in {summary.avg_tokens_in}</span>
              <span>avg out {summary.avg_tokens_out}</span>
              <span>relevance {summary.relevance_avg?.toFixed?.(2)}</span>
              <span>grounded {summary.groundedness_avg?.toFixed?.(2)}</span>
            </div>
          )}
          <RunsTable runs={runs} onSelect={open}/>
          {/* NEW: Pagination controls */}
          <div className="flex items-center justify-between pt-2 border-t border-neutral-800 text-[11px]">
            <div>
              Page {page} (showing {runs.length} of {runs.length < PAGE_SIZE ? runs.length : PAGE_SIZE} rows)
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                className="px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Prev
              </button>
              <button
                onClick={() => {
                  if (runs.length === PAGE_SIZE) setPage(p => p + 1);
                }}
                disabled={runs.length < PAGE_SIZE}
                className="px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Next
              </button>
            </div>
          </div>
        </div>

      </div>


      {/* Compare Modal (side-by-side) NEW */}
      {showCompareModal && compareItem && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <div className="bg-neutral-900 border border-neutral-700 rounded p-4 w-[90%] max-w-5xl space-y-3">
            <div className="flex justify-between items-center">
              <h4 className="text-sm font-semibold">Query: {compareItem.query}</h4>
              <button onClick={()=>setShowCompareModal(false)} className="text-xs bg-neutral-700 hover:bg-neutral-600 px-2 py-1 rounded">Close</button>
            </div>
            <div className="grid md:grid-cols-2 gap-4 text-[11px]">
              <div>
                <div className="font-semibold mb-1">Variant A</div>
                <div className="bg-neutral-800 rounded p-2 max-h-64 overflow-auto whitespace-pre-wrap">{compareItem.a?.answer || "—"}</div>
              </div>
              <div>
                <div className="font-semibold mb-1">Variant B</div>
                <div className="bg-neutral-800 rounded p-2 max-h-64 overflow-auto whitespace-pre-wrap">{compareItem.b?.answer || "—"}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RunsPage;
