"use client";
import React, { useEffect, useState } from "react";
import { listPrompts, getPrompt, createPromptVersion, diffPrompts, startABEval } from "../lib/api";

const PromptsPage: React.FC = () => {
  const [rows, setRows] = useState<any[]>([]);
  const [name, setName] = useState("rag");
  const [sourceVersion, setSourceVersion] = useState<number|undefined>();
  const [system, setSystem] = useState("");
  const [userTemplate, setUserTemplate] = useState("");
  const [params, setParams] = useState("{}");
  const [diffA, setDiffA] = useState<number|undefined>();
  const [diffB, setDiffB] = useState<number|undefined>();
  const [diffData, setDiffData] = useState<any|null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");

  const [abDatasetId, setAbDatasetId] = useState("");     // NEW
  const [abVerA, setAbVerA] = useState<number>(1);        // NEW
  const [abVerB, setAbVerB] = useState<number>(1);        // NEW
  const [abTopK, setAbTopK] = useState<number>(5);        // NEW
  const [abBusy, setAbBusy] = useState(false);            // NEW

  async function load() {
    try {
      const r = await listPrompts();
      setRows(r);
      const rag = r.filter(p => p.name === name).sort((a,b)=>a.version-b.version);
      if (rag.length && !sourceVersion) {
        setSourceVersion(rag[rag.length - 1].version);
        setSystem(rag[rag.length - 1].system);
        setUserTemplate(rag[rag.length - 1].user_template);
        setParams(JSON.stringify(rag[rag.length - 1].params || {}, null, 2));
      }
    } catch { setRows([]); }
  }

  useEffect(()=>{ load(); }, []);

  useEffect(() => {
    if (diffA !== undefined && diffB !== undefined && diffA !== diffB) {
      diffPrompts(name, diffA, diffB).then(setDiffData).catch(()=>setDiffData(null));
    } else setDiffData(null);
  }, [diffA, diffB, name]);

  useEffect(()=> {
    const rag = rows.filter(r=>r.name===name).map(r=>r.version).sort((a,b)=>a-b);
    if (rag.length) {
      setAbVerA(rag[0]);
      setAbVerB(rag[rag.length-1]);
    }
  }, [rows, name]); // NEW

  async function loadSource(version: number) {
    try {
      const p = await getPrompt(name, version);
      setSourceVersion(version);
      setSystem(p.system);
      setUserTemplate(p.user_template);
      setParams(JSON.stringify(p.params || {}, null, 2));
    } catch {}
  }

  async function save() {
    setSaving(true); setError(""); setToast("");
    let parsedParams = {};
    try { parsedParams = JSON.parse(params || "{}"); } catch (e:any) { setError("Params JSON invalid"); setSaving(false); return; }
    try {
      const res = await createPromptVersion({
        name,
        source_version: sourceVersion,
        system,
        user_template: userTemplate,
        params: parsedParams
      });
      setToast(`Created ${res.name}@${res.version}`);
      setTimeout(()=>setToast(""), 2500);
      setSourceVersion(res.version);
      load();
    } catch (ex:any) {
      setError(ex?.message || "Create failed");
    } finally {
      setSaving(false);
    }
  }

  function applyHybridExampleV2() {
    setSystem(
`You are a precise assistant. Generate exactly three sentences, one paragraph.
Use ONLY provided context; ban external knowledge.
Each sentence must end with one or more citations like [1] or [2][3].
Every claim about systems, benefits, or retrieval must be cited.
If unsupported: write Not in context.
Do NOT add extra sentences.`
    );
  }

  async function runAB() {              // NEW
    setError(""); setToast("");
    if (!abDatasetId.trim()) { setError("Dataset ID required"); return; }
    setAbBusy(true);
    try {
      const res = await startABEval({
        dataset_id: abDatasetId.trim(),
        variant_a: { prompt: { name, version: abVerA }, retriever: { top_k: abTopK } },
        variant_b: { prompt: { name, version: abVerB }, retriever: { top_k: abTopK } }
      });
      setToast(`A/B started group=${res.group_id.slice(0,8)} (runs: ${res.run_a_id.slice(0,8)}, ${res.run_b_id.slice(0,8)})`);
      setTimeout(()=>setToast(""), 4000);
    } catch (e:any) {
      setError(e?.message || "A/B run failed");
    } finally {
      setAbBusy(false);
    }
  } // NEW

  return (
    <div className="space-y-6">
      <div className="rounded border border-neutral-800 bg-neutral-900 p-4 space-y-4">
        <h3 className="text-sm font-semibold">Prompt Versions ({rows.filter(r=>r.name===name).length})</h3>
        <div className="flex flex-wrap gap-2 text-[11px]">
          {rows.filter(r=>r.name===name).sort((a,b)=>a.version-b.version).map(r=>
            <button key={r.version}
              onClick={()=>loadSource(r.version)}
              className={`px-2 py-1 rounded ${r.version===sourceVersion?'bg-indigo-600':'bg-neutral-800 hover:bg-neutral-700'}`}>
              v{r.version}
            </button>
          )}
        </div>
        <div className="grid md:grid-cols-2 gap-4 text-xs">
          <div className="space-y-2">
            <label className="font-semibold">System</label>
            <textarea value={system} onChange={e=>setSystem(e.target.value)}
              className="w-full h-48 bg-neutral-800 rounded p-2 font-mono text-[11px]"/>
            <button onClick={applyHybridExampleV2}
              className="px-2 py-1 rounded bg-neutral-700 hover:bg-neutral-600 text-[11px]">
              Apply 3‑sentence Hybrid Template
            </button>
          </div>
          <div className="space-y-2">
            <label className="font-semibold">User Template</label>
            <textarea value={userTemplate} onChange={e=>setUserTemplate(e.target.value)}
              className="w-full h-48 bg-neutral-800 rounded p-2 font-mono text-[11px]"/>
          </div>
          <div className="space-y-2 md:col-span-2">
            <label className="font-semibold">Params (JSON)</label>
            <textarea value={params} onChange={e=>setParams(e.target.value)}
              className="w-full h-32 bg-neutral-800 rounded p-2 font-mono text-[11px]"/>
          </div>
        </div>
        {error && <div className="text-xs text-red-300">{error}</div>}
        {toast && <div className="text-xs text-green-400">{toast}</div>}
        <button onClick={save} disabled={saving}
          className="px-3 py-2 rounded bg-indigo-600 hover:bg-indigo-500 text-xs disabled:opacity-50">
          {saving? "Saving..." : "Create New Version"}
        </button>
      </div>

      <div className="rounded border border-neutral-800 bg-neutral-900 p-4 space-y-3">
        <h3 className="text-sm font-semibold">Diff</h3>
        <div className="flex gap-2 text-[11px]">
          <select value={diffA ?? ""} onChange={e=>setDiffA(e.target.value?parseInt(e.target.value):undefined)}
            className="bg-neutral-800 rounded px-2 py-1">
            <option value="">A</option>
            {rows.filter(r=>r.name===name).map(r=><option key={r.version} value={r.version}>{r.version}</option>)}
          </select>
          <select value={diffB ?? ""} onChange={e=>setDiffB(e.target.value?parseInt(e.target.value):undefined)}
            className="bg-neutral-800 rounded px-2 py-1">
            <option value="">B</option>
            {rows.filter(r=>r.name===name).map(r=><option key={r.version} value={r.version}>{r.version}</option>)}
          </select>
        </div>
        {!diffData && <div className="text-xs text-neutral-500">Select two versions to diff.</div>}
        {diffData && (
          <div className="grid md:grid-cols-2 gap-4 text-[10px] font-mono">
            <div>
              <div className="text-neutral-400 mb-1">System Diff</div>
              <pre className="bg-neutral-800 rounded p-2 max-h-64 overflow-auto">
{diffData.system_diff.join("\n")}
              </pre>
            </div>
            <div>
              <div className="text-neutral-400 mb-1">User Template Diff</div>
              <pre className="bg-neutral-800 rounded p-2 max-h-64 overflow-auto">
{diffData.user_template_diff.join("\n")}
              </pre>
            </div>
            <div className="md:col-span-2 space-y-1">
              <div className="text-neutral-400">Params Changes</div>
              <pre className="bg-neutral-800 rounded p-2 max-h-40 overflow-auto">
Added: {JSON.stringify(diffData.params_added)}
Removed: {JSON.stringify(diffData.params_removed)}
Changed: {JSON.stringify(diffData.params_changed)}
              </pre>
            </div>
          </div>
        )}
      </div>

      <div className="rounded border border-neutral-800 bg-neutral-900 p-4 space-y-3">
        <h3 className="text-sm font-semibold">Run A/B (Prompt / Retriever)</h3>
        <div className="grid md:grid-cols-4 gap-2 text-[11px]">
          <div className="flex flex-col">
            <label className="text-neutral-400 mb-1">Dataset ID</label>
            <input
              value={abDatasetId}
              onChange={e=>setAbDatasetId(e.target.value)}
              placeholder="dataset uuid"
              className="bg-neutral-800 rounded px-2 py-1"
            />
          </div>
          <div className="flex flex-col">
            <label className="text-neutral-400 mb-1">Variant A ver</label>
            <select
              value={abVerA}
              onChange={e=>setAbVerA(parseInt(e.target.value))}
              className="bg-neutral-800 rounded px-2 py-1"
            >
              {rows.filter(r=>r.name===name).sort((a,b)=>a.version-b.version).map(r=>(
                <option key={r.version} value={r.version}>{r.version}</option>
              ))}
            </select>
          </div>
          <div className="flex flex-col">
            <label className="text-neutral-400 mb-1">Variant B ver</label>
            <select
              value={abVerB}
              onChange={e=>setAbVerB(parseInt(e.target.value))}
              className="bg-neutral-800 rounded px-2 py-1"
            >
              {rows.filter(r=>r.name===name).sort((a,b)=>a.version-b.version).map(r=>(
                <option key={r.version} value={r.version}>{r.version}</option>
              ))}
            </select>
          </div>
          <div className="flex flex-col">
            <label className="text-neutral-400 mb-1">top_k</label>
            <input
              type="number"
              min={1}
              max={20}
              value={abTopK}
              onChange={e=>setAbTopK(Math.max(1, Math.min(20, parseInt(e.target.value || "5"))))}
              className="bg-neutral-800 rounded px-2 py-1"
            />
          </div>
        </div>
        <button
          onClick={runAB}
          disabled={abBusy}
          className="px-3 py-2 rounded bg-indigo-600 hover:bg-indigo-500 text-xs disabled:opacity-50"
        >
          {abBusy ? "Starting..." : "Run A/B"}
        </button>
      </div>
    </div>
  );
};

export default PromptsPage;
