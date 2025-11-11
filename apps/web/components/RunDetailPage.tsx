"use client";
import React, { useEffect, useState } from "react";
import { getRun, judgeRun, startBatchEval, getBatchStatus } from "../lib/api";
import { useRouter } from "next/navigation";

type Judgement = {
  query: string;
  answer: string;
  relevance: number;
  groundedness: number;
  rationale: string;
  variant?: string;
};

const RunDetailPage: React.FC<{ runId: string }> = ({ runId }) => {
  const [run, setRun] = useState<any>(null);
  const [judgements, setJudgements] = useState<Judgement[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [toast, setToast] = useState("");
  const [filter, setFilter] = useState<"all" | "a_wins" | "b_wins" | "ties">("all");
  const [judging, setJudging] = useState(false);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [batchStatus, setBatchStatus] = useState<any>(null);

  async function load() {
    setLoading(true);
    setErrorMsg("");
    try {
      const data = await getRun(runId);
      setRun(data);
      
      // Parse judgements from run data
      if (data.judgements) {
        setJudgements(data.judgements);
      }
    } catch (ex: any) {
      setErrorMsg(ex?.message || "Failed to load run");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (runId) load();
  }, [runId]);

  async function handleJudge() {
    setJudging(true);
    setErrorMsg("");
    try {
      const res = await judgeRun(runId);
      setToast(`Judged ${res.judged} items. Relevance: ${res.relevance_avg}, Groundedness: ${res.groundedness_avg}`);
      setTimeout(() => setToast(""), 3000);
      await load();
    } catch (ex: any) {
      setErrorMsg(ex?.message || "Judge failed");
    } finally {
      setJudging(false);
    }
  }

  async function handleBatchEval() {
    const group_id = run?.run?.meta?.group_id; // CHANGED
    if (!group_id) {
      setErrorMsg("No group_id found in run meta");
      return;
    }
    
    setErrorMsg("");
    try {
      const res = await startBatchEval(group_id);
      setBatchId(res.batch_id);
      setToast(`Batch job started: ${res.batch_id.slice(0, 8)}. ETA: ${res.eta_seconds}s`);
      setTimeout(() => setToast(""), 3000);
      // Start polling
      pollBatchStatus(res.batch_id);
    } catch (ex: any) {
      setErrorMsg(ex?.message || "Batch eval failed");
    }
  }

  async function pollBatchStatus(id: string) {
    const interval = setInterval(async () => {
      try {
        const status = await getBatchStatus(id);
        setBatchStatus(status);
        if (status.status === "completed") {
          clearInterval(interval);
          setToast("Batch evaluation completed!");
          await load();
        }
      } catch (ex) {
        clearInterval(interval);
      }
    }, 5000);  // poll every 5s
  }

  const filteredJudgements = judgements.filter(j => {
    if (filter === "all") return true;
    const score = (j.relevance + j.groundedness);
    // This is simplified - real implementation should compare A vs B
    if (filter === "a_wins") return j.variant === "A" && score > 6;
    if (filter === "b_wins") return j.variant === "B" && score > 6;
    if (filter === "ties") return score === 6;
    return true;
  });

  const avgRelevance = judgements.length
    ? (judgements.reduce((sum, j) => sum + j.relevance, 0) / judgements.length).toFixed(2)
    : "0.00";
  const avgGroundedness = judgements.length
    ? (judgements.reduce((sum, j) => sum + j.groundedness, 0) / judgements.length).toFixed(2)
    : "0.00";

  return (
    <div className="space-y-6">
      <div className="rounded border border-neutral-800 bg-neutral-900 p-4 space-y-3">
        <div className="flex justify-between items-center">
          <h3 className="text-sm font-semibold">Run: {runId.slice(0, 8)}</h3>
          <div className="space-x-2">
            <button
              onClick={handleJudge}
              disabled={judging}
              className="px-3 py-1 rounded bg-indigo-600 hover:bg-indigo-500 text-xs disabled:opacity-50"
            >
              {judging ? "Judging..." : "Judge Now"}
            </button>
            <button
              onClick={handleBatchEval}
              disabled={!!batchId}
              className="px-3 py-1 rounded bg-green-600 hover:bg-green-500 text-xs disabled:opacity-50"
            >
              Evaluate (Batch)
            </button>
          </div>
        </div>
        
        {toast && <div className="text-xs text-green-400">{toast}</div>}
        {errorMsg && (
          <div className="text-xs text-red-300 border border-red-700 bg-red-900/30 rounded px-2 py-1">
            {errorMsg}
          </div>
        )}

        {batchStatus && (
          <div className="text-xs text-neutral-400">
            Batch Status: {batchStatus.status} ({batchStatus.completed}/{batchStatus.total})
          </div>
        )}

        <div className="grid grid-cols-2 gap-4 text-xs">
          <div className="bg-neutral-800 rounded p-3">
            <div className="text-neutral-400">Avg Relevance</div>
            <div className="text-2xl font-bold">{avgRelevance}</div>
          </div>
          <div className="bg-neutral-800 rounded p-3">
            <div className="text-neutral-400">Avg Groundedness</div>
            <div className="text-2xl font-bold">{avgGroundedness}</div>
          </div>
        </div>
      </div>

      <div className="rounded border border-neutral-800 bg-neutral-900 p-4">
        <div className="flex justify-between items-center mb-3">
          <h4 className="text-sm font-semibold">Judgements ({filteredJudgements.length})</h4>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="bg-neutral-800 rounded px-2 py-1 text-xs"
          >
            <option value="all">All</option>
            <option value="a_wins">A Wins</option>
            <option value="b_wins">B Wins</option>
            <option value="ties">Ties</option>
          </select>
        </div>

        <table className="w-full text-xs">
          <thead className="text-neutral-400">
            <tr>
              <th className="text-left p-2">Question</th>
              <th className="text-left p-2">Answer</th>
              <th className="text-left p-2">Rel</th>
              <th className="text-left p-2">Gnd</th>
              <th className="text-left p-2">Rationale</th>
            </tr>
          </thead>
          <tbody>
            {filteredJudgements.map((j, idx) => (
              <tr key={idx} className="border-t border-neutral-800">
                <td className="p-2 max-w-xs truncate">{j.query}</td>
                <td className="p-2 max-w-md truncate">{j.answer}</td>
                <td className="p-2">{j.relevance}</td>
                <td className="p-2">{j.groundedness}</td>
                <td className="p-2 max-w-lg truncate" title={j.rationale}>{j.rationale}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default RunDetailPage;
