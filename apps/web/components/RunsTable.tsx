import React from "react";

export const RunsTable: React.FC<{ runs: any[]; onSelect?: (id:string)=>void }> = ({ runs, onSelect }) => {
  return (
    <table style={{width:"100%", fontSize:"12px"}}>
      <thead>
        <tr>
          <th>ID</th><th>Kind</th><th>Latency</th><th>Rel</th><th>Grd</th><th>Tokens In</th><th>Tokens Out</th><th>Cost¢</th>
        </tr>
      </thead>
      <tbody>
        {runs.map(r=>(
          <tr key={r.id} style={{cursor:"pointer"}} onClick={()=>onSelect && onSelect(r.id)}>
            <td>{r.id.slice(0,8)}</td>
            <td>{r.kind}</td>
            <td>{r.latency_ms ?? "-"}</td>
            <td>{r.avg_relevance?.toFixed?.(2) ?? "-"}</td>
            <td>{r.avg_groundedness?.toFixed?.(2) ?? "-"}</td>
            <td>{r.input_tokens ?? "-"}</td>
            <td>{r.output_tokens ?? "-"}</td>
            <td>{r.cost_cents?.toFixed?.(2) ?? "-"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};
