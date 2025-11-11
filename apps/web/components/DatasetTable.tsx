import React from "react";

function isoParts(iso?: string | null): { date: string; time: string } {
  if (!iso || typeof iso !== "string") return { date: "-", time: "-" };
  const [d, rest] = iso.split("T");
  const time = (rest || "").split(/[.+-]/)[0] || "-";
  return { date: d || "-", time };
}

export const DatasetTable: React.FC<{
  datasets: any[];
  onPreview: (id: string) => void;
  onUse: (id: string) => void;
  onDelete: (id: string) => void;
}> = ({ datasets, onPreview, onUse, onDelete }) => {
  return (
    <table style={{ width: "100%", fontSize: "12px" }}>
      <thead>
        <tr>
          <th style={{ textAlign: "left" }}>Name</th>
          <th style={{ textAlign: "left" }}>Items</th>
          <th style={{ textAlign: "left" }}>Doc cov</th>
          <th style={{ textAlign: "left" }}>Status</th>
          <th style={{ textAlign: "left" }}>Date</th>
          <th style={{ textAlign: "left" }}>Time</th>
          <th style={{ textAlign: "left" }}>Actions</th>
        </tr>
      </thead>
      <tbody>
        {datasets.map((d) => {
          const items = d.items || 0;
          const cv = d.doc_coverage || { covered: 0, total: items, percent: 0 };
          const { date, time } = isoParts(d.created_at);
          return (
            <tr key={d.dataset_id || d.id}>
              <td>{d.name}</td>
              <td>{items}</td>
              <td>{cv.covered}/{cv.total} ({cv.percent}%)</td>
              <td>{d.status || "ready"}</td>
              <td>{date}</td>
              <td>{time}</td>
              <td style={{ whiteSpace: "nowrap" }}>
                <button onClick={()=>onPreview(d.dataset_id || d.id)}>Preview</button>{" "}
                <button onClick={()=>onUse(d.dataset_id || d.id)}>Use in A/B</button>{" "}
                <button onClick={()=>onDelete(d.dataset_id || d.id)}>Delete</button>
              </td>
            </tr>
          );
        })}
        {!datasets.length && (
          <tr><td colSpan={7} style={{ padding: 8, color: "#888" }}>No QA datasets.</td></tr>
        )}
      </tbody>
    </table>
  );
};
