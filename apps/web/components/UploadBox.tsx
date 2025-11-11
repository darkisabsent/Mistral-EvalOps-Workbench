import React, { useState } from "react";
import { API_BASE_URL } from "../lib/api";

export const UploadBox: React.FC<{ onUploaded:(d:any)=>void }> = ({ onUploaded }) => {
  const [pending, setPending] = useState(false);
  const [name, setName] = useState("");
  const [err, setErr] = useState("");
  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    setErr("");
    const file = e.target.files?.[0];
    if (!file || !name) return;
    setPending(true);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`${API_BASE_URL}/datasets/upload?name=${encodeURIComponent(name)}`, { method:"POST", body: fd });
      if (!res.ok) {
        const txt = await res.text();
        setErr(txt || `Upload failed (${res.status})`);
      } else {
        const json = await res.json();
        onUploaded(json);
      }
    } catch (ex:any) {
      setErr(ex?.message || "Network error");
    } finally {
      setPending(false);
    }
  }
  return (
    <div style={{border:"1px solid #444", padding:"8px"}}>
      <input placeholder="Dataset name" value={name} onChange={e=>setName(e.target.value)} />
      <input type="file" accept=".jsonl" onChange={handleFile} disabled={pending}/>
      {pending && <span> Uploading...</span>}
      {err && <div style={{color:"#f55", fontSize:"11px"}}>{err}</div>}
    </div>
  );
};
