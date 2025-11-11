import os, sys, json, requests

API = os.getenv("API_BASE", "http://localhost:8001")

def main():
    if len(sys.argv) < 6:
        print("Usage: python run_eval.py <dataset_id> <promptA_ver> <promptB_ver> <top_k> <rerank:true|false>")
        sys.exit(1)
    dataset_id, va, vb, top_k, rerank = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].lower()=="true"
    payload = {
        "dataset_id": dataset_id,
        "variant_a": { "prompt": {"name":"rag","version":va}, "retriever": {"top_k": top_k, "rerank": rerank} },
        "variant_b": { "prompt": {"name":"rag","version":vb}, "retriever": {"top_k": top_k, "rerank": rerank} }
    }
    r = requests.post(f"{API}/eval/run", json=payload, timeout=600)
    print(r.status_code, json.dumps(r.json() if r.headers.get("content-type","").startswith("application/json") else {"text": r.text}, indent=2))

if __name__ == "__main__":
    main()
