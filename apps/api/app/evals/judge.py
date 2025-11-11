from ..core.mistral_client import get_client
from ..core.settings import settings
from ..core.logging import log_event, tracer, hash_text, token_estimate_from_texts, model_event_sampled
import time, statistics, json
from typing import Optional

SCHEMA = {
  "type":"object",
  "properties":{
    "relevance":{"type":"integer","minimum":1,"maximum":5},
    "groundedness":{"type":"integer","minimum":1,"maximum":5},
    "rationale":{"type":"string"}
  },
  "required":["relevance","groundedness","rationale"]
}

def judge_one(question: str, answer: str, context: str, config: Optional[dict] = None) -> dict:
    cfg = config or {}
    model = cfg.get("model", settings.model_chat)
    temperature = float(cfg.get("temperature", 0.1))
    max_tokens = int(cfg.get("max_tokens", 256))
    repeats = max(1, int(cfg.get("repeats", 1)))

    client = get_client()
    base_prompt = (
        "Evaluate the answer strictly using ONLY the provided context. "
        "Return JSON with fields: relevance (1-5), groundedness (1-5), rationale (short; cite which context snippets mattered by number/index, not full text)."
    )
    user_content = f"Question: {question}\nAnswer: {answer}\nContext:\n{context}"

    def _call(reminder: bool = False):
        msgs = [
            {"role": "system", "content": "Return strict JSON matching the provided schema."},
            {"role": "user", "content": base_prompt + ("\nReminder: return a JSON object that validates against the schema." if reminder else "")},
            {"role": "user", "content": user_content},
        ]
        sampled = model_event_sampled()
        tokens_est = token_estimate_from_texts([m["content"] for m in msgs])
        if sampled:
            log_event(
                "llm.request",
                provider=settings.backend, model=model, mode="json",
                prompt_hash=hash_text(user_content), tokens_estimate=tokens_est,
            )
        start = time.time()
        with tracer.start_as_current_span("llm.judge.json") as span:
            span.set_attribute("llm.provider", settings.backend)
            span.set_attribute("llm.model", model)
            resp = client.chat.complete(
                model=model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object", "schema": SCHEMA},
            )
        latency = int((time.time() - start) * 1000)
        usage = getattr(resp, "usage", None) or {}
        in_tok = getattr(usage, "input_tokens", None) or usage.get("input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", None) or usage.get("output_tokens", 0) or 0
        if sampled:
            log_event(
                "llm.response",
                latency_ms=latency, input_tokens=in_tok, output_tokens=out_tok,
                finish_reason="ok", cost_cents=0,
            )
        # resp.output_parsed should already respect schema; still guard
        try:
            out = resp.output_parsed
            # sanity checks
            _ = int(out["relevance"]); _ = int(out["groundedness"]); _ = str(out["rationale"])
        except Exception:
            # try to parse raw content if present
            try:
                raw = resp.output_text if hasattr(resp, "output_text") else ""
                out = json.loads(raw)
            except Exception:
                raise
        return out, {"latency_ms": latency, "input_tokens": in_tok, "output_tokens": out_tok}

    results, latencies, inputs, outputs, rationales = [], [], [], [], []
    for i in range(repeats):
        try:
            out, meta = _call(reminder=False)
        except Exception:
            # retry once with reminder
            try:
                out, meta = _call(reminder=True)
            except Exception:
                out = {"relevance": 0, "groundedness": 0, "rationale": "invalid_json"}
                meta = {"latency_ms": 0, "input_tokens": 0, "output_tokens": 0}
        results.append(out)
        latencies.append(meta["latency_ms"])
        inputs.append(meta["input_tokens"])
        outputs.append(meta["output_tokens"])
        rationales.append(out.get("rationale", ""))

    # aggregate (median of scores to smooth variance)
    rels = [int(r.get("relevance", 0)) for r in results]
    grs = [int(r.get("groundedness", 0)) for r in results]
    relevance = int(statistics.median(rels)) if rels else 0
    groundedness = int(statistics.median(grs)) if grs else 0
    rationale = rationales[0] if rationales else ""

    # eval.judge summary (no raw texts)
    log_event(
        "eval.judge",
        timing_ms=sum(latencies) // max(1, len(latencies)),
        scores={"relevance": relevance, "groundedness": groundedness},
        question_hash=hash_text(question),
        answer_hash=hash_text(answer),
        context_hash=hash_text(context),
    )

    return {
        "relevance": relevance,
        "groundedness": groundedness,
        "rationale": rationale,
        "latency_ms": sum(latencies),
        "input_tokens": sum(inputs),
        "output_tokens": sum(outputs),
        "model": model,
        "backend": settings.backend,
        "repeats": repeats,
    }
