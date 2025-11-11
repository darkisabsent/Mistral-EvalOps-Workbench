from . import retriever
from ..core.mistral_client import get_client
from ..core.settings import settings
from ..core.logging import log_event, tracer, hash_text, token_estimate_from_texts, model_event_sampled
import time, logging
from httpx import HTTPStatusError

logger = logging.getLogger("api")

def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_client()
    sampled = model_event_sampled()
    token_est = token_estimate_from_texts(texts)
    prompt_hash = hash_text("|".join(texts[:10]))  # avoid logging raw inputs
    start = time.time()
    if sampled:
        log_event(
            "llm.request",
            provider=settings.backend,
            model=settings.model_embed,
            mode="embed",
            prompt_hash=prompt_hash,
            tokens_estimate=token_est,
        )
    with tracer.start_as_current_span("llm.embed") as span:
        span.set_attribute("llm.provider", settings.backend)
        span.set_attribute("llm.model", settings.model_embed)
        resp = client.embeddings.create(model=settings.model_embed, inputs=texts)
    latency = int((time.time() - start) * 1000)
    if sampled:
        log_event(
            "llm.response",
            latency_ms=latency,
            input_tokens=token_est,  # API may not return usage for embeddings
            output_tokens=0,
            finish_reason="ok",
            cost_cents=0,  # left 0 unless you calculate
        )
    return [item.embedding for item in resp.data]

def embed_texts_with_retry(texts: list[str], max_retries: int = 3, backoff_base: float = 0.6):
    """
    Wrapper with exponential backoff for transient/provider 429 errors.
    Returns list[list[float]] or raises last error.
    """
    last_ex = None
    for attempt in range(max_retries):
        try:
            return embed_texts(texts)  # original function
        except HTTPStatusError as ex:  # mistral client might raise wrapped httpx errors
            status = getattr(ex.response, "status_code", None)
            if status == 429 and attempt < max_retries - 1:
                delay = backoff_base * (2 ** attempt)
                logger.warning(f"embed 429 rate limit — retry {attempt+1}/{max_retries} in {delay:.2f}s")
                time.sleep(delay)
                continue
            last_ex = ex
            break
        except Exception as ex:
            # Non-HTTP or other error; retry only if attempt < max_retries-1
            last_ex = ex
            if attempt < max_retries - 1:
                delay = backoff_base * (2 ** attempt)
                logger.warning(f"embed error ({ex}); retry {attempt+1}/{max_retries} in {delay:.2f}s")
                time.sleep(delay)
                continue
            break
    raise last_ex or RuntimeError("embedding_failed")
