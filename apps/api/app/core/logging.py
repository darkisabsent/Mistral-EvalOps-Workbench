import json, logging, time, uuid, hashlib, random, traceback, contextvars
from typing import Any, Dict, Iterable, Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from .settings import settings

# Correlation/context
_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("run_id", default=None)
_route: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("route", default=None)

# Logger and tracer
_logger = logging.getLogger("api")
tracer = trace.get_tracer("api")

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z"

def _trace_ids() -> Dict[str, str]:
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx or not ctx.is_valid:
        return {}
    return {
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
    }

def hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def hash_vector(vec: Iterable[float]) -> str:
    # round to 4 decimals to stabilize hash a bit
    return hashlib.sha256(json.dumps([round(x, 4) for x in vec]).encode("utf-8")).hexdigest()

def token_estimate_from_texts(texts: Iterable[str]) -> int:
    # crude: ~4 chars per token
    total_chars = sum(len(t or "") for t in texts)
    return max(1, total_chars // 4)

def model_event_sampled() -> bool:
    try:
        rate = float(settings.log_sample_model_events)
    except Exception:
        rate = 0.2
    rate = min(max(rate, 0.0), 1.0)
    return random.random() < rate

def log_event(event: str, level: str = "info", **fields: Any) -> None:
    base: Dict[str, Any] = {
        "ts": _now_iso(),
        "level": level.lower(),
        "event": event,
        "service": "api",
        "env": settings.env,
        "route": _route.get(),
        "run_id": _run_id.get(),
        "request_id": _request_id.get(),
    }
    base.update(_trace_ids())
    base.update(fields)
    line = json.dumps(base, separators=(",", ":"), ensure_ascii=False)
    getattr(_logger, level.lower() if hasattr(_logger, level.lower()) else "info")(line)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Correlation IDs
        rid = request.headers.get(settings.request_id_header) or request.headers.get(settings.request_id_header.upper())
        runid = request.headers.get(settings.run_id_header) or request.headers.get(settings.run_id_header.upper())
        if not rid:
            rid = str(uuid.uuid4())
        if not runid:
            runid = rid

        token_r = _request_id.set(rid)
        token_ru = _run_id.set(runid)
        token_route = _route.set(request.url.path)

        # OTel span
        attrs = {
            "http.method": request.method,
            "http.target": request.url.path,
            "net.peer.ip": request.client.host if request.client else None,
            "user_agent.original": request.headers.get("user-agent"),
            "run_id": runid,
            "request_id": rid,
        }
        start = time.time()
        try:
            with tracer.start_as_current_span(f"http {request.method} {request.url.path}") as span:
                for k, v in attrs.items():
                    if v is not None:
                        span.set_attribute(k, v)
                # http.request (always)
                log_event(
                    "http.request",
                    method=request.method,
                    path=request.url.path,
                    user_agent=request.headers.get("user-agent"),
                    remote_ip=request.client.host if request.client else None,
                )
                response = await call_next(request)
                latency = int((time.time() - start) * 1000)
                # http.response
                log_event(
                    "http.response",
                    status=response.status_code,
                    latency_ms=latency,
                )
                # propagate ids back
                response.headers[settings.request_id_header] = rid
                response.headers[settings.run_id_header] = runid
                return response
        except Exception as ex:
            # error
            log_event(
                "error",
                level="error",
                kind=ex.__class__.__name__,
                message=str(ex),
                stack=traceback.format_exc(limit=20),
            )
            raise
        finally:
            _request_id.reset(token_r)
            _run_id.reset(token_ru)
            _route.reset(token_route)

def setup_logging() -> None:
    # std logging to stdout, one line JSON per event
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, handlers=[logging.StreamHandler()])
    # OpenTelemetry tracer provider if endpoint configured
    if settings.otlp_endpoint:
        resource = Resource.create({"service.name": "api", "deployment.environment": settings.env})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
