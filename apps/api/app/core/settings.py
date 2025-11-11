from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    env: str = "dev"
    mistral_api_key: str
    backend: str = "mistral"
    llm_base_url: str = "https://api.mistral.ai"
    model_chat: str = "mistral-small-latest"
    model_embed: str = "mistral-embed"

    database_url: str

    # logging / tracing
    log_level: str = "INFO"
    log_sample_model_events: float = 0.2  # 0..1
    request_id_header: str = "x-request-id"
    run_id_header: str = "x-run-id"
    otlp_endpoint: Optional[str] = None  # e.g. http://otel-collector:4318
    dataset_max_size: int = 500
    max_prompt_chars: int = 8000  # guard extremely long prompt versions
    # token cost (cents per million tokens)
    input_tokens_cost_per_million: float = 30.0
    output_tokens_cost_per_million: float = 60.0
    # retriever defaults
    retriever_top_k: int = 8
    retriever_rerank_strategy: str = "none"        # none|llm|cross_encoder
    retriever_rerank_k: int = 8
    # chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # judge defaults
    judge_model: Optional[str] = None                 # fallback to model_chat if None
    judge_temperature: float = 0.1
    judge_repeats: int = 1
    # logging sampling alias
    log_sampling_rate: Optional[float] = None         # if set overrides log_sample_model_events

    class Config:
        case_sensitive = False
        env_file = ".env"  # compose mounts root .env.example as .env in container

settings = Settings()

# alias override
if settings.log_sampling_rate is not None:
    settings.log_sample_model_events = settings.log_sampling_rate
