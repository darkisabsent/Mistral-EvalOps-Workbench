from pydantic import BaseModel

class IngestOut(BaseModel):
    document_ids: list[str]
    chunks: int

class ChatReq(BaseModel):
    query: str
    prompt_name: str = "rag"
    prompt_version: int = 1
    top_k: int = 8
    rerank: bool = False

class EvalReq(BaseModel):
    dataset_id: str
    variant_a: dict
    variant_b: dict
    backend: str = "mistral"
    use_batch_api: bool = False
