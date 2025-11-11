import os
from mistralai import Mistral
from .settings import settings

def get_client() -> Mistral:
    return Mistral(
        api_key=settings.mistral_api_key,
        server_url=settings.llm_base_url,
    )