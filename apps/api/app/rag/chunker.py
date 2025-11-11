from typing import List
from ..core.settings import settings

def chunk_text(text: str, size: int = None, overlap: int = None) -> List[str]:
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + size, n)
        chunks.append(text[i:end])
        i += max(1, size - overlap)
    return chunks
