from pydantic import BaseModel
from typing import List, Optional


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class Source(BaseModel):
    doc_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    latency_ms: float


class UploadResponse(BaseModel):
    document_id: str
    status: str


class DocumentMetadata(BaseModel):
    id: str
    filename: str


class EvaluationRequest(BaseModel):
    sample_size: int = 25


class EvaluationResponse(BaseModel):
    recall_at_k: float
    mrr: float
    sample_size: int