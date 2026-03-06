# RAG Document Platform

Production-style Retrieval-Augmented Generation (RAG) scaffold for document ingestion, chunking, embedding, retrieval, and API/UI integration.

## Repository Layout

- `src/config.py`: environment and model/runtime settings
- `src/models/`: embedding and LlamaIndex integration modules
- `src/processing/`: PDF processing, chunking, and document store/retriever
- `src/api/`: FastAPI app and metrics helpers
- `src/ui/`: Gradio-oriented UI logic
- `tests/`: unit tests for processing and retrieval components
- `data/`: local PDFs, indexes, and logs

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn src.api.fastapi_app:app --reload
```
