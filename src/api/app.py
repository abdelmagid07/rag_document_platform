from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()  
from .routes_documents import router as documents_router
from .routes_query import router as query_router
from .routes_evaluation import router as eval_router
from .metrics import metrics_store

app = FastAPI(
    title="RAG Document Platform",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router, prefix="/documents", tags=["Documents"])
app.include_router(query_router, prefix="/query", tags=["Query"])
app.include_router(eval_router, prefix="/evaluation", tags=["Evaluation"])


@app.get("/", tags=["Health"])
def root():
    return {"message": "RAG Document Platform API", "docs": "/docs"}


@app.get("/status", tags=["Health"])
def status() -> dict:
    return {
        "service": "rag-document-platform",
        "status": "ok",
        "metrics_summary": metrics_store.summary(),
    }


@app.get("/metrics", tags=["Health"])
def metrics() -> dict:
    return metrics_store.summary()
