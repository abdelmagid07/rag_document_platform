from fastapi import APIRouter, UploadFile, File, Form
from ..services.ingestion_service import ingest_document
from .schemas import UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), user_id: str = Form(...)):
    
    document_id = await ingest_document(file, user_id)

    return UploadResponse(
        document_id=document_id,
        status="processed"
    )