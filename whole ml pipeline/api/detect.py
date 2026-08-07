"""
FastAPI Detect API for InTakeoff Pipeline.
"""

from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import shutil
import os
import uuid
from typing import Dict, Any

from pipelines.full_pipeline import FullPipeline
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["detect"])

# Ideally instantiated at app startup and injected
# For this script, we declare a global mock placeholder
pipeline_instance = None

class ProcessingResponse(BaseModel):
    job_id: str
    status: str
    message: str

@router.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf_endpoint(background_tasks: BackgroundTasks, file: UploadFile = File(...), scale: str = "1/8"):
    """
    Accepts a PDF upload and triggers the asynchronous ML pipeline.
    """
    job_id = str(uuid.uuid4())
    temp_path = f"data/raw/{job_id}_{file.filename}"
    
    os.makedirs("data/raw", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    logger.info(f"Received PDF {file.filename}. Job ID: {job_id}")
    
    # In a real system, this would be pushed to Celery or Redis Queue.
    # Here we use FastAPI BackgroundTasks for async execution.
    background_tasks.add_task(run_pipeline_task, temp_path, scale, job_id)
    
    return ProcessingResponse(
        job_id=job_id,
        status="processing",
        message="PDF uploaded successfully. Pipeline is running in the background."
    )

def run_pipeline_task(pdf_path: str, scale: str, job_id: str):
    """Background task to run the ML pipeline."""
    global pipeline_instance
    if not pipeline_instance:
        try:
            pipeline_instance = FullPipeline()
        except Exception as e:
            logger.error(f"Job {job_id} failed to init pipeline: {e}")
            return
            
    try:
        result = pipeline_instance.process_pdf(pdf_path, scale_str=scale)
        logger.info(f"Job {job_id} completed successfully. Found {result['symbols_detected']} symbols.")
        # Store result to DB...
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
