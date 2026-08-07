from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, List

from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/process-schedule",
    tags=["processing"]
)

@router.post("/", response_model=Dict[str, Any])
async def process_schedule_page(file: UploadFile = File(...)):
    """
    Extracts structured equipment schedules from an uploaded schedule drawing/PDF.
    """
    logger.info(f"Received request to process schedule from file: {file.filename}")
    
    if not file.filename.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        logger.error("Invalid file format submitted for schedule processing.")
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and images are supported.")
        
    try:
        # TODO: Integrate actual schedule extraction pipeline here
        # content = await file.read()
        # pipeline = SchedulePipeline()
        # tables = pipeline.run(content)
        
        mock_response = {
            "filename": file.filename,
            "status": "success",
            "extracted_tables": [
                {
                    "table_id": 1,
                    "rows": [
                        {"tag": "VAV-1", "description": "Variable Air Volume Box", "quantity": 10},
                        {"tag": "FCU-1", "description": "Fan Coil Unit", "quantity": 5}
                    ]
                }
            ]
        }
        
        logger.info(f"Successfully processed schedule {file.filename}.")
        return mock_response
        
    except Exception as e:
        logger.error(f"Error during schedule processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during schedule extraction.")
