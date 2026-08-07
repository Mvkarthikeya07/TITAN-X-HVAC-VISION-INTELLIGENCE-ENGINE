"""
Health Check API for InTakeoff Pipeline.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["health"])

@router.get("/health")
def health_check():
    """Returns the operational status of the ML API."""
    return {
        "status": "online",
        "gpu_available": True, # Usually derived dynamically
        "service": "InTakeoff ML Engine"
    }
