"""
Pydantic Schemas for API request/response models.
"""

from pydantic import BaseModel
from typing import Dict, Any, Optional

class AnalysisPaths(BaseModel):
    """Input model for file paths."""
    active_traj: str
    active_topo: str
    inactive_traj: str
    inactive_topo: str
    config_file: str # Path to the config file (on the server)

class StaticAnalysisParams(BaseModel):
    """Parameters for Goal 1."""
    paths: AnalysisPaths

class DynamicAnalysisParams(BaseModel):
    """Parameters for Goal 3."""
    paths: AnalysisPaths
    te_lag: Optional[int] = 10
    
class StaticAnalysisResponse(BaseModel):
    """Response model for Goal 1."""
    status: str
    goal: str
    results: Dict[str, float]

class DynamicAnalysisResponse(BaseModel):
    """Response model for Goal 3."""
    status: str
    goal: str
    results: Dict[str, Any] # Contains M_inactive, M_active
    
class ErrorResponse(BaseModel):
    """Error response model."""
    status: str
    error: str