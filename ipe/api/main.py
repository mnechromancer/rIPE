"""
API-001: Core REST API

FastAPI setup with OpenAPI docs, CRUD operations for simulations,
state space endpoints, and authentication/authorization.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import uuid
from datetime import datetime

# Import route modules
from .routes import simulations, states, export
from .websocket import realtime

logger = logging.getLogger(__name__)

# Initialize FastAPI app with OpenAPI docs
app = FastAPI(
    title="IPE API",
    description="Integrated Physiological Evolution API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Security
security = HTTPBearer()


# Data models
class SimulationParams(BaseModel):
    """Parameters for creating a new simulation."""

    name: str
    duration: int = 100
    population_size: int = 1000
    mutation_rate: float = 0.001
    environment_params: Optional[Dict[str, Any]] = None


class SimulationResponse(BaseModel):
    """Response model for simulation operations."""

    id: str
    name: str
    status: str
    created_at: datetime
    parameters: SimulationParams


class StateSpacePoint(BaseModel):
    """A point in the physiological state space."""

    id: str
    coordinates: List[float]
    fitness: float
    generation: int


# In-memory storage (for demo purposes)
simulations_db: Dict[str, Dict[str, Any]] = {}
state_space_db: Dict[str, List[StateSpacePoint]] = {}


# Authentication (simplified for demo)
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token (simplified implementation)."""
    token = credentials.credentials
    if not token or token != "demo-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# Include routers
app.include_router(simulations.router)
app.include_router(states.router)
app.include_router(export.router)
app.include_router(realtime.router)


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "IPE API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
