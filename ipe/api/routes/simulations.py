"""
Simulation management routes for the IPE API.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/simulations", tags=["simulations"])


class SimulationParams(BaseModel):
    """Parameters for simulation configuration."""

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


# In-memory storage for demo
simulations_store: Dict[str, Dict[str, Any]] = {}


@router.post("", response_model=SimulationResponse)
async def create_simulation(params: SimulationParams) -> SimulationResponse:
    """Create a new simulation."""
    sim_id = str(uuid.uuid4())
    simulation = {
        "id": sim_id,
        "name": params.name,
        "status": "created",
        "created_at": datetime.now(),
        "parameters": params,
    }
    simulations_store[sim_id] = simulation
    logger.info(f"Created simulation {sim_id}")
    return SimulationResponse(**simulation)


@router.get("", response_model=List[SimulationResponse])
async def list_simulations() -> List[SimulationResponse]:
    """List all simulations."""
    return [SimulationResponse(**sim) for sim in simulations_store.values()]


@router.get("/{sim_id}", response_model=SimulationResponse)
async def get_simulation(sim_id: str) -> SimulationResponse:
    """Get a specific simulation."""
    if sim_id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return SimulationResponse(**simulations_store[sim_id])


@router.put("/{sim_id}", response_model=SimulationResponse)
async def update_simulation(
    sim_id: str, params: SimulationParams
) -> SimulationResponse:
    """Update a simulation."""
    if sim_id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")

    simulations_store[sim_id]["parameters"] = params
    simulations_store[sim_id]["name"] = params.name
    return SimulationResponse(**simulations_store[sim_id])


@router.delete("/{sim_id}")
async def delete_simulation(sim_id: str):
    """Delete a simulation."""
    if sim_id not in simulations_store:
        raise HTTPException(status_code=404, detail="Simulation not found")

    del simulations_store[sim_id]
    return {"message": f"Simulation {sim_id} deleted"}
