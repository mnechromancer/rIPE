"""
State space management routes for the IPE API.
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/states", tags=["states"])


class StateSpacePoint(BaseModel):
    """A point in the physiological state space."""

    id: str
    coordinates: List[float]
    fitness: float
    generation: int
    metadata: Optional[Dict[str, Any]] = None


class StateSpaceQuery(BaseModel):
    """Query parameters for state space filtering."""

    min_fitness: Optional[float] = None
    max_fitness: Optional[float] = None
    generation: Optional[int] = None
    limit: Optional[int] = 100


# In-memory storage for demo
state_space_store: Dict[str, List[StateSpacePoint]] = {}


@router.get("/{sim_id}", response_model=List[StateSpacePoint])
async def get_state_space(
    sim_id: str,
    min_fitness: Optional[float] = Query(None, description="Minimum fitness filter"),
    max_fitness: Optional[float] = Query(None, description="Maximum fitness filter"),
    generation: Optional[int] = Query(None, description="Generation filter"),
    limit: int = Query(100, description="Maximum number of points to return"),
) -> List[StateSpacePoint]:
    """Get state space points for a simulation with optional filtering."""
    if sim_id not in state_space_store:
        # Return empty list if no states exist yet
        return []

    points = state_space_store[sim_id]

    # Apply filters
    if min_fitness is not None:
        points = [p for p in points if p.fitness >= min_fitness]
    if max_fitness is not None:
        points = [p for p in points if p.fitness <= max_fitness]
    if generation is not None:
        points = [p for p in points if p.generation == generation]

    # Apply limit
    return points[:limit]


@router.post("/{sim_id}")
async def add_state_point(sim_id: str, state_point: StateSpacePoint):
    """Add a state space point to a simulation."""
    if sim_id not in state_space_store:
        state_space_store[sim_id] = []

    state_space_store[sim_id].append(state_point)
    logger.info(f"Added state point to simulation {sim_id}")
    return {"message": "State point added", "point_id": state_point.id}


@router.post("/{sim_id}/batch")
async def add_state_points_batch(sim_id: str, state_points: List[StateSpacePoint]):
    """Add multiple state space points to a simulation."""
    if sim_id not in state_space_store:
        state_space_store[sim_id] = []

    state_space_store[sim_id].extend(state_points)
    logger.info(f"Added {len(state_points)} state points to simulation {sim_id}")
    return {"message": f"Added {len(state_points)} state points"}


@router.get("/{sim_id}/summary")
async def get_state_summary(sim_id: str):
    """Get summary statistics for the state space."""
    if sim_id not in state_space_store:
        return {"total_points": 0}

    points = state_space_store[sim_id]
    if not points:
        return {"total_points": 0}

    fitnesses = [p.fitness for p in points]
    generations = [p.generation for p in points]

    return {
        "total_points": len(points),
        "fitness_stats": {
            "min": min(fitnesses),
            "max": max(fitnesses),
            "avg": sum(fitnesses) / len(fitnesses),
        },
        "generation_range": {"min": min(generations), "max": max(generations)},
    }


@router.delete("/{sim_id}")
async def clear_state_space(sim_id: str):
    """Clear all state space points for a simulation."""
    if sim_id in state_space_store:
        del state_space_store[sim_id]
    return {"message": f"State space cleared for simulation {sim_id}"}
