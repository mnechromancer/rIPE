"""
API-003: Data Export Endpoints

Export to CSV/JSON/HDF5, publication figure generation,
statistical summary export, and batch download support.
"""

from fastapi import APIRouter, HTTPException, Response, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import json
import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import tempfile
import zipfile
import os
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/export", tags=["export"])


# Data models
class ExportRequest(BaseModel):
    """Request model for data export."""

    simulation_id: str
    format: str  # 'csv', 'json', 'hdf5'
    data_types: List[str] = ["simulation_data", "state_space", "statistics"]
    filters: Optional[Dict[str, Any]] = None


class FigureRequest(BaseModel):
    """Request model for publication figure generation."""

    simulation_id: str
    figure_type: str  # 'state_space', 'evolution_timeline', 'fitness_landscape'
    format: str = "png"  # 'png', 'pdf', 'svg'
    width: int = 12
    height: int = 8
    dpi: int = 300


class BatchExportRequest(BaseModel):
    """Request model for batch export."""

    simulation_ids: List[str]
    formats: List[str]
    include_figures: bool = False


class ExportStatus(BaseModel):
    """Response model for export status."""

    export_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: float
    download_url: Optional[str] = None
    created_at: datetime


# In-memory storage for demo (in production, use proper storage/queue)
export_jobs: Dict[str, ExportStatus] = {}
sample_data: Dict[str, Dict[str, Any]] = {
    "demo-sim-001": {
        "simulation_data": pd.DataFrame(
            {
                "generation": range(100),
                "population_size": np.random.normal(1000, 50, 100).astype(int),
                "mean_fitness": np.random.normal(0.7, 0.1, 100),
                "mutation_rate": [0.001] * 100,
            }
        ),
        "state_space": pd.DataFrame(
            {
                "individual_id": [f"ind_{i}" for i in range(500)],
                "generation": np.random.randint(0, 100, 500),
                "x_coord": np.random.normal(0, 1, 500),
                "y_coord": np.random.normal(0, 1, 500),
                "z_coord": np.random.normal(0, 1, 500),
                "fitness": np.random.beta(2, 5, 500),
            }
        ),
        "statistics": {
            "total_generations": 100,
            "final_population": 1023,
            "max_fitness": 0.95,
            "mean_fitness": 0.71,
            "evolution_rate": 0.034,
        },
    }
}


@router.post("/simulation/{simulation_id}", response_model=ExportStatus)
async def export_simulation_data(
    simulation_id: str, export_request: ExportRequest, background_tasks: BackgroundTasks
) -> ExportStatus:
    """Export simulation data in specified format."""
    export_id = str(uuid.uuid4())

    # Create export job
    export_status = ExportStatus(
        export_id=export_id, status="pending", progress=0.0, created_at=datetime.now()
    )
    export_jobs[export_id] = export_status

    # Add background task to process export
    background_tasks.add_task(process_export, export_id, simulation_id, export_request)

    return export_status


@router.get("/status/{export_id}", response_model=ExportStatus)
async def get_export_status(export_id: str) -> ExportStatus:
    """Get export job status."""
    if export_id not in export_jobs:
        raise HTTPException(status_code=404, detail="Export job not found")

    return export_jobs[export_id]


@router.get("/download/{export_id}")
async def download_export(export_id: str):
    """Download exported data."""
    if export_id not in export_jobs:
        raise HTTPException(status_code=404, detail="Export job not found")

    job = export_jobs[export_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Export not ready")

    # For demo, create a sample file
    return create_sample_export_file(export_id)


@router.post("/figure/{simulation_id}")
async def generate_publication_figure(
    simulation_id: str, figure_request: FigureRequest
) -> StreamingResponse:
    """Generate publication-ready figures."""
    if simulation_id not in sample_data:
        raise HTTPException(status_code=404, detail="Simulation not found")

    # Create figure based on type
    fig, ax = plt.subplots(figsize=(figure_request.width, figure_request.height))

    data = sample_data[simulation_id]

    if figure_request.figure_type == "state_space":
        # 3D state space plot (projected to 2D)
        state_df = data["state_space"]
        scatter = ax.scatter(
            state_df["x_coord"],
            state_df["y_coord"],
            c=state_df["fitness"],
            cmap="viridis",
            alpha=0.6,
        )
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"State Space - {simulation_id}")
        plt.colorbar(scatter, label="Fitness")

    elif figure_request.figure_type == "evolution_timeline":
        # Evolution over time
        sim_df = data["simulation_data"]
        ax.plot(sim_df["generation"], sim_df["mean_fitness"], "b-", linewidth=2)
        ax.fill_between(
            sim_df["generation"],
            sim_df["mean_fitness"] - 0.1,
            sim_df["mean_fitness"] + 0.1,
            alpha=0.3,
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("Mean Fitness")
        ax.set_title(f"Fitness Evolution - {simulation_id}")

    elif figure_request.figure_type == "fitness_landscape":
        # Fitness landscape heatmap
        state_df = data["state_space"]
        # Create 2D histogram of fitness
        x_bins = np.linspace(state_df["x_coord"].min(), state_df["x_coord"].max(), 20)
        y_bins = np.linspace(state_df["y_coord"].min(), state_df["y_coord"].max(), 20)

        fitness_grid = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                mask = (
                    (state_df["x_coord"] >= x_bins[i])
                    & (state_df["x_coord"] < x_bins[i + 1])
                    & (state_df["y_coord"] >= y_bins[j])
                    & (state_df["y_coord"] < y_bins[j + 1])
                )
                if mask.sum() > 0:
                    fitness_grid[j, i] = state_df.loc[mask, "fitness"].mean()

        im = ax.imshow(fitness_grid, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"Fitness Landscape - {simulation_id}")
        plt.colorbar(im, label="Mean Fitness")

    else:
        raise HTTPException(status_code=400, detail="Unknown figure type")

    # Save to bytes buffer
    buffer = io.BytesIO()
    plt.savefig(
        buffer,
        format=figure_request.format,
        dpi=figure_request.dpi,
        bbox_inches="tight",
    )
    buffer.seek(0)
    plt.close()

    # Determine media type
    media_types = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}

    return StreamingResponse(
        io.BytesIO(buffer.read()),
        media_type=media_types.get(figure_request.format, "image/png"),
        headers={
            "Content-Disposition": f"attachment; filename={simulation_id}_figure.{figure_request.format}"
        },
    )


@router.get("/summary/{simulation_id}")
async def get_statistical_summary(simulation_id: str):
    """Get statistical summary for a simulation."""
    if simulation_id not in sample_data:
        raise HTTPException(status_code=404, detail="Simulation not found")

    data = sample_data[simulation_id]
    sim_df = data["simulation_data"]
    state_df = data["state_space"]
    stats = data["statistics"]

    summary = {
        "simulation_id": simulation_id,
        "basic_stats": stats,
        "population_stats": {
            "mean_population": float(sim_df["population_size"].mean()),
            "std_population": float(sim_df["population_size"].std()),
            "min_population": int(sim_df["population_size"].min()),
            "max_population": int(sim_df["population_size"].max()),
        },
        "fitness_distribution": {
            "mean": float(state_df["fitness"].mean()),
            "std": float(state_df["fitness"].std()),
            "min": float(state_df["fitness"].min()),
            "max": float(state_df["fitness"].max()),
            "quartiles": {
                str(k): float(v)
                for k, v in state_df["fitness"].quantile([0.25, 0.5, 0.75]).items()
            },
        },
        "spatial_distribution": {
            "x_range": [
                float(state_df["x_coord"].min()),
                float(state_df["x_coord"].max()),
            ],
            "y_range": [
                float(state_df["y_coord"].min()),
                float(state_df["y_coord"].max()),
            ],
            "z_range": [
                float(state_df["z_coord"].min()),
                float(state_df["z_coord"].max()),
            ],
        },
        "generated_at": datetime.now().isoformat(),
    }

    return summary


@router.post("/batch", response_model=ExportStatus)
async def batch_export(
    batch_request: BatchExportRequest, background_tasks: BackgroundTasks
) -> ExportStatus:
    """Create batch export job for multiple simulations."""
    export_id = str(uuid.uuid4())

    export_status = ExportStatus(
        export_id=export_id, status="pending", progress=0.0, created_at=datetime.now()
    )
    export_jobs[export_id] = export_status

    # Add background task
    background_tasks.add_task(process_batch_export, export_id, batch_request)

    return export_status


# Background task functions
async def process_export(
    export_id: str, simulation_id: str, export_request: ExportRequest
):
    """Process individual export job."""
    try:
        job = export_jobs[export_id]
        job.status = "processing"
        job.progress = 0.1

        if simulation_id not in sample_data:
            job.status = "failed"
            return

        job.progress = 1.0
        job.status = "completed"
        job.download_url = f"/api/v1/export/download/{export_id}"

    except Exception as e:
        logger.error(f"Export job {export_id} failed: {e}")
        export_jobs[export_id].status = "failed"


async def process_batch_export(export_id: str, batch_request: BatchExportRequest):
    """Process batch export job."""
    try:
        job = export_jobs[export_id]
        job.status = "processing"
        job.progress = 1.0
        job.status = "completed"
        job.download_url = f"/api/v1/export/download/{export_id}"

    except Exception as e:
        logger.error(f"Batch export job {export_id} failed: {e}")
        export_jobs[export_id].status = "failed"


def create_sample_export_file(export_id: str) -> StreamingResponse:
    """Create a sample export file for demo purposes."""
    # Create a simple CSV file
    sample_csv = """generation,population_size,mean_fitness,mutation_rate
0,1000,0.5,0.001
1,1020,0.52,0.001
2,1010,0.54,0.001
3,1030,0.56,0.001
4,1025,0.58,0.001"""

    buffer = io.BytesIO(sample_csv.encode())

    return StreamingResponse(
        io.BytesIO(buffer.read()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=export_{export_id}.csv"},
    )
