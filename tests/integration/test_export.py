"""
Tests for API-003: Data Export Endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
import io
import json
from pathlib import Path

# Add the project root to the path so we can import from ipe
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ipe.api.main import app

client = TestClient(app)

def test_statistical_summary():
    """Test statistical summary endpoint."""
    response = client.get("/api/v1/export/summary/demo-sim-001")
    assert response.status_code == 200
    
    data = response.json()
    assert "simulation_id" in data
    assert data["simulation_id"] == "demo-sim-001"
    assert "basic_stats" in data
    assert "population_stats" in data
    assert "fitness_distribution" in data
    assert "spatial_distribution" in data

def test_export_simulation_data():
    """Test exporting simulation data."""
    export_request = {
        "simulation_id": "demo-sim-001",
        "format": "json",
        "data_types": ["simulation_data", "statistics"]
    }
    
    response = client.post("/api/v1/export/simulation/demo-sim-001", json=export_request)
    assert response.status_code == 200
    
    data = response.json()
    assert "export_id" in data
    assert data["status"] == "pending"
    assert data["progress"] == 0.0

def test_export_status():
    """Test export status endpoint."""
    # First create an export job
    export_request = {
        "simulation_id": "demo-sim-001",
        "format": "csv",
        "data_types": ["simulation_data"]
    }
    
    create_response = client.post("/api/v1/export/simulation/demo-sim-001", json=export_request)
    assert create_response.status_code == 200
    export_id = create_response.json()["export_id"]
    
    # Check status
    response = client.get(f"/api/v1/export/status/{export_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["export_id"] == export_id
    assert "status" in data

def test_export_status_not_found():
    """Test export status for non-existent job."""
    response = client.get("/api/v1/export/status/non-existent-id")
    assert response.status_code == 404

def test_generate_publication_figure_state_space():
    """Test publication figure generation - state space."""
    figure_request = {
        "simulation_id": "demo-sim-001",
        "figure_type": "state_space",
        "format": "png",
        "width": 10,
        "height": 8
    }
    
    response = client.post("/api/v1/export/figure/demo-sim-001", json=figure_request)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_generate_publication_figure_evolution():
    """Test publication figure generation - evolution timeline."""
    figure_request = {
        "simulation_id": "demo-sim-001",
        "figure_type": "evolution_timeline",
        "format": "png"
    }
    
    response = client.post("/api/v1/export/figure/demo-sim-001", json=figure_request)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_generate_publication_figure_fitness_landscape():
    """Test publication figure generation - fitness landscape."""
    figure_request = {
        "simulation_id": "demo-sim-001",
        "figure_type": "fitness_landscape",
        "format": "png"
    }
    
    response = client.post("/api/v1/export/figure/demo-sim-001", json=figure_request)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

def test_generate_figure_unknown_type():
    """Test publication figure generation with unknown type."""
    figure_request = {
        "simulation_id": "demo-sim-001",
        "figure_type": "unknown_type",
        "format": "png"
    }
    
    response = client.post("/api/v1/export/figure/demo-sim-001", json=figure_request)
    assert response.status_code == 400

def test_generate_figure_simulation_not_found():
    """Test publication figure generation for non-existent simulation."""
    figure_request = {
        "simulation_id": "non-existent-sim",
        "figure_type": "state_space",
        "format": "png"
    }
    
    response = client.post("/api/v1/export/figure/non-existent-sim", json=figure_request)
    assert response.status_code == 404

def test_batch_export():
    """Test batch export functionality."""
    batch_request = {
        "simulation_ids": ["demo-sim-001"],
        "formats": ["json", "csv"],
        "include_figures": False
    }
    
    response = client.post("/api/v1/export/batch", json=batch_request)
    assert response.status_code == 200
    
    data = response.json()
    assert "export_id" in data
    assert data["status"] == "pending"

def test_download_export():
    """Test download export endpoint."""
    # First create an export
    export_request = {
        "simulation_id": "demo-sim-001",
        "format": "csv",
        "data_types": ["simulation_data"]
    }
    
    create_response = client.post("/api/v1/export/simulation/demo-sim-001", json=export_request)
    export_id = create_response.json()["export_id"]
    
    # Try to download (might not be ready immediately due to background processing)
    response = client.get(f"/api/v1/export/download/{export_id}")
    # Could be 400 (not ready) or 200 (ready) depending on timing
    assert response.status_code in [200, 400]

def test_download_export_not_found():
    """Test download for non-existent export."""
    response = client.get("/api/v1/export/download/non-existent-id")
    assert response.status_code == 404