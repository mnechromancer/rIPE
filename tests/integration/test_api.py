"""
Integration tests for API-001: Core REST API
"""

from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add the project root to the path so we can import from ipe
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ipe.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "IPE API"
    assert data["version"] == "1.0.0"


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_create_simulation():
    """Test creating a simulation."""
    simulation_data = {
        "name": "Test Simulation",
        "duration": 50,
        "population_size": 500,
        "mutation_rate": 0.01,
    }

    response = client.post("/api/v1/simulations", json=simulation_data)

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Simulation"
    assert data["status"] == "created"
    assert "id" in data


def test_list_simulations():
    """Test listing simulations."""
    response = client.get("/api/v1/simulations")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_get_simulation():
    """Test getting a specific simulation."""
    # First create a simulation
    simulation_data = {"name": "Test Get Simulation", "duration": 30}

    create_response = client.post("/api/v1/simulations", json=simulation_data)
    assert create_response.status_code == 200
    sim_id = create_response.json()["id"]

    # Now get it
    response = client.get(f"/api/v1/simulations/{sim_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == sim_id
    assert data["name"] == "Test Get Simulation"


def test_get_nonexistent_simulation():
    """Test getting a simulation that doesn't exist."""
    response = client.get("/api/v1/simulations/nonexistent-id")
    assert response.status_code == 404


def test_state_space_endpoints():
    """Test state space endpoints."""
    sim_id = "test-sim-123"

    # Test getting states (should return empty list initially)
    response = client.get(f"/api/v1/states/{sim_id}")
    assert response.status_code == 200
    assert response.json() == []

    # Test adding a state point
    state_point = {
        "id": "state-1",
        "coordinates": [1.0, 2.0, 3.0],
        "fitness": 0.85,
        "generation": 1,
    }

    response = client.post(f"/api/v1/states/{sim_id}", json=state_point)
    assert response.status_code == 200

    # Test getting states again (should have one point now)
    response = client.get(f"/api/v1/states/{sim_id}")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == "state-1"
    assert data[0]["fitness"] == 0.85


def test_state_summary():
    """Test state space summary endpoint."""
    sim_id = "test-sim-summary"

    # Initially should have no points
    response = client.get(f"/api/v1/states/{sim_id}/summary")
    assert response.status_code == 200
    assert response.json()["total_points"] == 0

    # Add some state points
    state_points = [
        {"id": "state-1", "coordinates": [1.0, 2.0], "fitness": 0.8, "generation": 1},
        {"id": "state-2", "coordinates": [2.0, 3.0], "fitness": 0.9, "generation": 2},
    ]

    response = client.post(f"/api/v1/states/{sim_id}/batch", json=state_points)
    assert response.status_code == 200

    # Check summary
    response = client.get(f"/api/v1/states/{sim_id}/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_points"] == 2
    assert data["fitness_stats"]["min"] == 0.8
    assert data["fitness_stats"]["max"] == 0.9


def test_openapi_docs():
    """Test that OpenAPI docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi_spec = response.json()
    assert openapi_spec["info"]["title"] == "IPE API"
