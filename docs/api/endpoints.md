# API Endpoints Reference

## Table of Contents

- [Simulations](#simulations)
- [State Space](#state-space)
- [Data Export](#data-export)
- [WebSocket](#websocket)

## Simulations

Manage evolutionary simulations and their lifecycle.

### Create Simulation

**POST** `/api/v1/simulations`

Create a new evolutionary simulation.

**Request Body:**
```json
{
  "name": "string",
  "duration": 100,
  "population_size": 1000,
  "mutation_rate": 0.001,
  "environment_params": {
    "altitude": 3000,
    "temperature": -5,
    "oxygen_level": 0.7
  }
}
```

**Response:**
```json
{
  "id": "uuid",
  "name": "string",
  "status": "created",
  "created_at": "2024-01-01T00:00:00Z",
  "parameters": {
    "name": "string",
    "duration": 100,
    "population_size": 1000,
    "mutation_rate": 0.001,
    "environment_params": {}
  }
}
```

### List Simulations

**GET** `/api/v1/simulations`

Retrieve all simulations.

**Query Parameters:**
- `status` (optional): Filter by status (`created`, `running`, `completed`, `failed`)
- `limit` (optional): Maximum number of results (default: 100)
- `offset` (optional): Number of results to skip (default: 0)

**Response:**
```json
[
  {
    "id": "uuid",
    "name": "string",
    "status": "created",
    "created_at": "2024-01-01T00:00:00Z",
    "parameters": {}
  }
]
```

### Get Simulation

**GET** `/api/v1/simulations/{sim_id}`

Retrieve a specific simulation by ID.

**Path Parameters:**
- `sim_id` (string): Simulation UUID

**Response:**
```json
{
  "id": "uuid",
  "name": "string",
  "status": "created",
  "created_at": "2024-01-01T00:00:00Z",
  "parameters": {}
}
```

### Update Simulation

**PUT** `/api/v1/simulations/{sim_id}`

Update simulation parameters.

**Path Parameters:**
- `sim_id` (string): Simulation UUID

**Request Body:**
```json
{
  "name": "Updated Simulation Name",
  "duration": 150,
  "population_size": 1200,
  "mutation_rate": 0.002,
  "environment_params": {}
}
```

### Delete Simulation

**DELETE** `/api/v1/simulations/{sim_id}`

Delete a simulation and all associated data.

**Path Parameters:**
- `sim_id` (string): Simulation UUID

**Response:**
```json
{
  "message": "Simulation {sim_id} deleted"
}
```

## State Space

Manage physiological state space data for simulations.

### Get State Space

**GET** `/api/v1/states/{sim_id}`

Retrieve state space points for a simulation with optional filtering.

**Path Parameters:**
- `sim_id` (string): Simulation UUID

**Query Parameters:**
- `min_fitness` (float, optional): Minimum fitness filter
- `max_fitness` (float, optional): Maximum fitness filter  
- `generation` (int, optional): Generation filter
- `limit` (int, optional): Maximum number of points (default: 100)

**Response:**
```json
[
  {
    "id": "string",
    "coordinates": [1.2, -0.5, 2.1],
    "fitness": 0.75,
    "generation": 42,
    "metadata": {
      "heart_mass": 0.8,
      "lung_capacity": 150,
      "hematocrit": 0.45
    }
  }
]
```

### Add State Point

**POST** `/api/v1/states/{sim_id}`

Add a single state space point to a simulation.

**Path Parameters:**
- `sim_id` (string): Simulation UUID

**Request Body:**
```json
{
  "id": "individual_001",
  "coordinates": [1.2, -0.5, 2.1],
  "fitness": 0.75,
  "generation": 42,
  "metadata": {
    "heart_mass": 0.8,
    "lung_capacity": 150,
    "hematocrit": 0.45
  }
}
```

### Add State Points (Batch)

**POST** `/api/v1/states/{sim_id}/batch`

Add multiple state space points to a simulation.

**Path Parameters:**
- `sim_id` (string): Simulation UUID

**Request Body:**
```json
[
  {
    "id": "individual_001",
    "coordinates": [1.2, -0.5, 2.1],
    "fitness": 0.75,
    "generation": 42,
    "metadata": {}
  },
  {
    "id": "individual_002", 
    "coordinates": [0.8, -0.3, 1.9],
    "fitness": 0.82,
    "generation": 42,
    "metadata": {}
  }
]
```

### Get State Summary

**GET** `/api/v1/states/{sim_id}/summary`

Get statistical summary of the state space.

**Path Parameters:**
- `sim_id` (string): Simulation UUID

**Response:**
```json
{
  "total_points": 1500,
  "fitness_stats": {
    "min": 0.12,
    "max": 0.95,
    "avg": 0.67
  },
  "generation_range": {
    "min": 0,
    "max": 99
  }
}
```

### Clear State Space

**DELETE** `/api/v1/states/{sim_id}`

Clear all state space points for a simulation.

**Path Parameters:**
- `sim_id` (string): Simulation UUID

## Data Export

Export simulation data in various formats and generate publication figures.

### Export Simulation Data

**POST** `/api/v1/export/simulation/{simulation_id}`

Export simulation data in specified format.

**Path Parameters:**
- `simulation_id` (string): Simulation UUID

**Request Body:**
```json
{
  "simulation_id": "uuid",
  "format": "csv",
  "data_types": ["simulation_data", "state_space", "statistics"],
  "filters": {
    "min_generation": 10,
    "max_generation": 90
  }
}
```

**Response:**
```json
{
  "export_id": "uuid",
  "status": "pending",
  "progress": 0.0,
  "download_url": null,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Get Export Status

**GET** `/api/v1/export/status/{export_id}`

Check the status of an export job.

**Path Parameters:**
- `export_id` (string): Export job UUID

**Response:**
```json
{
  "export_id": "uuid",
  "status": "completed",
  "progress": 1.0,
  "download_url": "/api/v1/export/download/uuid",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### Download Export

**GET** `/api/v1/export/download/{export_id}`

Download the exported file.

**Path Parameters:**
- `export_id` (string): Export job UUID

**Response:** File download (CSV, JSON, or HDF5)

### Generate Publication Figure

**POST** `/api/v1/export/figure/{simulation_id}`

Generate publication-ready figures.

**Path Parameters:**
- `simulation_id` (string): Simulation UUID

**Request Body:**
```json
{
  "simulation_id": "uuid",
  "figure_type": "state_space",
  "format": "png",
  "width": 12,
  "height": 8,
  "dpi": 300
}
```

**Response:** Image file (PNG, PDF, or SVG)

**Available Figure Types:**
- `state_space`: 3D state space visualization
- `evolution_timeline`: Fitness evolution over time
- `fitness_landscape`: Fitness landscape heatmap

### Get Statistical Summary

**GET** `/api/v1/export/summary/{simulation_id}`

Get comprehensive statistical summary for a simulation.

**Path Parameters:**
- `simulation_id` (string): Simulation UUID

**Response:**
```json
{
  "simulation_id": "uuid",
  "basic_stats": {
    "total_generations": 100,
    "final_population": 1023,
    "max_fitness": 0.95,
    "mean_fitness": 0.71,
    "evolution_rate": 0.034
  },
  "population_stats": {
    "mean_population": 1015.5,
    "std_population": 25.3,
    "min_population": 980,
    "max_population": 1050
  },
  "fitness_distribution": {
    "mean": 0.71,
    "std": 0.15,
    "min": 0.12,
    "max": 0.95,
    "quartiles": {
      "0.25": 0.62,
      "0.5": 0.70,
      "0.75": 0.82
    }
  },
  "spatial_distribution": {
    "x_range": [-2.1, 3.4],
    "y_range": [-1.8, 2.7],
    "z_range": [-1.5, 3.1]
  },
  "generated_at": "2024-01-01T00:00:00Z"
}
```

### Batch Export

**POST** `/api/v1/export/batch`

Create batch export job for multiple simulations.

**Request Body:**
```json
{
  "simulation_ids": ["uuid1", "uuid2", "uuid3"],
  "formats": ["csv", "json"],
  "include_figures": true
}
```

## WebSocket

Real-time updates for running simulations.

### Connect to Simulation Updates

**WebSocket** `/ws/simulation/{sim_id}`

Connect to real-time simulation updates.

**Path Parameters:**
- `sim_id` (string): Simulation UUID

**Message Types:**

**Status Update:**
```json
{
  "type": "status",
  "data": {
    "simulation_id": "uuid",
    "status": "running",
    "generation": 42,
    "progress": 0.42
  }
}
```

**State Point Update:**
```json
{
  "type": "state_point",
  "data": {
    "simulation_id": "uuid",
    "point": {
      "id": "individual_001",
      "coordinates": [1.2, -0.5, 2.1],
      "fitness": 0.75,
      "generation": 42
    }
  }
}
```

**Error:**
```json
{
  "type": "error",
  "data": {
    "message": "Simulation not found",
    "code": 404
  }
}
```

## Status Codes Summary

| Code | Description | Usage |
|------|-------------|-------|
| 200 | OK | Successful GET, PUT, DELETE |
| 201 | Created | Successful POST |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Missing/invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 422 | Validation Error | Request data validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |