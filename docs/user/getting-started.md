# Getting Started with IPE

Welcome to the Interactionist Phylogeny Engine (IPE) - a comprehensive platform for simulating evolutionary processes in physiological state spaces with environmental interactions.

## Overview

IPE allows researchers to:

- **Simulate Evolution**: Run complex evolutionary simulations with physiological constraints
- **Explore State Spaces**: Navigate multi-dimensional physiological trait spaces
- **Analyze Adaptation**: Study how organisms adapt to environmental challenges like altitude
- **Generate Insights**: Export data and create publication-ready visualizations

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows (WSL2 recommended)
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for large simulations
- **Storage**: 10GB+ free space
- **CPU**: Multi-core processor recommended (4+ cores)

### Software Dependencies

- **Docker** (20.10+) and **Docker Compose** (2.0+)
- **Python** (3.9+) for client scripts
- **Web Browser** (Chrome, Firefox, Safari, Edge - latest versions)

## Installation

### Method 1: Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mnechromancer/RIPE.git
   cd RIPE
   ```

2. **Start the services**:
   ```bash
   docker-compose up -d
   ```

3. **Verify installation**:
   ```bash
   curl http://localhost:8000/docs
   ```

### Method 2: Local Development

1. **Clone and setup**:
   ```bash
   git clone https://github.com/mnechromancer/RIPE.git
   cd RIPE
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Start the database**:
   ```bash
   docker-compose up -d postgres redis
   ```

3. **Run migrations**:
   ```bash
   alembic upgrade head
   ```

4. **Start the API server**:
   ```bash
   uvicorn ipe.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Start the web interface**:
   ```bash
   cd web
   npm install
   npm start
   ```

## First Steps

### 1. Verify Installation

Open your browser and navigate to:

- **API Documentation**: http://localhost:8000/docs
- **Web Interface**: http://localhost:3000 (if running locally)
- **API Health Check**: http://localhost:8000/health

### 2. Create Your First Simulation

#### Using the Web Interface

1. Navigate to http://localhost:3000
2. Click "New Simulation"
3. Configure parameters:
   - **Name**: "My First Simulation"
   - **Duration**: 100 generations
   - **Population Size**: 1000 individuals
   - **Environment**: Altitude = 3000m, Temperature = -5°C
4. Click "Start Simulation"

#### Using the API

```bash
curl -X POST "http://localhost:8000/api/v1/simulations" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My First Simulation",
    "duration": 100,
    "population_size": 1000,
    "mutation_rate": 0.001,
    "environment_params": {
      "altitude": 3000,
      "temperature": -5,
      "oxygen_level": 0.7
    }
  }'
```

#### Using Python

```python
import requests

# Create simulation
response = requests.post("http://localhost:8000/api/v1/simulations", json={
    "name": "My First Simulation",
    "duration": 100,
    "population_size": 1000,
    "environment_params": {"altitude": 3000}
})

simulation = response.json()
print(f"Created simulation: {simulation['id']}")
```

### 3. Monitor Progress

#### Real-time Monitoring

Use WebSocket connection to monitor simulation progress:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/simulation/YOUR_SIMULATION_ID');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'status') {
        console.log(`Generation ${data.data.generation}: ${data.data.progress * 100}% complete`);
    }
};
```

#### Check Status via API

```bash
curl "http://localhost:8000/api/v1/simulations/YOUR_SIMULATION_ID"
```

### 4. Explore Results

#### View State Space

```bash
curl "http://localhost:8000/api/v1/states/YOUR_SIMULATION_ID?limit=100"
```

#### Generate Visualizations

```bash
curl -X POST "http://localhost:8000/api/v1/export/figure/YOUR_SIMULATION_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "figure_type": "state_space",
    "format": "png",
    "dpi": 300
  }' \
  --output my_state_space.png
```

## Key Concepts

### Physiological State Space

IPE models organisms as points in a multi-dimensional physiological state space where each dimension represents a trait (e.g., heart mass, lung capacity, hematocrit).

**Example State Vector:**
```
[heart_mass: 0.8, lung_capacity: 150, hematocrit: 0.45, muscle_density: 1.2]
```

### Environmental Challenges

Organisms face environmental pressures that affect their fitness:

- **Altitude**: Affects oxygen availability and pressure
- **Temperature**: Influences metabolic rates and energy requirements  
- **Resource Availability**: Limits population growth and competition

### Evolutionary Dynamics

The system simulates:

- **Natural Selection**: Fitness-based survival and reproduction
- **Mutation**: Random changes in physiological traits
- **Gene Flow**: Movement between populations
- **Genetic Drift**: Random sampling effects

## Basic Workflows

### Workflow 1: Single Environment Study

1. **Setup**: Create simulation with specific environmental parameters
2. **Run**: Let simulation evolve for 100-500 generations
3. **Analyze**: Examine final trait distributions and fitness landscapes
4. **Export**: Generate figures and data for publication

### Workflow 2: Environmental Gradient Analysis

1. **Create Multiple Simulations**: Different altitude levels (0m, 1000m, 2000m, 3000m, 4000m)
2. **Run in Parallel**: Monitor convergence across environments
3. **Compare**: Analyze how traits change across the gradient
4. **Visualize**: Create comparative plots showing adaptation patterns

### Workflow 3: Plasticity vs. Evolution

1. **Configure Parameters**: Set plasticity vs. genetic response ratios
2. **Run Short-term**: Observe immediate plastic responses
3. **Run Long-term**: Track genetic changes over many generations  
4. **Compare**: Distinguish plastic vs. evolutionary responses

## Configuration Guide

### Simulation Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `duration` | Number of generations | 100 | 10-10000 |
| `population_size` | Individuals per generation | 1000 | 100-10000 |
| `mutation_rate` | Per-trait mutation probability | 0.001 | 0.0001-0.1 |
| `selection_strength` | Intensity of natural selection | 1.0 | 0.1-10.0 |

### Environment Parameters

| Parameter | Description | Units | Typical Range |
|-----------|-------------|-------|---------------|
| `altitude` | Elevation above sea level | meters | 0-6000 |
| `temperature` | Mean temperature | °C | -40 to +50 |
| `oxygen_level` | Relative oxygen availability | fraction | 0.3-1.0 |
| `resource_availability` | Food/energy resources | relative | 0.1-2.0 |

### Performance Tuning

For large simulations:

```yaml
# docker-compose.override.yml
services:
  ipe-api:
    environment:
      - WORKERS=4
      - MAX_POPULATION_SIZE=50000
      - ENABLE_GPU=true
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

## Troubleshooting

### Common Issues

#### "Connection refused" Error

```bash
# Check if services are running
docker-compose ps

# Restart services
docker-compose restart

# Check logs
docker-compose logs ipe-api
```

#### Slow Simulation Performance

1. **Reduce Population Size**: Use 500-1000 for initial testing
2. **Decrease Duration**: Start with 50-100 generations
3. **Enable GPU**: Set `ENABLE_GPU=true` if available
4. **Increase Resources**: Allocate more CPU cores and memory

#### Out of Memory Errors

```bash
# Monitor resource usage
docker stats

# Increase memory limits in docker-compose.yml
services:
  ipe-api:
    deploy:
      resources:
        limits:
          memory: 16G
```

### Getting Help

1. **Check the FAQ**: [FAQ](faq.md)
2. **API Documentation**: http://localhost:8000/docs
3. **GitHub Issues**: https://github.com/mnechromancer/RIPE/issues
4. **Community Forum**: (coming soon)

## Next Steps

Once you're comfortable with the basics:

1. **Explore Tutorials**: [tutorials/](tutorials/) for specific use cases
2. **Advanced Workflows**: [workflows/](workflows/) for complex analyses  
3. **API Integration**: [API Examples](../api/examples.md) for custom scripts
4. **Scientific Background**: [Science Documentation](../science/algorithms.md)

## Quick Reference

### Essential Commands

```bash
# Start system
docker-compose up -d

# Stop system  
docker-compose down

# View logs
docker-compose logs -f

# Update system
git pull && docker-compose build && docker-compose up -d

# Backup data
docker-compose exec postgres pg_dump -U ipe > backup.sql
```

### Key URLs

- **Web Interface**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs  
- **Health Check**: http://localhost:8000/health
- **Monitoring**: http://localhost:3001 (if monitoring enabled)

Welcome to IPE! Start with a simple simulation and gradually explore the advanced features as you become more familiar with the system.