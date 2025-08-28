# API Code Examples

This document provides comprehensive code examples for integrating with the IPE API in different programming languages.

## Table of Contents

- [Python Examples](#python-examples)
- [JavaScript Examples](#javascript-examples)
- [R Examples](#r-examples)
- [cURL Examples](#curl-examples)
- [Authentication Examples](#authentication-examples)
- [Complete Workflow Examples](#complete-workflow-examples)

## Python Examples

### Basic Setup

```python
import requests
import json
from typing import Dict, List, Optional

class IPEClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def create_simulation(self, name: str, duration: int = 100, 
                         population_size: int = 1000, mutation_rate: float = 0.001,
                         environment_params: Optional[Dict] = None) -> Dict:
        """Create a new simulation."""
        data = {
            "name": name,
            "duration": duration,
            "population_size": population_size,
            "mutation_rate": mutation_rate,
            "environment_params": environment_params or {}
        }
        
        response = self.session.post(f"{self.base_url}/api/v1/simulations", json=data)
        response.raise_for_status()
        return response.json()

    def get_simulation(self, sim_id: str) -> Dict:
        """Get simulation details."""
        response = self.session.get(f"{self.base_url}/api/v1/simulations/{sim_id}")
        response.raise_for_status()
        return response.json()

    def list_simulations(self, status: Optional[str] = None) -> List[Dict]:
        """List all simulations."""
        params = {}
        if status:
            params["status"] = status
            
        response = self.session.get(f"{self.base_url}/api/v1/simulations", params=params)
        response.raise_for_status()
        return response.json()

    def get_state_space(self, sim_id: str, min_fitness: Optional[float] = None,
                       max_fitness: Optional[float] = None, generation: Optional[int] = None,
                       limit: int = 100) -> List[Dict]:
        """Get state space points for a simulation."""
        params = {"limit": limit}
        if min_fitness is not None:
            params["min_fitness"] = min_fitness
        if max_fitness is not None:
            params["max_fitness"] = max_fitness
        if generation is not None:
            params["generation"] = generation
            
        response = self.session.get(f"{self.base_url}/api/v1/states/{sim_id}", params=params)
        response.raise_for_status()
        return response.json()

    def export_simulation(self, sim_id: str, format_type: str = "csv",
                         data_types: Optional[List[str]] = None) -> Dict:
        """Export simulation data."""
        data = {
            "simulation_id": sim_id,
            "format": format_type,
            "data_types": data_types or ["simulation_data", "state_space", "statistics"]
        }
        
        response = self.session.post(f"{self.base_url}/api/v1/export/simulation/{sim_id}", json=data)
        response.raise_for_status()
        return response.json()

    def download_export(self, export_id: str, filename: str):
        """Download exported file."""
        response = self.session.get(f"{self.base_url}/api/v1/export/download/{export_id}")
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)

# Usage example
client = IPEClient("http://localhost:8000")

# Create a simulation
simulation = client.create_simulation(
    name="Altitude Adaptation Study",
    duration=200,
    environment_params={"altitude": 3000, "oxygen_level": 0.7}
)
sim_id = simulation["id"]

# Get state space data
state_points = client.get_state_space(sim_id, min_fitness=0.5)

# Export data
export_job = client.export_simulation(sim_id, "csv")
print(f"Export job created: {export_job['export_id']}")
```

### Advanced Example with WebSocket

```python
import asyncio
import websockets
import json

async def monitor_simulation(sim_id: str):
    """Monitor simulation in real-time via WebSocket."""
    uri = f"ws://localhost:8000/ws/simulation/{sim_id}"
    
    async with websockets.connect(uri) as websocket:
        print(f"Connected to simulation {sim_id}")
        
        try:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "status":
                    print(f"Generation {data['data']['generation']}: {data['data']['progress']*100:.1f}%")
                elif data["type"] == "state_point":
                    point = data["data"]["point"]
                    print(f"New point: fitness={point['fitness']:.3f}")
                elif data["type"] == "error":
                    print(f"Error: {data['data']['message']}")
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")

# Run the monitor
# asyncio.run(monitor_simulation("your-sim-id"))
```

### Data Analysis Example

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_simulation_results(client: IPEClient, sim_id: str):
    """Comprehensive analysis of simulation results."""
    
    # Get statistical summary
    summary_response = client.session.get(f"{client.base_url}/api/v1/export/summary/{sim_id}")
    summary = summary_response.json()
    
    print("Simulation Summary:")
    print(f"- Total generations: {summary['basic_stats']['total_generations']}")
    print(f"- Final population: {summary['basic_stats']['final_population']}")
    print(f"- Max fitness: {summary['basic_stats']['max_fitness']:.3f}")
    print(f"- Mean fitness: {summary['basic_stats']['mean_fitness']:.3f}")
    
    # Get state space data
    state_points = client.get_state_space(sim_id, limit=1000)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(state_points)
    
    # Plot fitness distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(df['fitness'], bins=30, alpha=0.7)
    plt.xlabel('Fitness')
    plt.ylabel('Frequency')
    plt.title('Fitness Distribution')
    
    plt.subplot(1, 3, 2)
    plt.scatter(df['generation'], df['fitness'], alpha=0.6)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness vs Generation')
    
    plt.subplot(1, 3, 3)
    # Assuming 3D coordinates
    coords = np.array(df['coordinates'].tolist())
    plt.scatter(coords[:, 0], coords[:, 1], c=df['fitness'], cmap='viridis', alpha=0.6)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('State Space (2D projection)')
    plt.colorbar(label='Fitness')
    
    plt.tight_layout()
    plt.savefig(f'simulation_analysis_{sim_id}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df, summary

# Usage
# df, summary = analyze_simulation_results(client, sim_id)
```

## JavaScript Examples

### Node.js Client

```javascript
const axios = require('axios');
const WebSocket = require('ws');

class IPEClient {
    constructor(baseUrl = 'http://localhost:8000', apiKey = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiKey = apiKey;
        
        this.client = axios.create({
            baseURL: this.baseUrl,
            headers: apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}
        });
    }

    async createSimulation(params) {
        const response = await this.client.post('/api/v1/simulations', params);
        return response.data;
    }

    async getSimulation(simId) {
        const response = await this.client.get(`/api/v1/simulations/${simId}`);
        return response.data;
    }

    async listSimulations(status = null) {
        const params = status ? { status } : {};
        const response = await this.client.get('/api/v1/simulations', { params });
        return response.data;
    }

    async getStateSpace(simId, filters = {}) {
        const response = await this.client.get(`/api/v1/states/${simId}`, { params: filters });
        return response.data;
    }

    async exportSimulation(simId, exportParams) {
        const response = await this.client.post(`/api/v1/export/simulation/${simId}`, exportParams);
        return response.data;
    }

    connectWebSocket(simId) {
        const ws = new WebSocket(`ws://${this.baseUrl.replace('http://', '')}/ws/simulation/${simId}`);
        
        ws.on('open', () => {
            console.log(`Connected to simulation ${simId}`);
        });

        ws.on('message', (data) => {
            const message = JSON.parse(data);
            console.log('Received:', message);
        });

        ws.on('close', () => {
            console.log('WebSocket connection closed');
        });

        return ws;
    }
}

// Usage example
async function runExample() {
    const client = new IPEClient();
    
    try {
        // Create simulation
        const simulation = await client.createSimulation({
            name: "JavaScript Test",
            duration: 100,
            population_size: 1000,
            environment_params: { altitude: 2500 }
        });
        
        console.log('Created simulation:', simulation.id);
        
        // Monitor via WebSocket
        const ws = client.connectWebSocket(simulation.id);
        
        // Get state space after some time
        setTimeout(async () => {
            const statePoints = await client.getStateSpace(simulation.id, { limit: 10 });
            console.log(`Retrieved ${statePoints.length} state points`);
        }, 5000);
        
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

// runExample();
```

### Browser JavaScript (Fetch API)

```javascript
class IPEWebClient {
    constructor(baseUrl = window.location.origin, apiKey = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiKey = apiKey;
    }

    async request(endpoint, options = {}) {
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };

        if (this.apiKey) {
            headers.Authorization = `Bearer ${this.apiKey}`;
        }

        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            ...options,
            headers
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json();
    }

    async createSimulation(params) {
        return this.request('/api/v1/simulations', {
            method: 'POST',
            body: JSON.stringify(params)
        });
    }

    async getSimulations() {
        return this.request('/api/v1/simulations');
    }

    async exportAsCSV(simId) {
        const exportJob = await this.request(`/api/v1/export/simulation/${simId}`, {
            method: 'POST',
            body: JSON.stringify({
                simulation_id: simId,
                format: 'csv',
                data_types: ['simulation_data']
            })
        });

        // Poll for completion
        while (exportJob.status !== 'completed') {
            await new Promise(resolve => setTimeout(resolve, 1000));
            const status = await this.request(`/api/v1/export/status/${exportJob.export_id}`);
            if (status.status === 'completed') {
                // Trigger download
                window.open(`${this.baseUrl}/api/v1/export/download/${exportJob.export_id}`);
                break;
            }
        }
    }
}

// Usage in browser
const client = new IPEWebClient();

document.getElementById('create-sim-btn').addEventListener('click', async () => {
    const simulation = await client.createSimulation({
        name: 'Browser Test',
        duration: 50
    });
    
    console.log('Simulation created:', simulation);
    
    // Update UI
    document.getElementById('sim-id').textContent = simulation.id;
});
```

## R Examples

### Basic R Client

```r
library(httr)
library(jsonlite)

IPEClient <- function(base_url = "http://localhost:8000", api_key = NULL) {
  list(
    base_url = gsub("/$", "", base_url),
    api_key = api_key,
    
    create_simulation = function(name, duration = 100, population_size = 1000, 
                                mutation_rate = 0.001, environment_params = list()) {
      body <- list(
        name = name,
        duration = duration,
        population_size = population_size,
        mutation_rate = mutation_rate,
        environment_params = environment_params
      )
      
      headers <- list("Content-Type" = "application/json")
      if (!is.null(api_key)) {
        headers$Authorization <- paste("Bearer", api_key)
      }
      
      response <- POST(
        url = paste0(base_url, "/api/v1/simulations"),
        body = toJSON(body, auto_unbox = TRUE),
        add_headers(.headers = headers)
      )
      
      stop_for_status(response)
      fromJSON(content(response, "text"))
    },
    
    get_state_space = function(sim_id, min_fitness = NULL, max_fitness = NULL, 
                              generation = NULL, limit = 100) {
      query <- list(limit = limit)
      if (!is.null(min_fitness)) query$min_fitness <- min_fitness
      if (!is.null(max_fitness)) query$max_fitness <- max_fitness
      if (!is.null(generation)) query$generation <- generation
      
      response <- GET(
        url = paste0(base_url, "/api/v1/states/", sim_id),
        query = query
      )
      
      stop_for_status(response)
      fromJSON(content(response, "text"))
    },
    
    export_simulation = function(sim_id, format = "csv", data_types = c("simulation_data")) {
      body <- list(
        simulation_id = sim_id,
        format = format,
        data_types = data_types
      )
      
      response <- POST(
        url = paste0(base_url, "/api/v1/export/simulation/", sim_id),
        body = toJSON(body, auto_unbox = TRUE),
        add_headers("Content-Type" = "application/json")
      )
      
      stop_for_status(response)
      fromJSON(content(response, "text"))
    }
  )
}

# Usage example
client <- IPEClient()

# Create simulation
sim <- client$create_simulation(
  name = "R Test Simulation",
  duration = 150,
  environment_params = list(altitude = 3500, temperature = -10)
)

cat("Created simulation:", sim$id, "\n")

# Get state space data
state_data <- client$get_state_space(sim$id, min_fitness = 0.6)
cat("Retrieved", length(state_data), "state points\n")

# Convert to data frame for analysis
if (length(state_data) > 0) {
  df <- data.frame(
    id = sapply(state_data, `[[`, "id"),
    fitness = sapply(state_data, `[[`, "fitness"),
    generation = sapply(state_data, `[[`, "generation"),
    stringsAsFactors = FALSE
  )
  
  # Basic analysis
  summary(df$fitness)
  hist(df$fitness, main = "Fitness Distribution", xlab = "Fitness")
}
```

## cURL Examples

### Basic Operations

```bash
# Create simulation
curl -X POST "http://localhost:8000/api/v1/simulations" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "cURL Test",
    "duration": 100,
    "population_size": 1000,
    "environment_params": {"altitude": 3000}
  }'

# List simulations
curl "http://localhost:8000/api/v1/simulations"

# Get specific simulation
curl "http://localhost:8000/api/v1/simulations/{simulation_id}"

# Get state space with filters
curl "http://localhost:8000/api/v1/states/{simulation_id}?min_fitness=0.7&limit=50"

# Export simulation data
curl -X POST "http://localhost:8000/api/v1/export/simulation/{simulation_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "{simulation_id}",
    "format": "csv",
    "data_types": ["simulation_data", "statistics"]
  }'

# Download export
curl -O "http://localhost:8000/api/v1/export/download/{export_id}"

# Generate publication figure
curl -X POST "http://localhost:8000/api/v1/export/figure/{simulation_id}" \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "{simulation_id}",
    "figure_type": "state_space",
    "format": "png",
    "dpi": 300
  }' \
  --output state_space_plot.png
```

## Authentication Examples

### API Key Authentication

```python
# Python with API key
client = IPEClient("https://api.ipe.velottalab.com", api_key="your-api-key-here")

# JavaScript with API key
const client = new IPEClient('https://api.ipe.velottalab.com', 'your-api-key-here');

# cURL with API key
curl -H "Authorization: Bearer your-api-key-here" \
     "https://api.ipe.velottalab.com/api/v1/simulations"
```

### OAuth2 Flow (Future Implementation)

```python
# OAuth2 example (placeholder for future implementation)
from requests_oauthlib import OAuth2Session

def authenticate_oauth2(client_id, client_secret, redirect_uri):
    oauth = OAuth2Session(client_id, redirect_uri=redirect_uri)
    
    # Get authorization URL
    authorization_url, state = oauth.authorization_url(
        'https://auth.ipe.velottalab.com/oauth/authorize'
    )
    
    print(f'Please visit: {authorization_url}')
    
    # After user authorization, exchange code for token
    authorization_response = input('Enter the full callback URL: ')
    
    token = oauth.fetch_token(
        'https://auth.ipe.velottalab.com/oauth/token',
        authorization_response=authorization_response,
        client_secret=client_secret
    )
    
    return token['access_token']
```

## Complete Workflow Examples

### Full Simulation Workflow

```python
def complete_simulation_workflow():
    """Complete workflow from creation to analysis."""
    client = IPEClient()
    
    # Step 1: Create simulation
    print("Creating simulation...")
    simulation = client.create_simulation(
        name="Complete Workflow Demo",
        duration=200,
        population_size=1500,
        mutation_rate=0.002,
        environment_params={
            "altitude": 4000,
            "temperature": -15,
            "oxygen_level": 0.6
        }
    )
    sim_id = simulation["id"]
    print(f"Created simulation: {sim_id}")
    
    # Step 2: Monitor progress (placeholder - would use WebSocket in real implementation)
    print("Simulation running... (monitoring)")
    
    # Step 3: Get results
    print("Retrieving results...")
    state_points = client.get_state_space(sim_id, limit=1000)
    print(f"Retrieved {len(state_points)} state points")
    
    # Step 4: Generate publication figures
    print("Generating figures...")
    # This would save figures to files
    figure_types = ["state_space", "evolution_timeline", "fitness_landscape"]
    for fig_type in figure_types:
        response = client.session.post(
            f"{client.base_url}/api/v1/export/figure/{sim_id}",
            json={
                "simulation_id": sim_id,
                "figure_type": fig_type,
                "format": "png",
                "dpi": 300
            }
        )
        with open(f"{fig_type}_{sim_id}.png", "wb") as f:
            f.write(response.content)
        print(f"Saved {fig_type}_{sim_id}.png")
    
    # Step 5: Export data
    print("Exporting data...")
    export_job = client.export_simulation(sim_id, "csv", 
                                         ["simulation_data", "state_space", "statistics"])
    export_id = export_job["export_id"]
    
    # Poll for completion (simplified)
    import time
    while True:
        status_response = client.session.get(f"{client.base_url}/api/v1/export/status/{export_id}")
        status = status_response.json()
        if status["status"] == "completed":
            break
        time.sleep(1)
    
    # Download
    client.download_export(export_id, f"simulation_data_{sim_id}.csv")
    print(f"Downloaded simulation_data_{sim_id}.csv")
    
    print("Workflow complete!")
    return sim_id

# Run the complete workflow
# sim_id = complete_simulation_workflow()
```

### Batch Processing Example

```python
def batch_simulation_analysis(simulation_ids):
    """Analyze multiple simulations and generate comparative report."""
    client = IPEClient()
    results = {}
    
    for sim_id in simulation_ids:
        print(f"Analyzing simulation {sim_id}...")
        
        # Get summary statistics
        summary_response = client.session.get(f"{client.base_url}/api/v1/export/summary/{sim_id}")
        summary = summary_response.json()
        
        # Get high-fitness individuals
        high_fitness = client.get_state_space(sim_id, min_fitness=0.8, limit=100)
        
        results[sim_id] = {
            "summary": summary,
            "high_fitness_count": len(high_fitness),
            "top_fitness": max([p["fitness"] for p in high_fitness]) if high_fitness else 0
        }
    
    # Generate comparison report
    print("\n=== Batch Analysis Report ===")
    for sim_id, data in results.items():
        stats = data["summary"]["basic_stats"]
        print(f"\nSimulation {sim_id}:")
        print(f"  - Generations: {stats['total_generations']}")
        print(f"  - Max fitness: {stats['max_fitness']:.3f}")
        print(f"  - High-fitness individuals: {data['high_fitness_count']}")
        print(f"  - Top fitness: {data['top_fitness']:.3f}")
    
    return results

# Usage
# simulation_ids = ["sim1", "sim2", "sim3"]
# batch_results = batch_simulation_analysis(simulation_ids)
```

This documentation provides comprehensive examples for integrating with the IPE API across multiple programming languages and use cases. Each example is designed to be practical and immediately usable for researchers and developers working with the system.