# Tutorial: Altitude Adaptation Study

This tutorial demonstrates how to simulate evolutionary adaptation to high altitude using IPE, mimicking real-world studies of organisms adapting to hypoxic environments.

## Learning Objectives

By the end of this tutorial, you will:

- Understand how to set up altitude adaptation simulations
- Know how to analyze physiological trait evolution
- Be able to generate publication-ready visualizations
- Understand the relationship between altitude and physiological adaptations

## Background

High altitude environments present significant physiological challenges:

- **Reduced oxygen availability** (~50% at 5000m vs sea level)
- **Lower air pressure** and temperature
- **Increased UV radiation**
- **Limited resources**

Organisms adapt through multiple strategies:
- **Increased hematocrit** (more red blood cells)
- **Enhanced heart capacity** for improved circulation
- **Improved lung efficiency** for oxygen uptake
- **Metabolic adjustments** to conserve energy

## Step 1: Setup Base Simulation

First, let's create a control simulation at sea level:

```python
import requests
import json
import matplotlib.pyplot as plt
import pandas as pd

# IPE API base URL
BASE_URL = "http://localhost:8000"

def create_simulation(name, altitude, duration=200, population_size=1000):
    """Create a simulation with specific altitude."""
    
    # Calculate oxygen level based on altitude (simplified model)
    # At sea level: 1.0, at 5000m: ~0.5
    oxygen_level = max(0.3, 1.0 - (altitude / 10000))
    
    # Temperature decreases with altitude (~6.5°C per 1000m)
    temperature = 15 - (altitude / 1000) * 6.5
    
    response = requests.post(f"{BASE_URL}/api/v1/simulations", json={
        "name": name,
        "duration": duration,
        "population_size": population_size,
        "mutation_rate": 0.002,  # Slightly higher for faster adaptation
        "environment_params": {
            "altitude": altitude,
            "oxygen_level": oxygen_level,
            "temperature": temperature,
            "resource_availability": max(0.5, 1.0 - (altitude / 15000))
        }
    })
    
    response.raise_for_status()
    return response.json()

# Create sea level control
sea_level_sim = create_simulation("Sea Level Control", altitude=0)
print(f"Sea level simulation ID: {sea_level_sim['id']}")
```

## Step 2: Create Altitude Gradient

Now create simulations across an altitude gradient:

```python
# Define altitude gradient (meters)
altitudes = [0, 1000, 2000, 3000, 4000, 5000]
simulations = {}

for altitude in altitudes:
    sim_name = f"Altitude {altitude}m"
    sim = create_simulation(sim_name, altitude)
    simulations[altitude] = sim
    print(f"Created {sim_name}: {sim['id']}")

# Store simulation IDs for later use
sim_ids = {alt: sim['id'] for alt, sim in simulations.items()}
```

## Step 3: Monitor Simulation Progress

Set up monitoring for all simulations:

```python
import time
import asyncio
import websockets

async def monitor_simulation(sim_id, altitude):
    """Monitor a single simulation via WebSocket."""
    uri = f"ws://localhost:8000/ws/simulation/{sim_id}"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Monitoring {altitude}m altitude simulation...")
            
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data["type"] == "status":
                    progress = data["data"]["progress"] * 100
                    generation = data["data"]["generation"]
                    print(f"{altitude}m - Gen {generation}: {progress:.1f}% complete")
                    
                    # Stop monitoring when complete
                    if progress >= 100:
                        break
                        
    except websockets.exceptions.ConnectionClosed:
        print(f"Monitoring finished for {altitude}m")

# Monitor all simulations (run this in a separate terminal or async environment)
# async def monitor_all():
#     await asyncio.gather(*[
#         monitor_simulation(sim_id, altitude) 
#         for altitude, sim_id in sim_ids.items()
#     ])
# 
# asyncio.run(monitor_all())
```

For this tutorial, we'll simulate the monitoring by checking status periodically:

```python
def check_simulation_status(sim_id):
    """Check if simulation is complete."""
    response = requests.get(f"{BASE_URL}/api/v1/simulations/{sim_id}")
    return response.json()

# Wait for simulations to complete (simplified)
print("Waiting for simulations to complete...")
all_complete = False
while not all_complete:
    statuses = {}
    for altitude, sim_id in sim_ids.items():
        status = check_simulation_status(sim_id)
        statuses[altitude] = status['status']
    
    print("Current status:", statuses)
    
    # Check if all are done (in real system, would check actual status)
    all_complete = True  # For tutorial purposes, assume complete
    time.sleep(10)

print("All simulations complete!")
```

## Step 4: Analyze Results

Extract and analyze the final populations:

```python
def get_final_population(sim_id, min_generation=150):
    """Get final population from last generations."""
    response = requests.get(f"{BASE_URL}/api/v1/states/{sim_id}", params={
        "min_generation": min_generation,
        "limit": 1000
    })
    return response.json()

def analyze_adaptation(altitude, sim_id):
    """Analyze adaptations for a specific altitude."""
    
    # Get final population
    population = get_final_population(sim_id)
    
    if not population:
        print(f"No data available for {altitude}m")
        return None
    
    # Extract traits (assuming specific physiological traits)
    traits = []
    for individual in population:
        coords = individual['coordinates']
        # Assuming coordinates represent: [heart_mass, lung_capacity, hematocrit, muscle_efficiency]
        traits.append({
            'heart_mass': coords[0],
            'lung_capacity': coords[1], 
            'hematocrit': coords[2] if len(coords) > 2 else 0.4,
            'muscle_efficiency': coords[3] if len(coords) > 3 else 1.0,
            'fitness': individual['fitness'],
            'altitude': altitude
        })
    
    return pd.DataFrame(traits)

# Analyze all altitudes
results = {}
for altitude, sim_id in sim_ids.items():
    df = analyze_adaptation(altitude, sim_id)
    if df is not None:
        results[altitude] = df
        print(f"Altitude {altitude}m: {len(df)} individuals analyzed")
```

## Step 5: Visualize Adaptations

Create comprehensive visualizations:

```python
# Combine all results
combined_df = pd.concat(results.values(), ignore_index=True)

# Create publication-ready figures
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Hematocrit vs Altitude
axes[0,0].boxplot([results[alt]['hematocrit'] for alt in altitudes], 
                  labels=altitudes)
axes[0,0].set_xlabel('Altitude (m)')
axes[0,0].set_ylabel('Hematocrit')
axes[0,0].set_title('Hematocrit Adaptation to Altitude')

# 2. Heart Mass vs Altitude  
axes[0,1].boxplot([results[alt]['heart_mass'] for alt in altitudes],
                  labels=altitudes)
axes[0,1].set_xlabel('Altitude (m)')
axes[0,1].set_ylabel('Relative Heart Mass')
axes[0,1].set_title('Cardiac Adaptation to Altitude')

# 3. Fitness vs Altitude
mean_fitness = [results[alt]['fitness'].mean() for alt in altitudes]
axes[1,0].plot(altitudes, mean_fitness, 'bo-', linewidth=2, markersize=8)
axes[1,0].set_xlabel('Altitude (m)')
axes[1,0].set_ylabel('Mean Population Fitness')
axes[1,0].set_title('Fitness Across Altitude Gradient')

# 4. Multi-trait adaptation
for altitude in [0, 2000, 5000]:
    if altitude in results:
        df = results[altitude]
        axes[1,1].scatter(df['lung_capacity'], df['hematocrit'], 
                         alpha=0.6, label=f'{altitude}m')
axes[1,1].set_xlabel('Lung Capacity')
axes[1,1].set_ylabel('Hematocrit')
axes[1,1].set_title('Trait Correlations by Altitude')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('altitude_adaptation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Step 6: Statistical Analysis

Perform statistical tests on the adaptations:

```python
from scipy import stats
import numpy as np

def analyze_trait_trends(trait_name):
    """Analyze trends in a specific trait across altitude."""
    
    trait_data = []
    altitude_data = []
    
    for altitude in altitudes:
        if altitude in results:
            trait_values = results[altitude][trait_name]
            trait_data.extend(trait_values)
            altitude_data.extend([altitude] * len(trait_values))
    
    # Correlation with altitude
    correlation, p_value = stats.pearsonr(altitude_data, trait_data)
    
    # Linear regression
    slope, intercept, r_value, p_val_reg, std_err = stats.linregress(altitude_data, trait_data)
    
    print(f"\n{trait_name.title()} Analysis:")
    print(f"  Correlation with altitude: r = {correlation:.3f}, p = {p_value:.3e}")
    print(f"  Linear trend: slope = {slope:.2e}, R² = {r_value**2:.3f}")
    
    return {
        'trait': trait_name,
        'correlation': correlation,
        'p_value': p_value,
        'slope': slope,
        'r_squared': r_value**2
    }

# Analyze each trait
trait_analyses = []
for trait in ['hematocrit', 'heart_mass', 'lung_capacity', 'muscle_efficiency']:
    analysis = analyze_trait_trends(trait)
    trait_analyses.append(analysis)

# Summary table
analysis_df = pd.DataFrame(trait_analyses)
print("\nSummary of Trait Adaptations:")
print(analysis_df.round(4))
```

## Step 7: Generate Publication Figure

Create a publication-ready figure using the API:

```python
# Generate state space figures for key altitudes
key_altitudes = [0, 3000, 5000]

for altitude in key_altitudes:
    if altitude in sim_ids:
        response = requests.post(f"{BASE_URL}/api/v1/export/figure/{sim_ids[altitude]}", 
                               json={
                                   "simulation_id": sim_ids[altitude],
                                   "figure_type": "state_space",
                                   "format": "png",
                                   "width": 10,
                                   "height": 8,
                                   "dpi": 300
                               })
        
        with open(f"state_space_{altitude}m.png", "wb") as f:
            f.write(response.content)
        
        print(f"Saved state_space_{altitude}m.png")
```

## Step 8: Export Data

Export all data for further analysis:

```python
# Export simulation data for each altitude
for altitude, sim_id in sim_ids.items():
    
    # Create export job
    export_response = requests.post(f"{BASE_URL}/api/v1/export/simulation/{sim_id}",
                                   json={
                                       "simulation_id": sim_id,
                                       "format": "csv", 
                                       "data_types": ["simulation_data", "state_space", "statistics"]
                                   })
    
    export_job = export_response.json()
    export_id = export_job["export_id"]
    
    # Poll for completion (simplified)
    time.sleep(2)  # In real usage, poll until status is 'completed'
    
    # Download
    download_response = requests.get(f"{BASE_URL}/api/v1/export/download/{export_id}")
    
    with open(f"altitude_{altitude}m_data.csv", "wb") as f:
        f.write(download_response.content)
    
    print(f"Exported data for {altitude}m altitude")
```

## Results Interpretation

### Expected Patterns

1. **Hematocrit Increase**: Higher red blood cell count at altitude for improved oxygen transport
2. **Cardiac Enhancement**: Larger hearts for better circulation
3. **Lung Efficiency**: Improved oxygen extraction capacity  
4. **Metabolic Optimization**: Better energy utilization under hypoxic conditions

### Biological Significance

These adaptations mirror real-world patterns observed in:
- **High-altitude populations**: Tibetans, Andeans, Ethiopians
- **High-altitude animals**: Vicuñas, bar-headed geese, Himalayan mammals
- **Laboratory studies**: Hypoxia chamber experiments

## Extensions and Variations

### Variation 1: Rapid vs. Gradual Exposure

```python
# Simulate gradual altitude increase
def gradual_altitude_exposure(max_altitude, steps=5):
    """Simulate gradual adaptation to increasing altitude."""
    step_size = max_altitude // steps
    
    for step in range(steps):
        current_altitude = step_size * (step + 1)
        # Run simulation for shorter duration at each step
        # Transfer final population to next altitude level
```

### Variation 2: Population Migration

```python
# Simulate gene flow between altitude populations
def simulate_migration(source_sim_id, target_sim_id, migration_rate=0.1):
    """Simulate migration between populations."""
    # Get individuals from source population
    # Add them to target population
    # Useful for studying gene flow effects
```

### Variation 3: Plasticity vs. Evolution

```python
# Compare plastic responses vs. genetic adaptations
def plasticity_study(altitude):
    """Compare immediate plastic responses vs. long-term evolution."""
    
    # Short-term simulation (plastic response)
    plastic_sim = create_simulation(f"Plastic {altitude}m", altitude, duration=10)
    
    # Long-term simulation (evolutionary response)  
    evolved_sim = create_simulation(f"Evolved {altitude}m", altitude, duration=500)
    
    return plastic_sim, evolved_sim
```

## Conclusion

This tutorial demonstrated:

1. **Experimental Design**: Setting up altitude gradient studies
2. **Data Collection**: Monitoring and extracting simulation results
3. **Statistical Analysis**: Quantifying adaptation patterns
4. **Visualization**: Creating publication-ready figures
5. **Biological Interpretation**: Understanding adaptive responses

The IPE platform enables sophisticated evolutionary studies that complement field research and laboratory experiments, providing insights into the mechanisms and patterns of physiological adaptation.

## Next Steps

- **Advanced Analysis**: Explore [Phylogenetic Analysis Tutorial](phylogenetic-analysis.md)
- **Multi-stressor Studies**: Combine altitude with temperature variation  
- **Genomic Integration**: Link trait changes to gene expression patterns
- **Comparative Studies**: Compare with empirical data from literature