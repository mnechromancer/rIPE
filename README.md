# RIPE - Interactionist Phylogeny Engine

[![License: Public Domain](https://img.shields.io/badge/License-Public%20Domain-blue.svg)](https://unlicense.org/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

> A computational platform for simulating evolutionary processes in physiological state spaces with environmental interactions.

**RIPE** (Repository for **I**nteractionist **P**hylogeny **E**ngine) enables researchers to predict evolutionary responses to environmental challenges like high-altitude adaptation, temperature stress, and rapid environmental change by modeling evolution as navigation through physiologically-explicit trait spaces.

## Table of Contents

- [🔬 What is IPE?](#-what-is-ipe)
- [🚀 Quick Start](#-quick-start)
- [📖 Documentation](#-documentation)
- [🔬 Key Features](#-key-features)
- [💻 Technology Stack](#-technology-stack)
- [🏗️ Architecture](#️-architecture)
- [📊 Example Use Cases](#-example-use-cases)
- [🤝 Contributing](#-contributing)
- [📧 Support](#-support)

## 🔬 What is IPE?

The Interactionist Phylogeny Engine treats evolution as a strategic game played in multi-dimensional physiological state space. Unlike traditional phylogenetic reconstruction, IPE **predicts** evolutionary trajectories by computing equilibria between organisms and their environments.

**Key capabilities:**
- **Physiological Evolution**: Model heart mass, hematocrit, lung capacity, metabolic rate, and other traits
- **Environmental Interactions**: Simulate responses to altitude, temperature, oxygen availability, and salinity
- **Plasticity Evolution**: Track adaptive and maladaptive phenotypic plasticity
- **Game-Theoretic Modeling**: Strategic interactions between organisms and environments
- **High-Altitude Specialization**: Purpose-built for altitude adaptation research

## 🚀 Quick Start

### Prerequisites
- **Docker** and **Docker Compose** (recommended)
- **Python 3.9+** (for development)
- **8GB+ RAM** (16GB+ recommended for large simulations)

### Installation

#### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/mnechromancer/RIPE.git
   cd RIPE
   ```

2. **Start all services**
   ```bash
   docker-compose up -d
   ```

3. **Verify installation**
   - API: http://localhost:8000/docs
   - Web Interface: http://localhost:3000 (if enabled)
   - Health Check: http://localhost:8000/health

#### Option 2: Local Development Setup

1. **Clone and install dependencies**
   ```bash
   git clone https://github.com/mnechromancer/RIPE.git
   cd RIPE
   pip install -r requirements.txt
   ```

2. **Test the installation**
   ```bash
   python demo_core_001.py
   ```

3. **Start development services** (requires Docker for database)
   ```bash
   docker-compose up db redis -d  # Start database and cache
   python -m ipe.api.server      # Start API server
   ```

### Your First Simulation

#### Using Python API
```python
from ipe.simulation import AltitudeAdaptationSimulation

# Create a high-altitude adaptation simulation
sim = AltitudeAdaptationSimulation(
    population_size=1000,
    generations=500,
    altitude_range=(0, 4000),  # Sea level to 4000m
    selection_strength=0.1
)

# Run simulation
results = sim.run()

# Analyze results
print(f"Final mean heart mass: {results.final_heart_mass:.2f}g")
print(f"Final mean hematocrit: {results.final_hematocrit:.1f}%")
```

#### Using the Web Interface
1. Navigate to http://localhost:3000
2. Click "New Simulation" 
3. Select "Altitude Adaptation" template
4. Configure parameters and click "Start"
5. Monitor progress in real-time

#### Using the REST API
```bash
# Create simulation
curl -X POST "http://localhost:8000/api/v1/simulations" \
     -H "Content-Type: application/json" \
     -d '{
       "type": "altitude_adaptation",
       "population_size": 1000,
       "generations": 500,
       "parameters": {"altitude_range": [0, 4000]}
     }'
```

For more examples, see the [Complete Getting Started Guide](docs/user/getting-started.md).

## 📖 Documentation

### For Users
- **[Getting Started Guide](docs/user/getting-started.md)** - Complete installation and setup
- **[Tutorials](docs/user/tutorials/)** - Step-by-step examples for common use cases
  - [Altitude Adaptation Simulation](docs/user/tutorials/altitude-adaptation.md)
- **[FAQ](docs/user/faq.md)** - Common questions and troubleshooting

### For Developers  
- **[API Reference](docs/api/README.md)** - REST API documentation
- **[Architecture Guide](docs/developer/architecture.md)** - System design and components
- **[Development Lifecycle](docs/developer/development-lifecycle.md)** - Development process and workflows

### For Scientists
- **[Scientific Algorithms](docs/science/algorithms.md)** - Mathematical foundations
- **[Validation Methods](docs/science/validation.md)** - How IPE ensures accuracy
- **[Publications Guide](docs/science/publications.md)** - How to cite IPE in your research

### Operations
- **[Deployment Guide](docs/operations/deployment.md)** - Production deployment
- **[Monitoring Setup](docs/operations/monitoring.md)** - System monitoring and alerts

## 🔬 Key Features

### Physiological State Modeling
- **Multi-organ systems**: Heart, lungs, blood, muscle, kidneys
- **Environmental gradients**: Altitude, temperature, oxygen, salinity
- **Performance metrics**: Aerobic scope, thermal tolerance, locomotor capacity

### Evolutionary Mechanisms
- **Natural selection** with physiological constraints
- **Phenotypic plasticity** (adaptive and maladaptive responses)
- **Genetic drift** and population effects
- **Migration** between environments
- **Rapid evolution** scenarios

### Research Applications
- **High-altitude adaptation** in small mammals
- **Climate change responses** across species
- **Physiological trade-offs** and constraints
- **Reaction norm evolution** and plasticity costs
- **Contemporary evolution** in changing environments

## 💻 Technology Stack

- **Backend**: Python 3.9+, FastAPI, NumPy, SciPy
- **Database**: PostgreSQL with TimescaleDB extension
- **Cache**: Redis
- **Frontend**: React, TypeScript, Three.js (for 3D visualizations)
- **Deployment**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Web Interface (React/TypeScript)           │
├─────────────────────────────────────────────────────────┤
│                   API Gateway (FastAPI)                 │
├──────────────┬──────────────┬──────────────┬──────────┤
│  Simulation  │   Analysis   │     Data     │   Lab    │
│   Engine     │   Service    │   Pipeline   │Integration│
├──────────────┴──────────────┴──────────────┴──────────┤
│              Database (PostgreSQL + Redis)             │
└─────────────────────────────────────────────────────────┘
```

## 📊 Example Use Cases

1. **Predict cardiac adaptation** to high altitude in deer mice
2. **Model thermoregulatory evolution** across temperature gradients  
3. **Simulate rapid adaptation** to environmental change
4. **Explore plasticity evolution** and its costs
5. **Generate testable predictions** for field/lab experiments

## 🤝 Contributing

We welcome contributions from the scientific computing and evolutionary biology communities!

- **Report Issues**: [GitHub Issues](https://github.com/mnechromancer/RIPE/issues)
- **Request Features**: Use issue templates for new functionality
- **Submit Code**: Follow our [development guidelines](docs/developer/development-lifecycle.md)
- **Scientific Validation**: Help validate algorithms against empirical data

## 📜 License

This project is released into the public domain under [The Unlicense](LICENSE). You are free to use, modify, and distribute this software without restriction.

## 🙏 Acknowledgments

Developed for the **Velotta Lab** and the broader evolutionary physiology research community. Special thanks to contributors, collaborators, and the open-source scientific computing ecosystem.

## 📧 Support

- **Documentation**: [Complete User Guide](docs/user/getting-started.md)
- **API Docs**: [Interactive API Documentation](docs/api/README.md) (also at `/docs` when running)
- **Issues**: [GitHub Issues](https://github.com/mnechromancer/RIPE/issues)
- **Scientific Questions**: See our [FAQ](docs/user/faq.md) and [Publications Guide](docs/science/publications.md)

---

*IPE is designed for researchers studying evolutionary responses to environmental challenges, with special focus on high-altitude adaptation, phenotypic plasticity, and rapid contemporary evolution.*