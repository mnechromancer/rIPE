# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**rIPE** (rudimentary Interactionist Phylogeny Engine) is a computational platform for simulating evolutionary processes in physiological state spaces with environmental interactions. The system enables prediction of evolutionary responses to environmental challenges like high-altitude adaptation, temperature stress, and rapid environmental change by modeling evolution as navigation through physiologically-explicit trait spaces.

## Common Commands

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests
pytest -m e2e              # End-to-end tests
pytest -m performance      # Performance/benchmark tests

# Run tests with coverage
pytest --cov=ipe --cov-report=html

# Run tests in parallel
pytest -n auto
```

### Development
```bash
# Run the core demo
python demo_core_001.py

# Run altitude adaptation simulation
python run_alpine_simulation.py

# Start API server (requires database running)
python -m ipe.api.server

# Start database services only
docker-compose up db redis -d

# Start all services
docker-compose up -d
```

### Code Quality
```bash
# Format code
black .

# Lint
flake8 ipe/ tests/
pylint ipe/ tests/

# Type checking
mypy ipe/

# Security scanning
bandit -r ipe/ -f json -o bandit-report.json
```

### Database
```bash
# Run migrations
alembic upgrade head

# Create new migration
alembic revision -m "description"

# Database setup
./scripts/db-setup.sh
```

### Docker
```bash
# Build containers
./scripts/docker-build.sh

# Deploy (production)
./scripts/deploy.sh
```

## Architecture

### High-Level Structure

rIPE follows a layered architecture with these core components:

```
┌─────────────────────────────────────────────────────┐
│              Web Interface (React/TypeScript)       │
├─────────────────────────────────────────────────────┤
│                   API Gateway (FastAPI)             │
├──────────────┬──────────────┬──────────────┬────────┤
│  Simulation  │   Analysis   │     Data     │   Lab  │
│   Engine     │   Service    │   Pipeline   │Integration│
├──────────────┴──────────────┴──────────────┴────────┤
│              Database (PostgreSQL + Redis)         │
└─────────────────────────────────────────────────────┘
```

### Core Module Organization

The main `ipe/` package is organized into:

- **`core/`** - Core computational algorithms
  - `physiology/` - Physiological state modeling (metabolism, cardiovascular, respiratory, thermoregulation, osmoregulation)
  - `plasticity/` - Phenotypic plasticity mechanisms (reaction norms, genetic assimilation, maladaptive responses)
  - `games/` - Game-theoretic models (hypoxia allocation, thermal trade-offs, resource games)
  - `state/` - State space operations and management
  - `thermodynamics/` - Thermodynamic constraints and energy balance

- **`simulation/`** - Simulation engine
  - Population dynamics and lifecycle
  - Selection mechanisms and fitness calculations
  - Mutation operators and genetic architecture
  - Contemporary/rapid evolution scenarios
  - Demographics and gene flow

- **`data/`** - Data management and integration
  - Data models and database schema
  - Import/export pipelines
  - Validation and quality control

- **`lab_integration/`** - Laboratory equipment and data integration
  - Respirometry data import (Sable Systems)
  - RNA-seq integration (DESeq2/EdgeR results)
  - Field data import (environmental, morphology, survival)
  - Molecular data (gene mapping, pathway analysis)

- **`api/`** - REST and WebSocket APIs
  - Experiment management endpoints
  - Simulation control
  - Real-time updates via WebSocket
  - Data export

- **`monitoring/`** - System monitoring and observability
  - Prometheus metrics
  - Health checks
  - Performance tracking

### Key Design Patterns

1. **Repository Pattern** - Data access abstraction for simulations, experiments, and results
2. **Strategy Pattern** - Pluggable selection algorithms and game-theoretic models
3. **Observer Pattern** - Real-time simulation updates via WebSocket
4. **Factory Pattern** - Creating different simulation types (altitude adaptation, thermal, etc.)

### Physiological State Management

The core abstraction is `PhysiologicalState`, which represents complete organism state including:
- Environmental conditions (pO2, temperature, altitude, salinity)
- Cardiovascular traits (heart mass, hematocrit, hemoglobin, cardiac output)
- Respiratory traits (lung volume, diffusion capacity, ventilation)
- Metabolic traits (BMR, VO2max, mitochondrial density)
- Thermoregulation (thermal conductance, critical temperatures)
- Osmoregulation (for aquatic organisms)

States are immutable dataclasses that support computation of derived metrics like aerobic scope and tissue-specific oxygen delivery.

### Game-Theoretic Framework

Evolutionary problems are modeled as games where organisms optimize trait allocations under constraints:
- **HypoxiaAllocationGame** - Optimal oxygen allocation to tissues under hypoxic stress
- **ThermogenesisTradeoffGame** - Balance heat production vs O2 consumption in cold
- **Resource allocation games** - Energy budgeting across competing demands

Each game implements `compute_payoff()` and `find_optimum()` methods to solve for evolutionarily stable strategies.

## Development Workflow

### Git Branching Strategy
```
main                    # Production-ready code
├── develop            # Integration branch
│   ├── feature/*     # New features
│   └── fix/*         # Bug fixes
├── release/*         # Release preparation
└── hotfix/*          # Critical production fixes
```

### Commit Message Format
```
<type>(<scope>): <subject>

Types: feat, fix, perf, docs, test, refactor
Scopes: simulation, viz, api, plasticity, games, etc.

Examples:
feat(simulation): add genetic assimilation algorithm
fix(viz): correct 3D rotation in state space
perf(calc): optimize fitness calculation with numba
test(plasticity): add maladaptive response tests
```

### Code Quality Standards

**Python:**
- Style: PEP 8 + Black formatter (line length 88)
- Type hints required for all public APIs
- Docstrings: NumPy style
- Test coverage target: >80%

**TypeScript/React:**
- Style: ESLint + Prettier
- Strict TypeScript mode enabled
- Functional components with hooks
- Props interfaces required

### Testing Strategy

Testing pyramid:
- **60% Unit tests** - Algorithm correctness, component behavior, utility functions
- **30% Integration tests** - Service communication, database operations, API workflows
- **10% E2E tests** - Full workflow validation, multi-user scenarios

**Scientific Validation**: Critical tests in `tests/validation/` ensure biological/physical accuracy:
- `known_adaptations.py` - Reproduces documented evolutionary outcomes
- `allometry.py` - Validates allometric scaling relationships
- Thermodynamic constraints and energy balance
- Parameter sensitivity analysis

### Performance Considerations

1. **Use NumPy vectorization** - Avoid Python loops for numerical operations
2. **Memory management** - Population data stored as numpy arrays; clear intermediate results
3. **Numba JIT** - Applied to hot paths in fitness calculations
4. **Multiprocessing** - For parallel population updates
5. **Database optimization** - Connection pooling, batch operations, indexed queries
6. **Caching** - Redis for expensive computation results

## Scientific Context

### Research Applications
- High-altitude adaptation in small mammals (primary focus: deer mice)
- Rapid evolution in contemporary time scales (10-100 generations)
- Phenotypic plasticity evolution and genetic assimilation
- Maladaptive plasticity detection and costs
- Freshwater invasion scenarios (e.g., alewife landlocking)

### Key Scientific Concepts
- **Reaction norms** - Genotype × Environment (G×E) interactions
- **Aerobic scope** - Difference between VO2max and BMR, critical performance metric
- **Genetic assimilation** - Evolution reducing plasticity when environment is stable
- **Maladaptive plasticity** - Plastic responses that reduce fitness
- **Nash equilibria** - Evolutionarily stable strategies in physiological trade-off games

## Lab Integration

### Data Import Workflows
- **Respirometry**: Sable Systems equipment data → VO2, VCO2, RER calculations
- **Transcriptomics**: DESeq2/EdgeR results → gene expression integration
- **Field data**: Environmental monitoring, morphometrics, mark-recapture
- **Molecular**: Gene ID conversion, KEGG/GO pathway enrichment

### Equipment APIs
- PySerial for respirometry equipment
- PyVISA for other lab instruments
- Apache Arrow for efficient columnar data transfer
- R integration via rpy2 for statistical exports

## Deployment

### Environments
- **Development**: Local or dev.ipe.velottalab.com
- **Staging**: staging.ipe.velottalab.com (automated on release/* branches)
- **Production**: ipe.velottalab.com (manual approval required)

### Infrastructure
- PostgreSQL 14+ with TimescaleDB extension for time-series physiology data
- Redis 6+ for caching and session management
- Celery + RabbitMQ for distributed simulation execution
- Docker + Docker Compose for containerization
- Prometheus + Grafana for monitoring

### Database Migrations
- Alembic manages schema changes
- Migration files in `migrations/versions/`
- Always test migrations on staging before production
- Prepare rollback plan for each migration

## Publication Alignment

Releases may be tagged with paper identifiers:
```
v2.3.1-pmde    # Peromyscus maniculatus deer mouse paper
v2.4.0-beta    # Beta release for testing
```

Feature freezes occur during paper submission periods. Scientific validation tests must pass before any publication-tagged release.
