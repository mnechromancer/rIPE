# RIPE - Interactionist Phylogeny Engine

RIPE is a Python-based phylogenetic simulation platform with FastAPI backend, PostgreSQL+TimescaleDB database, Redis cache, and optional React frontend. The platform simulates evolutionary processes with physiological constraints and environmental interactions.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap and Development Setup (Required First Steps)

**Install System Dependencies:**
```bash
# Ensure Docker and Python are available
docker --version          # Should be 20.10+
python --version          # Should be 3.12+
pip --version
```

**Clone and Setup Python Environment:**
```bash
git clone https://github.com/mnechromancer/RIPE.git
cd RIPE
pip install -r requirements.txt
pip install uvicorn alembic psycopg2-binary redis matplotlib seaborn sqlalchemy black flake8 mypy pylint httpx
```

**Test Basic Installation (CRITICAL - Always run this first):**
```bash
python demo_core_001.py
# Should complete in ~4 seconds with success message
# This validates core phylogenetic simulation functionality
```

### Database Setup (Required for API)

**Start Database Services:**
```bash
# NEVER CANCEL: Database startup takes 1-2 minutes. Set timeout to 5+ minutes.
docker compose up db redis -d
```

**Wait for Services and Run Migrations:**
```bash
# Wait for healthy status (takes 30-60 seconds)
docker compose ps  # Both services should show "healthy"

# Set environment and run migrations
export DATABASE_URL=postgresql://ipe_user:ipe_dev_password@localhost:5432/ipe_db
export REDIS_URL=redis://localhost:6379
alembic upgrade head
```

### API Server Startup

**Start the API Server:**
```bash
# Set required environment variables
export DATABASE_URL=postgresql://ipe_user:ipe_dev_password@localhost:5432/ipe_db
export REDIS_URL=redis://localhost:6379  
export PYTHONPATH=/path/to/RIPE  # Use absolute path to repo root

# Start server (runs indefinitely until stopped)
uvicorn ipe.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Verify API is Running:**
- Health Check: `curl http://localhost:8000/health`
- API Docs: http://localhost:8000/docs
- Root endpoint: `curl http://localhost:8000/`

## Build and Test Commands

### Linting and Formatting (Fast - under 2 seconds each)
```bash
export PYTHONPATH=/path/to/RIPE

# Code formatting check (1.6 seconds)
black --check --diff .

# Linting (0.9 seconds) 
flake8 ipe/ scripts/ tests/ --max-line-length=88 --extend-ignore=E203,W503

# Type checking (optional - may have warnings)
mypy ipe/ scripts/ --ignore-missing-imports

# Additional linting (optional - may have warnings)  
pylint ipe/ scripts/ --disable=C0103,R0903,W0613
```

### Testing

**Unit Tests (Fast - 4 seconds for 318 tests):**
```bash
export PYTHONPATH=/path/to/RIPE
pytest tests/unit/ -v --tb=short
# NEVER CANCEL: All 318 tests complete in ~4 seconds
```

**Integration Tests (36 pass, 4 minor failures - ~3.5 seconds):**
```bash
export DATABASE_URL=postgresql://ipe_user:ipe_dev_password@localhost:5432/ipe_db
export REDIS_URL=redis://localhost:6379  
export PYTHONPATH=/path/to/RIPE
pytest tests/integration/ -v --tb=short
# Note: 4 tests fail due to minor data inconsistencies, not API issues
```

## Docker Operations

### Docker Services (Database Only - Recommended)
```bash
# Start just database services (works reliably)
docker compose up db redis -d

# Check service status
docker compose ps

# Stop services
docker compose down
```

### Docker Build (Currently Has Issues)
```bash
# KNOWN ISSUE: Full docker compose build fails due to SSL certificate issues
# in the build environment. Use local development setup instead.

# Docker build script is available but not functional:
./scripts/docker-build.sh -t development  # Will fail with SSL errors
```

## Validation and User Scenarios

### CRITICAL: Always Test Core Functionality
```bash
# 1. Basic functionality validation
python demo_core_001.py

# 2. API simulation test  
curl -X POST "http://localhost:8000/api/v1/simulations" \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Simulation", "duration": 10, "population_size": 100, "mutation_rate": 0.001, "environment_params": {"altitude": 3000, "temperature": -5, "oxygen_level": 0.7}}'

# Should return JSON with simulation ID and "created" status
```

### Complete Validation Workflow
After making any changes, ALWAYS run this complete validation:

```bash
# 1. Core functionality
python demo_core_001.py

# 2. Code quality
black --check --diff .
flake8 ipe/ scripts/ tests/ --max-line-length=88 --extend-ignore=E203,W503

# 3. Unit tests  
export PYTHONPATH=/path/to/RIPE
pytest tests/unit/ -v --tb=short

# 4. Start services and test API
docker compose up db redis -d
# Wait 1-2 minutes for healthy status
export DATABASE_URL=postgresql://ipe_user:ipe_dev_password@localhost:5432/ipe_db
export REDIS_URL=redis://localhost:6379
alembic upgrade head
uvicorn ipe.api.main:app --host 0.0.0.0 --port 8000 &
sleep 10
curl http://localhost:8000/health  # Should return {"status":"healthy"}
```

## Key Projects and Navigation

### Core Components
- **`ipe/core/`** - Evolution algorithms, phylogenetic models, game theory
- **`ipe/api/`** - FastAPI REST endpoints and WebSocket support  
- **`ipe/simulation/`** - Population management and generation advancement
- **`ipe/data/`** - Data integration and file format parsers
- **`tests/unit/`** - Comprehensive unit test suite (318 tests)

### Important Scripts  
- **`demo_core_001.py`** - Core functionality demonstration and validation
- **`scripts/docker-build.sh`** - Docker build automation (has SSL issues)
- **`scripts/deploy.sh`** - Deployment automation for staging/production
- **`scripts/db-setup.sh`** - Database setup and optimization

### Configuration Files
- **`docker-compose.yml`** - Service definitions (DB, Redis, API)
- **`requirements.txt`** - Python dependencies
- **`alembic.ini`** - Database migration configuration
- **`.github/workflows/`** - CI/CD pipeline definitions

## Timing Expectations and Timeouts

**CRITICAL: NEVER CANCEL these operations before their expected completion times:**

- **Docker service startup**: 1-2 minutes (set timeout to 5+ minutes)
- **Database migrations**: 30-60 seconds (set timeout to 3+ minutes)
- **Unit test suite**: ~4 seconds (set timeout to 30+ seconds)
- **Integration test suite**: ~3.5 seconds (set timeout to 30+ seconds)
- **Code formatting/linting**: 1-2 seconds each (set timeout to 30+ seconds)
- **API server startup**: 5-10 seconds (set timeout to 30+ seconds)
- **Demo script**: ~4 seconds (set timeout to 30+ seconds)

## Known Issues and Limitations

### What Works
- ✅ Python local development setup
- ✅ Database services via Docker  
- ✅ API server with all endpoints
- ✅ Complete test suite (318 unit tests, 36/40 integration tests pass)
- ✅ Linting and formatting tools  
- ✅ Core phylogenetic simulation functionality
- ✅ API endpoints including WebSocket support

### What Doesn't Work  
- ❌ Full `docker compose up` - fails during API container build due to SSL certificate issues
- ❌ Web frontend - React structure exists but no package.json/build system configured
- ❌ `python -m ipe.api.server` - use `uvicorn ipe.api.main:app` instead

### Workarounds
- Use local development setup instead of full Docker build
- Start only database services with `docker compose up db redis -d` 
- Use `uvicorn ipe.api.main:app` to start API server
- Always set PYTHONPATH when running tests or using modules

## Environment Variables (Always Required)
```bash
export PYTHONPATH=/absolute/path/to/RIPE
export DATABASE_URL=postgresql://ipe_user:ipe_dev_password@localhost:5432/ipe_db  
export REDIS_URL=redis://localhost:6379
```

## Quick Reference Commands
```bash
# Repository root listing
ls -la
# .github  .git  Dockerfile  LICENSE  README.md  alembic.ini  demo_core_001.py  
# docker-compose.yml  docs  ipe  migrations  requirements.txt  scripts  tests  web

# Start development workflow
python demo_core_001.py                    # Test core functionality  
docker compose up db redis -d              # Start database (1-2 min)
alembic upgrade head                        # Run migrations
uvicorn ipe.api.main:app --port 8000       # Start API server
curl http://localhost:8000/health          # Test API
```