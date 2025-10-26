# rIPE Project Management Guide

This document provides guidance for managing development tasks, tracking progress, and coordinating work on the rIPE platform.

## Project Philosophy

**rIPE is rudimental** - We focus on batch-processed simulations with post-hoc analysis rather than real-time visualization. The platform prioritizes:

1. **Accuracy over speed** - Phylogenetic simulations should be scientifically rigorous
2. **Visualization over live monitoring** - Users analyze completed simulations, not watch them run
3. **Reproducibility** - All simulations should be deterministic and repeatable
4. **Research workflow** - Tools for hypothesis testing and comparative analysis

## Current State Assessment

### Backend (Python/FastAPI) - ✅ Mostly Complete
- Core simulation engine: **COMPLETE**
- Phylogenetic algorithms: **COMPLETE**
- Game theory models: **COMPLETE**
- Physiology/thermodynamics: **COMPLETE**
- REST API endpoints: **COMPLETE**
- Database schema: **COMPLETE**
- Test coverage: **EXCELLENT** (318 unit tests, 36/40 integration tests pass)

### Frontend (React/TypeScript) - 🚧 Needs Major Development
- Project structure: **COMPLETE** (Vite + React + TypeScript configured)
- UI framework: **COMPLETE** (Tailwind CSS, Headless UI)
- Visualization components: **NOT IMPLEMENTED**
- API integration: **PARTIAL** (axios configured, but components need implementation)
- Data fetching: **NOT IMPLEMENTED**
- State management: **NOT IMPLEMENTED**

### Infrastructure - ⚠️ Partial
- Docker services (DB/Redis): **WORKING**
- Full Docker build: **BROKEN** (SSL certificate issues)
- CI/CD: **EXISTS** (GitHub Actions workflows defined)
- Monitoring: **CONFIGURED** (Prometheus/Alertmanager config exists)

## Development Priorities

### Phase 1: Core Visualization (High Priority)
Enable users to visualize completed simulations.

**Tasks:**
1. **Simulation Results Viewer**
   - Fetch simulation data from `/api/v1/simulations/{id}/states`
   - Display summary statistics (generations, fitness evolution, population size)
   - Show basic metadata (start time, duration, environment params)

2. **2D Fitness Charts** 
   - Use Recharts to plot fitness over generations
   - Show min/mean/max fitness bands
   - Enable comparison of multiple simulations side-by-side

3. **Population Statistics Dashboard**
   - Display trait distributions
   - Show genetic variance over time
   - Visualize selection pressures

**Estimated Effort:** 2-3 days  
**Dependencies:** API server must be running (already works)

### Phase 2: State Space Visualization (Medium Priority)
Implement 3D phylogenetic state space exploration.

**Tasks:**
1. **Three.js Integration**
   - Install `@react-three/fiber` and `@react-three/drei`
   - Create basic 3D scene component
   - Implement camera controls (orbit, zoom, pan)

2. **State Space Renderer**
   - Fetch state coordinates from API
   - Render points in 3D space
   - Color code by fitness or generation
   - Add trajectory lines between ancestor-descendant pairs

3. **Interactive Features**
   - Click on state to see details
   - Filter by generation range
   - Toggle different trait dimensions

**Estimated Effort:** 3-4 days  
**Dependencies:** Phase 1 complete, Three.js experience needed

### Phase 3: Comparative Analysis Tools (Medium Priority)
Enable hypothesis testing through simulation comparison.

**Tasks:**
1. **Simulation Queue Manager**
   - UI to create multiple simulations with different parameters
   - Batch submission
   - Status tracking (queued, running, complete)

2. **Comparison Dashboard**
   - Side-by-side simulation results
   - Statistical comparison (t-tests, ANOVA)
   - Differential fitness plots
   - Export comparison reports

3. **Hypothesis Testing Workflow**
   - Template system for common experimental designs
   - Control vs. experimental condition setup
   - Expected vs. actual outcome tracking

**Estimated Effort:** 4-5 days  
**Dependencies:** Phase 1 complete

### Phase 4: Advanced Features (Low Priority)
Features that enhance research capabilities but aren't essential.

**Tasks:**
1. **Genetic Assimilation Tracker**
   - Integrate with `CanalizationEngine`
   - Visualize plastic-to-genetic transition
   - Predict assimilation trajectories

2. **Data Export and Import**
   - Export simulation results to CSV/JSON
   - Import experimental data for comparison
   - Field data integration (respirometry, molecular)

3. **Publication-Ready Figures**
   - High-resolution plot export
   - Customizable styling for papers
   - LaTeX label support

**Estimated Effort:** 5-7 days  
**Dependencies:** Phases 1-3 complete

## Task Tracking System

### Issue Labels
Use GitHub Issues with these labels:

- `backend` - Python/FastAPI work
- `frontend` - React/TypeScript work
- `visualization` - Plotting and 3D rendering
- `api` - REST endpoint development
- `testing` - Test coverage and validation
- `bug` - Something is broken
- `enhancement` - New feature request
- `documentation` - Docs and guides
- `good-first-issue` - Easy tasks for new contributors

### Branch Naming Convention
```
feature/<short-description>    # New features
bugfix/<issue-number>          # Bug fixes
refactor/<component-name>      # Code refactoring
docs/<topic>                   # Documentation updates
```

### Commit Message Format
```
<type>(<scope>): <short summary>

<detailed description>

Fixes #<issue-number>
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `style`, `chore`

**Examples:**
```
feat(frontend): add simulation results viewer component

- Fetch data from /api/v1/simulations/{id}/states
- Display generation count and fitness statistics
- Add loading and error states

Fixes #42
```

```
fix(api): correct fitness calculation in selection.py

The fitness function was not properly normalizing by population size.
This led to inflated fitness values in large populations.

Fixes #58
```

## Development Workflow

### Starting a New Feature
1. **Check existing documentation**
   - Read `.github/copilot-instructions.md` for technical setup
   - Review `docs/developer/architecture.md` for design patterns
   - Check API docs at `http://localhost:8000/docs`

2. **Create a branch**
   ```bash
   git checkout -b feature/simulation-viewer
   ```

3. **Set up environment**
   ```bash
   # Backend
   export PYTHONPATH=/Users/jamison.ducey/rIPE
   export DATABASE_URL=postgresql://ipe_user:ipe_dev_password@localhost:5432/ipe_db
   export REDIS_URL=redis://localhost:6379
   
   # Start services
   docker compose up db redis -d
   alembic upgrade head
   uvicorn ipe.api.main:app --port 8000 --reload &
   
   # Frontend
   cd web/
   npm install
   npm run dev
   ```

4. **Implement feature**
   - Write code
   - Add tests
   - Update documentation

5. **Validate changes**
   ```bash
   # Backend validation
   python demo_core_001.py
   pytest tests/unit/ -v
   black --check .
   flake8 ipe/ scripts/ tests/ --max-line-length=88 --extend-ignore=E203,W503
   
   # Frontend validation
   cd web/
   npm run lint
   npx tsc --noEmit
   ```

6. **Create pull request**
   - Write clear description
   - Link related issues
   - Request review

### Code Review Guidelines

**For Reviewers:**
- Check test coverage
- Verify documentation updates
- Test functionality locally
- Look for potential edge cases
- Ensure code follows existing patterns

**For Authors:**
- Respond to feedback promptly
- Don't take criticism personally
- Explain design decisions when questioned
- Update PR description as code evolves

## Common Development Scenarios

### Adding a New API Endpoint

1. **Define the endpoint in appropriate route file**
   ```python
   # ipe/api/routes/simulations.py
   @router.get("/simulations/{simulation_id}/analysis")
   async def get_simulation_analysis(simulation_id: int):
       # Implementation
   ```

2. **Add database query if needed**
   ```python
   # ipe/data/models.py or create new query function
   ```

3. **Write tests**
   ```python
   # tests/integration/test_api.py
   def test_get_simulation_analysis():
       # Test implementation
   ```

4. **Update API documentation**
   - Endpoint docstring with OpenAPI annotations
   - Example in `docs/api/examples.md`

### Adding a Frontend Component

1. **Create component file**
   ```typescript
   // web/src/components/SimulationViewer.tsx
   import React, { useState, useEffect } from 'react';
   import axios from 'axios';
   
   export const SimulationViewer: React.FC<Props> = ({ simulationId }) => {
     // Implementation
   };
   ```

2. **Add TypeScript types**
   ```typescript
   // web/src/types/simulation.ts
   export interface SimulationData {
     id: number;
     name: string;
     status: 'created' | 'running' | 'completed';
     // ...
   }
   ```

3. **Connect to API**
   ```typescript
   const fetchSimulation = async (id: number) => {
     const response = await axios.get(`http://localhost:8000/api/v1/simulations/${id}`);
     return response.data;
   };
   ```

4. **Add to routing/page**
   ```typescript
   // web/src/App.tsx or appropriate route file
   ```

### Debugging Common Issues

**Problem: Frontend can't reach API**
```bash
# Check API is running
curl http://localhost:8000/health

# Check CORS settings in ipe/api/main.py
# Should allow http://localhost:5173 (Vite default)
```

**Problem: Database connection failed**
```bash
# Check services are running
docker compose ps

# Restart if needed
docker compose down
docker compose up db redis -d

# Wait for healthy status (1-2 minutes)
docker compose ps  # Should show "healthy"
```

**Problem: Tests failing after changes**
```bash
# Run specific test to see detailed error
pytest tests/unit/test_specific.py::test_function_name -v -s

# Check if database schema changed - may need migration
alembic revision --autogenerate -m "description"
alembic upgrade head
```

**Problem: Import errors in Python**
```bash
# Always set PYTHONPATH
export PYTHONPATH=/Users/jamison.ducey/rIPE

# Verify it's set
echo $PYTHONPATH
```

**Problem: Frontend build errors**
```bash
cd web/
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## Performance Considerations

### Simulation Timing
- **Small simulation (100 pop, 50 gen):** ~2-5 seconds
- **Medium simulation (500 pop, 200 gen):** ~30-60 seconds
- **Large simulation (2000 pop, 1000 gen):** ~5-10 minutes

**DO NOT implement "real-time" monitoring** - simulations should complete fully, then be visualized. Use job queues (Redis) for long-running simulations.

### API Response Times
- **GET /health:** < 50ms
- **GET /simulations:** < 200ms (paginated)
- **GET /simulations/{id}/states:** < 500ms for small datasets, up to 5s for large
- **POST /simulations:** < 100ms (just creates record, doesn't run sim)

**Optimization strategies:**
- Paginate large result sets
- Cache frequently accessed simulations (Redis)
- Use database indexes on commonly queried fields
- Batch-fetch related data to avoid N+1 queries

### Frontend Performance
- **Render < 1000 data points** directly in Recharts
- **For > 1000 points:** Downsample or use virtualization
- **3D scenes:** Limit to < 10,000 rendered points
- **Use React.memo** for expensive components
- **Lazy load** Three.js components (code splitting)

## Documentation Standards

### Code Comments
```python
# Good: Explain WHY, not WHAT
def calculate_fitness(phenotype: np.ndarray, environment: Environment) -> float:
    """Calculate fitness given phenotype and environment.
    
    Uses thermodynamic constraints from Denny (2017) to penalize
    physiologically impossible states. This prevents unrealistic
    adaptations like negative heart mass.
    
    Args:
        phenotype: Array of trait values [heart_mass, lung_capacity, ...]
        environment: Environmental parameters (altitude, temp, O2)
        
    Returns:
        Fitness value between 0 and 1
    """
    # Physiological impossibility check prevents negative organ masses
    if not is_physiologically_possible(phenotype):
        return 0.0
    # ... rest of implementation
```

### API Documentation
Every endpoint needs:
- Summary (one line)
- Description (paragraph explaining purpose)
- Parameters (with types and constraints)
- Response schema
- Example request/response
- Potential error codes

### README Updates
When adding major features, update:
- Main `README.md` - High-level overview
- `docs/user/getting-started.md` - User-facing changes
- `docs/developer/architecture.md` - Technical details

## Release Process

### Version Numbering
Follow semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking API changes
- **MINOR:** New features, backward compatible
- **PATCH:** Bug fixes only

### Pre-Release Checklist
- [ ] All tests pass (`pytest tests/`)
- [ ] Linting passes (`black --check .` and `flake8`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in appropriate files
- [ ] Database migrations tested
- [ ] Frontend builds without errors (`npm run build`)

### Deployment
Refer to `docs/operations/deployment.md` for detailed deployment procedures.

## Getting Help

### Internal Resources
1. `.github/copilot-instructions.md` - Technical setup and troubleshooting
2. `docs/developer/architecture.md` - System design and patterns
3. `docs/api/README.md` - API documentation
4. `docs/science/algorithms.md` - Scientific background

### External Resources
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **React Docs:** https://react.dev/
- **Three.js Examples:** https://threejs.org/examples/
- **TimescaleDB Docs:** https://docs.timescale.com/

### Questions and Discussion
- Create a GitHub Discussion for questions
- Use GitHub Issues for bugs and feature requests
- Keep discussions focused and searchable

## Contribution Guidelines

### Before Starting Work
1. Check if issue already exists
2. Discuss approach in issue comments
3. Wait for approval on large features
4. Keep PRs focused and reasonably sized

### Code Standards
- **Python:** Follow PEP 8, use Black formatter
- **TypeScript:** Follow ESLint rules, use Prettier
- **Line length:** 88 characters (Python), 100 (TypeScript)
- **Tests:** Required for all new features
- **Documentation:** Required for all public APIs

### Pull Request Process
1. Create PR from feature branch to `main`
2. Fill out PR template completely
3. Ensure CI passes
4. Request review from maintainers
5. Address review feedback
6. Squash commits if requested
7. Maintainer will merge when approved

## Project Roadmap

### Q4 2024 (Current)
- ✅ Core simulation engine
- ✅ API implementation
- 🚧 Basic frontend visualizations
- 🚧 Documentation completion

### Q1 2025
- 🎯 3D state space visualization
- 🎯 Comparative analysis tools
- 🎯 First stable release (v1.0.0)

### Q2 2025
- 🎯 Advanced prediction features
- 🎯 Lab data integration
- 🎯 Publication-ready export

### Future Considerations
- Multi-objective optimization
- GPU acceleration for large simulations
- Cloud deployment options
- Collaborative features (shared simulations)

## Notes

- Remember: **rudimental** means we don't need real-time updates
- Focus on **post-simulation analysis** not live monitoring
- Prioritize **scientific accuracy** over flashy features
- Keep **reproducibility** as a core principle
