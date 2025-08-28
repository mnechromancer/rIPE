# Frequently Asked Questions (FAQ)

## General Questions

### What is IPE?

The Interactionist Phylogeny Engine (IPE) is a scientific computing platform for simulating evolutionary processes in physiological state spaces. It models how organisms evolve in response to environmental challenges like altitude, temperature, and resource availability.

### Who is IPE designed for?

IPE is primarily designed for:
- **Evolutionary biologists** studying physiological adaptation
- **Comparative physiologists** exploring trait evolution
- **Ecology researchers** examining environmental responses
- **Graduate students** learning evolutionary dynamics
- **Lab groups** collaborating on adaptation studies

### How does IPE differ from other evolution simulators?

IPE is unique in its focus on:
- **Physiological state spaces** rather than abstract fitness landscapes
- **Environmental interactions** with realistic physical constraints
- **Lab integration** with experimental data workflows
- **Multi-organ systems** and their evolutionary trade-offs

## Installation & Setup

### What are the minimum system requirements?

**Minimum Requirements:**
- 8GB RAM, 4-core CPU
- 10GB free disk space
- Docker 20.10+ and Docker Compose 2.0+
- Modern web browser

**Recommended:**
- 16GB+ RAM, 8+ core CPU
- 50GB+ free disk space
- GPU support for large simulations
- High-speed internet for data downloads

### Why won't IPE start on my system?

Common issues:

1. **Docker not running**:
   ```bash
   # Check Docker status
   docker --version
   docker-compose --version
   
   # Start Docker (varies by OS)
   sudo systemctl start docker  # Linux
   # Or start Docker Desktop on Mac/Windows
   ```

2. **Port conflicts**:
   ```bash
   # Check if ports are in use
   lsof -i :8000  # API port
   lsof -i :3000  # Web interface port
   
   # Stop conflicting services or change ports in docker-compose.yml
   ```

3. **Insufficient permissions**:
   ```bash
   # Add user to docker group (Linux)
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

### Can I run IPE without Docker?

Yes, but it's more complex:

```bash
# Install dependencies
pip install -r requirements.txt
npm install  # In web/ directory

# Start database
sudo postgresql-service start
redis-server &

# Run migrations
alembic upgrade head

# Start services
uvicorn ipe.api.main:app --reload &
cd web && npm start &
```

Docker is strongly recommended for easier setup and deployment.

### How do I update IPE to the latest version?

```bash
# Pull latest code
git pull origin main

# Rebuild containers
docker-compose build

# Start updated services
docker-compose up -d

# Run any new migrations
docker-compose exec ipe-api alembic upgrade head
```

## Usage Questions

### How long do simulations take to run?

Simulation time depends on several factors:

| Parameters | Estimated Time |
|------------|----------------|
| 100 gen, 1K individuals | 2-5 minutes |
| 500 gen, 1K individuals | 10-30 minutes |
| 100 gen, 10K individuals | 20-60 minutes |
| 1000 gen, 10K individuals | 2-8 hours |

**Performance tips:**
- Start with smaller simulations (100 gen, 1K individuals)
- Enable GPU acceleration if available
- Use batch processing for multiple simulations
- Monitor resource usage with `docker stats`

### How do I know if my simulation is running correctly?

**Check simulation status:**
```bash
curl "http://localhost:8000/api/v1/simulations/YOUR_SIM_ID"
```

**Monitor via WebSocket:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/simulation/YOUR_SIM_ID');
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

**Look for these indicators:**
- Status changes from `created` → `running` → `completed`
- Generation counter increases steadily
- Population fitness shows variation
- No error messages in logs

### What do the physiological coordinates mean?

The state space coordinates represent physiological traits:

```python
# Example coordinate interpretation
coordinates = [2.1, -0.5, 1.3, 0.8]
# [0] = Heart mass (relative to body mass)
# [1] = Lung efficiency (relative to standard) 
# [2] = Hematocrit (blood oxygen capacity)
# [3] = Muscle fiber density (metabolic efficiency)
```

**Coordinate ranges:**
- Values are normalized (typically -3 to +3)
- Higher values generally indicate more of that trait
- Negative values are not "bad" - just relative differences
- Fitness depends on environment-trait interactions

### How do I interpret fitness values?

Fitness represents reproductive success probability:

- **1.0**: Perfect fitness, maximum reproduction
- **0.8**: Good fitness, above-average success  
- **0.5**: Average fitness, typical population member
- **0.2**: Low fitness, struggling in environment
- **0.0**: Lethal, cannot reproduce

**Fitness depends on:**
- How well traits match environmental demands
- Trade-offs between physiological systems
- Population-level competition effects
- Random environmental variation

### Can I modify simulation parameters during a run?

Currently, parameters cannot be changed once a simulation starts. However, you can:

1. **Pause and resume** (future feature)
2. **Create new simulations** with different parameters
3. **Use the final population** as starting point for new simulation
4. **Run parallel simulations** with parameter variations

### How do I export data in different formats?

**CSV Export:**
```bash
curl -X POST "http://localhost:8000/api/v1/export/simulation/SIM_ID" \
  -H "Content-Type: application/json" \
  -d '{"format": "csv", "data_types": ["simulation_data"]}'
```

**JSON Export:**
```bash
curl -X POST "http://localhost:8000/api/v1/export/simulation/SIM_ID" \
  -H "Content-Type: application/json" \
  -d '{"format": "json", "data_types": ["state_space", "statistics"]}'
```

**Publication Figures:**
```bash
curl -X POST "http://localhost:8000/api/v1/export/figure/SIM_ID" \
  -H "Content-Type: application/json" \
  -d '{"figure_type": "state_space", "format": "png", "dpi": 300}' \
  --output figure.png
```

## Scientific Questions

### Is IPE scientifically validated?

IPE incorporates established evolutionary and physiological principles:

**Evolutionary mechanisms:**
- Fisher's fundamental theorem
- Quantitative genetics models  
- Game theory equilibria
- Population genetics dynamics

**Physiological constraints:**
- Allometric scaling laws
- Thermodynamic principles
- Mass balance equations
- Oxygen transport physics

**Validation studies:**
- Comparison with published adaptation data
- Reproduction of known evolutionary outcomes
- Verification against laboratory experiments

See [Scientific Documentation](../science/validation.md) for detailed validation results.

### How realistic are the simulations?

IPE aims for biological realism within computational constraints:

**Realistic aspects:**
- Physiological trait interactions
- Environmental challenge responses
- Population-level evolutionary dynamics
- Multi-generational adaptation patterns

**Simplified aspects:**
- Genetic mechanisms (quantitative traits only)
- Environmental variation (constant within simulations)
- Species interactions (single species focus)
- Developmental processes (adult phenotypes only)

### Can IPE predict real evolutionary outcomes?

IPE is designed for **hypothesis generation** and **pattern exploration** rather than precise prediction:

**Appropriate uses:**
- Exploring "what if" scenarios
- Understanding adaptation mechanisms  
- Generating testable hypotheses
- Interpreting experimental results
- Planning field studies

**Limitations:**
- Cannot predict specific trait values
- Real environments are more complex
- Genetic architecture simplifications
- Historical contingency effects

### How do I validate my results?

**Internal validation:**
1. Run multiple replicate simulations
2. Check for consistent patterns across runs
3. Verify biological plausibility of results
4. Test parameter sensitivity

**External validation:**
1. Compare with published literature
2. Match patterns to empirical data
3. Design experiments to test predictions
4. Collaborate with field researchers

## Technical Questions

### How do I access the API programmatically?

**Python example:**
```python
import requests

# Create simulation
response = requests.post("http://localhost:8000/api/v1/simulations", 
                        json={"name": "Test", "duration": 100})
sim = response.json()

# Monitor progress
import websockets
import asyncio

async def monitor():
    uri = f"ws://localhost:8000/ws/simulation/{sim['id']}"
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            print(message)

# Get results
response = requests.get(f"http://localhost:8000/api/v1/states/{sim['id']}")
data = response.json()
```

**JavaScript example:**
```javascript
// Create simulation
const response = await fetch('http://localhost:8000/api/v1/simulations', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name: 'Test', duration: 100})
});
const sim = await response.json();

// Monitor via WebSocket
const ws = new WebSocket(`ws://localhost:8000/ws/simulation/${sim.id}`);
ws.onmessage = (event) => console.log(JSON.parse(event.data));
```

### Can I run IPE on a cluster or cloud?

Yes! IPE is designed for scalable deployment:

**Docker Swarm:**
```bash
docker swarm init
docker stack deploy -c docker-compose.yml ipe
```

**Kubernetes:**
```bash
kubectl apply -f kubernetes/
```

**Cloud platforms:**
- AWS ECS/EKS
- Google Cloud Run/GKE  
- Azure Container Instances/AKS
- DigitalOcean App Platform

### How do I backup my data?

**Database backup:**
```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U ipe ipe > backup.sql

# Restore
docker-compose exec -T postgres psql -U ipe ipe < backup.sql
```

**Export all simulations:**
```bash
# Use batch export API
curl -X POST "http://localhost:8000/api/v1/export/batch" \
  -H "Content-Type: application/json" \
  -d '{"simulation_ids": ["all"], "formats": ["json", "csv"]}'
```

**File system backup:**
```bash
# Backup entire data directory
docker-compose down
cp -r ./data ./data_backup_$(date +%Y%m%d)
docker-compose up -d
```

### What about security and authentication?

**Current security:**
- Local deployment by default
- API accessible only on localhost
- No authentication required for local use

**Production security (coming soon):**
- JWT authentication
- Role-based access control
- HTTPS/TLS encryption
- Rate limiting
- Input validation and sanitization

**Recommendations:**
- Use firewall rules for network access
- Run behind reverse proxy in production
- Enable authentication for shared deployments
- Regular security updates

## Troubleshooting

### My simulation appears stuck

**Check system resources:**
```bash
# Monitor resource usage
docker stats

# Check for out-of-memory issues
dmesg | grep -i "killed process"

# View simulation logs
docker-compose logs ipe-api
```

**Common causes:**
- Insufficient memory (increase in docker-compose.yml)
- Very large population sizes
- Infinite loops in simulation logic
- Database connection issues

### API returns 500 errors

**Check logs:**
```bash
docker-compose logs ipe-api
```

**Common fixes:**
```bash
# Restart services
docker-compose restart

# Check database connection
docker-compose exec postgres psql -U ipe -c "\l"

# Reset database if corrupted
docker-compose down -v
docker-compose up -d
docker-compose exec ipe-api alembic upgrade head
```

### WebSocket connections fail

**Verify WebSocket endpoint:**
```bash
# Test WebSocket connection
curl -H "Upgrade: websocket" \
     -H "Connection: Upgrade" \
     -H "Sec-WebSocket-Key: test" \
     -H "Sec-WebSocket-Version: 13" \
     http://localhost:8000/ws/simulation/test
```

**Common issues:**
- Firewall blocking WebSocket traffic
- Proxy server not configured for WebSocket upgrade
- Browser security policies
- Network connectivity problems

### How do I report bugs?

**Before reporting:**
1. Check this FAQ
2. Review GitHub issues
3. Try with minimal example
4. Collect error logs

**Bug report should include:**
- IPE version (`git rev-parse HEAD`)
- Operating system and version
- Docker/Docker Compose versions
- Complete error message
- Steps to reproduce
- Expected vs. actual behavior

**Submit to:**
- [GitHub Issues](https://github.com/mnechromancer/RIPE/issues)
- Include "Bug:" in title
- Use bug report template

## Getting Help

### Where can I find more documentation?

- **API Reference**: http://localhost:8000/docs
- **User Tutorials**: [tutorials/](tutorials/)
- **Scientific Background**: [../science/](../science/)
- **Developer Docs**: [../developer/](../developer/)

### How do I get community support?

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Most common questions answered here
- **Email Support**: support@ipe.velottalab.com

### Can I contribute to IPE development?

Yes! Contributions are welcome:

**Types of contributions:**
- Bug reports and fixes
- Feature requests and implementations
- Documentation improvements
- Scientific validation studies
- Tutorial and example development

**Getting started:**
1. Read [Contributing Guide](../developer/contributing.md)
2. Check open issues on GitHub
3. Join developer discussions
4. Start with small improvements

**Development setup:**
```bash
git clone https://github.com/mnechromancer/RIPE.git
cd RIPE
git checkout -b feature/my-contribution
# Make changes
git commit -m "Description of changes"
git push origin feature/my-contribution
# Create pull request
```