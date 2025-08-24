# Interactionist Phylogeny Engine - Development Document
**Version 1.1 - Velotta Lab Edition**

## Technology Stack

### Core Backend
- **Language**: Python 3.11+ (typing, async/await, pattern matching)
- **Scientific Computing**: NumPy, SciPy, Pandas
- **Physiology Modeling**: SimPy (systems), NetworkX (organ networks)
- **ML/Optimization**: JAX (autodiff), scikit-learn (dimensionality reduction)
- **Statistics**: statsmodels, pingouin (for experimental design)
- **Bioinformatics**: BioPython, pyDESeq2 (transcriptomics)
- **Database**: PostgreSQL with TimescaleDB (time-series physiology data)
- **Cache**: Redis for experiment results
- **Queue**: Celery with RabbitMQ for simulation distribution

### Frontend & Visualization
- **Framework**: React 18 with TypeScript
- **3D Engine**: Three.js with React Three Fiber
- **Scientific Plots**: Plotly.js (publication quality)
- **Heatmaps**: D3.js with biological color schemes
- **State Management**: Zustand
- **UI Components**: Ant Design (scientific UI patterns)
- **Data Tables**: AG-Grid (for experimental data)

### Lab Integration Tools
- **Data Import**: Apache Arrow (efficient columnar data)
- **Equipment APIs**: PySerial (respirometry), PyVISA (instruments)
- **Statistical Export**: R integration via rpy2
- **Publication**: Matplotlib/Seaborn for figures, LaTeX for reports

## Project Structure

```
ipe-velotta/
├── core/                          # Core computation engine
│   ├── physiology/
│   │   ├── state.py              # Physiological state vectors
│   │   ├── metabolism.py         # Metabolic calculations
│   │   ├── cardiovascular.py     # Heart & circulation
│   │   ├── respiratory.py        # Lung & gas exchange
│   │   ├── thermoregulation.py   # Heat production/loss
│   │   └── osmoregulation.py     # Ion/water balance
│   ├── plasticity/
│   │   ├── reaction_norm.py      # G×E interactions
│   │   ├── maladaptive.py        # Maladaptive response detection
│   │   ├── assimilation.py       # Genetic assimilation
│   │   └── costs.py              # Plasticity costs
│   ├── games/
│   │   ├── hypoxia_game.py       # Oxygen allocation
│   │   ├── thermal_game.py       # Temperature response
│   │   ├── osmotic_game.py       # Salinity adaptation
│   │   └── resource_game.py      # Energy budgeting
│   └── environment/
│       ├── altitude.py            # Altitude-specific parameters
│       ├── aquatic.py             # Freshwater/marine
│       ├── seasonal.py            # Temporal variation
│       └── climate.py             # Climate scenarios
├── simulation/                     # Simulation engine
│   ├── rapid_evolution.py        # Contemporary evolution
│   ├── population.py             # Population dynamics
│   ├── selection.py              # Selection differentials
│   ├── gene_flow.py              # Migration & connectivity
│   └── experimental.py           # Lab experiment simulator
├── analysis/                       # Data analysis
│   ├── transcriptomics.py        # RNA-seq integration
│   ├── morphometrics.py          # Morphological analysis
│   ├── performance.py            # Whole-organism performance
│   ├── statistics.py             # Statistical tests
│   └── validation.py             # Model validation
├── lab_integration/               # Lab-specific tools
│   ├── respirometry/
│   │   ├── sable_systems.py     # Sable Systems import
│   │   ├── calculations.py       # VO2, VCO2, RER
│   │   └── calibration.py        # Sensor calibration
│   ├── molecular/
│   │   ├── rnaseq_import.py     # DESeq2/EdgeR results
│   │   ├── gene_mapping.py       # Gene ID conversion
│   │   └── pathway_analysis.py   # KEGG/GO enrichment
│   └── field_data/
│       ├── environmental.py      # Weather station data
│       ├── morphology.py         # Morphometric data
│       └── survival.py           # Mark-recapture data
├── api/                           # REST & WebSocket APIs
│   ├── rest/
│   │   ├── experiments.py        # Experiment management
│   │   ├── simulations.py        # Simulation control
│   │   ├── analysis.py           # Analysis endpoints
│   │   └── export.py             # Data export
│   └── websocket/
│       ├── realtime.py           # Live simulation updates
│       └── collaboration.py       # Multi-user sessions
├── web/                           # Frontend application
│   ├── src/
│   │   ├── components/
│   │   │   ├── AltitudeExplorer/
│   │   │   ├── PlasticityLandscape/
│   │   │   ├── OrganSystemDashboard/
│   │   │   └── ExperimentDesigner/
│   │   ├── visualizers/
│   │   │   ├── PhysiologySpace3D/
│   │   │   ├── ReactionNormPlot/
│   │   │   ├── SelectionGradient/
│   │   │   └── TranscriptomeHeatmap/
│   │   └── workflows/
│   │       ├── DeerMouseWorkflow/
│   │       ├── AlewifeWorkflow/
│   │       └── PlasticityWorkflow/
├── data/                          # Data management
│   ├── models/                   # SQLAlchemy models
│   ├── migrations/               # Database migrations
│   ├── field_sites/             # Site-specific data
│   │   ├── mount_evans.json
│   │   ├── rmnp_transects.json
│   │   └── connecticut_rivers.json
│   └── experiments/              # Experimental protocols
│       ├── common_garden.yaml
│       ├── hypoxia_challenge.yaml
│       └── thermal_gradient.yaml
└── tests/
    ├── unit/
    ├── integration/
    ├── validation/               # Model validation tests
    └── performance/
```

## Core Classes & APIs

### Physiological State Management

```python
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

class Tissue(Enum):
    HEART = "heart"
    LUNG = "lung"
    MUSCLE = "muscle"
    BROWN_FAT = "brown_fat"
    BRAIN = "brain"
    KIDNEY = "kidney"
    GILL = "gill"  # For fish

@dataclass(frozen=True)
class PhysiologicalState:
    """Complete physiological state of an organism"""
    
    # Environmental conditions
    po2: float  # Partial pressure O2 (kPa)
    temperature: float  # °C
    altitude: float  # meters
    salinity: Optional[float] = None  # ppt for aquatic
    
    # Cardiovascular
    heart_mass: float  # g/kg body mass
    hematocrit: float  # %
    hemoglobin: float  # g/dL
    blood_volume: float  # mL/kg
    cardiac_output: float  # mL/min/kg
    
    # Respiratory
    lung_volume: float  # mL/kg
    diffusion_capacity: float  # mL O2/min/mmHg/kg
    ventilation_rate: float  # breaths/min
    tidal_volume: float  # mL/kg
    
    # Metabolic
    bmr: float  # mL O2/hr/kg
    vo2max: float  # mL O2/min/kg
    respiratory_exchange_ratio: float
    mitochondrial_density: Dict[Tissue, float]
    
    # Thermoregulation
    thermal_conductance: float  # mL O2/hr/°C
    lower_critical_temp: float  # °C
    upper_critical_temp: float  # °C
    max_thermogenesis: float  # mL O2/hr/kg
    
    # Osmoregulation (for fish)
    plasma_osmolality: Optional[float] = None
    gill_na_k_atpase: Optional[float] = None
    drinking_rate: Optional[float] = None
    
    def compute_aerobic_scope(self) -> float:
        """Calculate aerobic scope (VO2max - BMR)"""
        return self.vo2max - (self.bmr / 60)  # Convert BMR to per minute
    
    def oxygen_delivery(self, tissue: Tissue) -> float:
        """Calculate O2 delivery to specific tissue"""
        blood_flow_fraction = self.tissue_perfusion[tissue]
        o2_content = self.hemoglobin * 1.34 * (self.po2 / 100)  # mL O2/dL
        return self.cardiac_output * blood_flow_fraction * o2_content

@dataclass
class PlasticityProfile:
    """Characterizes phenotypic plasticity"""
    
    reaction_norm: np.ndarray  # Shape: (n_environments, n_traits)
    reversibility: float  # 0-1, speed of reversal
    lag_time: float  # Generations to full expression
    costs: Dict[str, float]  # Maintenance, production, etc.
    
    def is_maladaptive(self, environment: np.ndarray, 
                       fitness_function: callable) -> bool:
        """Check if plastic response reduces fitness"""
        plastic_phenotype = self.predict_phenotype(environment)
        constitutive_phenotype = self.reaction_norm.mean(axis=0)
        
        plastic_fitness = fitness_function(plastic_phenotype, environment)
        constitutive_fitness = fitness_function(constitutive_phenotype, environment)
        
        return plastic_fitness < constitutive_fitness
```

### High-Altitude Specific Games

```python
from abc import ABC, abstractmethod
import numpy as np

class PhysiologicalGame(ABC):
    """Base class for physiological trade-off games"""
    
    @abstractmethod
    def compute_payoff(self, strategy: np.ndarray, 
                      environment: PhysiologicalState) -> float:
        pass
    
    @abstractmethod
    def find_optimum(self, environment: PhysiologicalState) -> np.ndarray:
        pass

class HypoxiaAllocationGame(PhysiologicalGame):
    """Optimal O2 allocation under hypoxia"""
    
    def __init__(self):
        self.tissues = [Tissue.BRAIN, Tissue.HEART, Tissue.MUSCLE, 
                       Tissue.BROWN_FAT, Tissue.KIDNEY]
        self.min_requirements = {
            Tissue.BRAIN: 0.3,      # Brain gets priority
            Tissue.HEART: 0.2,      # Heart essential
            Tissue.MUSCLE: 0.1,     # Minimum for movement
            Tissue.BROWN_FAT: 0.05, # Thermogenesis
            Tissue.KIDNEY: 0.1      # Waste removal
        }
    
    def compute_payoff(self, allocation: np.ndarray, 
                       environment: PhysiologicalState) -> float:
        """
        Calculate fitness from O2 allocation strategy
        allocation: array of fractions summing to 1
        """
        available_o2 = self.calculate_o2_availability(environment)
        
        # Ensure minimum requirements met
        for i, tissue in enumerate(self.tissues):
            if allocation[i] < self.min_requirements[tissue]:
                return -np.inf  # Death
        
        # Calculate tissue-specific performance
        brain_function = self.brain_performance(
            allocation[0] * available_o2
        )
        cardiac_function = self.cardiac_performance(
            allocation[1] * available_o2
        )
        locomotion = self.muscle_performance(
            allocation[2] * available_o2
        )
        thermogenesis = self.thermogenic_capacity(
            allocation[3] * available_o2, environment.temperature
        )
        
        # Integrate into fitness
        if environment.temperature < environment.lower_critical_temp:
            # Cold stress - thermogenesis critical
            fitness = (0.3 * brain_function + 0.2 * cardiac_function + 
                      0.1 * locomotion + 0.4 * thermogenesis)
        else:
            # Normal conditions
            fitness = (0.4 * brain_function + 0.3 * cardiac_function + 
                      0.2 * locomotion + 0.1 * thermogenesis)
        
        return fitness

class ThermogenesisTradeoffGame(PhysiologicalGame):
    """Balance heat production vs O2 consumption"""
    
    def compute_payoff(self, strategy: np.ndarray, 
                       environment: PhysiologicalState) -> float:
        """
        strategy[0]: shivering intensity (0-1)
        strategy[1]: brown fat activation (0-1)
        strategy[2]: vasoconstriction level (0-1)
        """
        # Heat production
        shivering_heat = strategy[0] * environment.max_thermogenesis * 0.6
        bat_heat = strategy[1] * environment.max_thermogenesis * 0.4
        total_heat = shivering_heat + bat_heat
        
        # O2 cost
        shivering_o2 = strategy[0] * environment.vo2max * 0.5
        bat_o2 = strategy[1] * environment.vo2max * 0.3
        total_o2_demand = environment.bmr + shivering_o2 + bat_o2
        
        # Can we sustain this O2 demand?
        o2_available = self.calculate_o2_uptake(environment)
        if total_o2_demand > o2_available:
            return -np.inf  # Unsustainable
        
        # Temperature maintenance
        heat_loss = (environment.thermal_conductance * 
                    (37 - environment.temperature) * 
                    (1 - strategy[2] * 0.3))  # Vasoconstriction reduces loss
        
        heat_balance = total_heat - heat_loss
        
        if heat_balance < 0:
            # Hypothermia penalty
            return -abs(heat_balance) / 10
        else:
            # Fitness based on O2 efficiency
            efficiency = heat_balance / total_o2_demand
            return efficiency
```

### Rapid Evolution Engine

```python
from typing import List, Tuple, Generator
import numpy as np
from scipy import stats

class RapidEvolutionSimulator:
    """Simulates contemporary evolution (10-100 generations)"""
    
    def __init__(self, population_size: int, 
                 initial_state: PhysiologicalState,
                 genetic_variance: Dict[str, float]):
        self.N = population_size
        self.state = initial_state
        self.genetic_variance = genetic_variance
        self.generation = 0
        
    def selection_differential(self, trait_values: np.ndarray, 
                              fitness_values: np.ndarray) -> float:
        """Calculate selection differential S"""
        mean_fitness = np.mean(fitness_values)
        mean_trait = np.mean(trait_values)
        
        # Weighted covariance
        cov = np.sum((trait_values - mean_trait) * 
                    (fitness_values - mean_fitness)) / self.N
        
        return cov / mean_fitness
    
    def evolve_with_plasticity(self, 
                              environment_sequence: List[PhysiologicalState],
                              plasticity_profile: PlasticityProfile,
                              generations: int) -> Generator:
        """
        Simulate evolution with phenotypic plasticity
        Tracks genetic assimilation of plastic responses
        """
        for gen in range(generations):
            env = environment_sequence[gen % len(environment_sequence)]
            
            # Plastic response
            plastic_phenotypes = self.express_plasticity(
                self.population_genotypes, env, plasticity_profile
            )
            
            # Calculate fitness
            fitness = self.calculate_fitness(plastic_phenotypes, env)
            
            # Check for maladaptive plasticity
            maladaptive_fraction = self.assess_maladaptive_responses(
                plastic_phenotypes, self.population_genotypes, env
            )
            
            # Selection
            survivors = self.select(fitness)
            
            # Genetic assimilation pressure
            if maladaptive_fraction > 0.1:
                # Stronger selection for reduced plasticity
                plasticity_modifiers = self.evolve_plasticity_modifiers(
                    survivors, plasticity_profile
                )
                plasticity_profile = self.update_plasticity(
                    plasticity_profile, plasticity_modifiers
                )
            
            # Reproduction with mutation
            self.population_genotypes = self.reproduce(survivors)
            
            yield {
                'generation': gen,
                'mean_phenotype': np.mean(plastic_phenotypes, axis=0),
                'genetic_values': np.mean(self.population_genotypes, axis=0),
                'plasticity': plasticity_profile.reaction_norm.flatten(),
                'maladaptive_fraction': maladaptive_fraction,
                'mean_fitness': np.mean(fitness)
            }

class AlewifeInvasionSimulator(RapidEvolutionSimulator):
    """Specialized simulator for freshwater invasion"""
    
    def __init__(self, founding_size: int, 
                 marine_ancestor: PhysiologicalState):
        super().__init__(founding_size, marine_ancestor, {
            'gill_na_k_atpase': 0.2,
            'drinking_rate': 0.1,
            'kidney_function': 0.15
        })
        self.marine_traits = marine_ancestor
        
    def landlocking_event(self, generations: int = 50) -> Dict:
        """Simulate rapid adaptation after landlocking"""
        freshwater_env = PhysiologicalState(
            salinity=0,  # Fresh water
            gill_na_k_atpase=self.marine_traits.gill_na_k_atpase * 0.3,
            # Other traits...
        )
        
        results = []
        for gen_data in self.evolve_with_plasticity(
            [freshwater_env], self.marine_plasticity, generations
        ):
            results.append(gen_data)
            
        return {
            'trajectory': results,
            'gill_evolution': [r['mean_phenotype'][0] for r in results],
            'swimming_performance': self.calculate_swimming_trajectory(results)
        }
```

### Transcriptomic Integration

```python
import pandas as pd
from typing import Dict, List
from scipy.stats import