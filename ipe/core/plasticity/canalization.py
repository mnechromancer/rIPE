"""
Canalization Module

This module implements canalization processes, modeling how developmental
pathways become buffered against genetic and environmental perturbations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import numpy as np
import json

from .reaction_norm import ReactionNorm
from .assimilation import AssimilationTrajectory


class CanalizationType(Enum):
    """Types of canalization"""
    GENETIC = "genetic"              # Buffering against genetic variation
    ENVIRONMENTAL = "environmental"  # Buffering against environmental variation
    DEVELOPMENTAL = "developmental"  # Buffering against developmental noise
    PHENOTYPIC = "phenotypic"       # Overall phenotypic stability


@dataclass
class CanalizationMeasure:
    """
    Quantitative measure of canalization strength.
    """
    canalization_type: CanalizationType  # Type of canalization measured
    strength: float                      # Canalization strength (0-1)
    trait_name: str                      # Trait being measured
    environment: Optional[float] = None  # Environment (if environmental canalization)
    variance_component: float = 0.0      # Variance explained by canalization
    
    def __post_init__(self):
        """Validate canalization measure"""
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if self.variance_component < 0:
            raise ValueError("variance_component must be non-negative")


@dataclass
class CanalizationTrajectory:
    """
    Trajectory of canalization over evolutionary time.
    """
    generations: np.ndarray              # Generation numbers
    canalization_strength: np.ndarray   # Canalization strength over time
    phenotypic_variance: np.ndarray     # Phenotypic variance over time
    environmental_sensitivity: np.ndarray  # Environmental sensitivity over time
    canalization_rate: float            # Rate of canalization increase
    trait_name: str                     # Trait being canalized
    
    def __post_init__(self):
        """Validate trajectory data"""
        arrays = [self.generations, self.canalization_strength, 
                 self.phenotypic_variance, self.environmental_sensitivity]
        
        if not all(len(arr) == len(arrays[0]) for arr in arrays):
            raise ValueError("All arrays must have same length")
        
        if not np.all(self.canalization_strength >= 0):
            raise ValueError("canalization_strength must be non-negative")
        if not np.all(self.phenotypic_variance >= 0):
            raise ValueError("phenotypic_variance must be non-negative")


@dataclass  
class CanalizationEngine:
    """
    Engine for modeling canalization processes.
    
    Models how developmental pathways become buffered against
    genetic and environmental perturbations over evolutionary time.
    """
    
    selection_strength: float = 0.05     # Selection for canalization
    mutation_rate: float = 0.01          # Mutation rate affecting canalization
    developmental_noise: float = 0.1     # Level of developmental noise
    population_size: int = 1000          # Effective population size
    
    def __post_init__(self):
        """Validate engine parameters"""
        if not 0 < self.selection_strength <= 1:
            raise ValueError("selection_strength must be between 0 and 1")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1") 
        if self.developmental_noise < 0:
            raise ValueError("developmental_noise must be non-negative")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")

    def measure_genetic_canalization(self, reaction_norms: List[ReactionNorm],
                                   environment: float) -> CanalizationMeasure:
        """
        Measure genetic canalization (buffering against genetic variation).
        
        Args:
            reaction_norms: List of reaction norms representing genetic variants
            environment: Environment to measure canalization at
            
        Returns:
            CanalizationMeasure for genetic canalization
        """
        if len(reaction_norms) < 2:
            raise ValueError("Need at least 2 reaction norms for genetic canalization")
        
        # Get phenotypes of all genotypes at the specified environment
        phenotypes = []
        for norm in reaction_norms:
            phenotype = norm.predict_phenotype(environment)
            phenotypes.append(phenotype)
        
        phenotypes = np.array(phenotypes)
        
        # Genetic canalization = 1 - (variance among genotypes)
        phenotypic_variance = np.var(phenotypes)
        mean_phenotype = np.mean(phenotypes)
        
        # Normalize variance by mean to get coefficient of variation
        if mean_phenotype != 0:
            cv = np.sqrt(phenotypic_variance) / abs(mean_phenotype)
            genetic_canalization = 1.0 / (1.0 + cv)  # Transform to 0-1 scale
        else:
            genetic_canalization = 1.0 if phenotypic_variance == 0 else 0.0
        
        return CanalizationMeasure(
            canalization_type=CanalizationType.GENETIC,
            strength=genetic_canalization,
            trait_name=reaction_norms[0].trait_name,
            environment=environment,
            variance_component=phenotypic_variance
        )

    def measure_environmental_canalization(self, reaction_norm: ReactionNorm,
                                         environment_range: Optional[Tuple[float, float]] = None,
                                         n_environments: int = 20) -> CanalizationMeasure:
        """
        Measure environmental canalization (buffering against environmental variation).
        
        Args:
            reaction_norm: Single reaction norm to analyze
            environment_range: Range of environments to test
            n_environments: Number of environments to sample
            
        Returns:
            CanalizationMeasure for environmental canalization
        """
        if environment_range is None:
            min_env = np.min(reaction_norm.environments)
            max_env = np.max(reaction_norm.environments)
            environment_range = (min_env, max_env)
        
        min_env, max_env = environment_range
        test_environments = np.linspace(min_env, max_env, n_environments)
        
        # Get phenotypes across environmental range
        phenotypes = []
        for env in test_environments:
            phenotype = reaction_norm.predict_phenotype(env)
            phenotypes.append(phenotype)
        
        phenotypes = np.array(phenotypes)
        
        # Environmental canalization = 1 - (environmental sensitivity)
        env_variance = np.var(phenotypes)
        mean_phenotype = np.mean(phenotypes)
        
        if mean_phenotype != 0:
            env_cv = np.sqrt(env_variance) / abs(mean_phenotype)
            env_canalization = 1.0 / (1.0 + env_cv)
        else:
            env_canalization = 1.0 if env_variance == 0 else 0.0
        
        return CanalizationMeasure(
            canalization_type=CanalizationType.ENVIRONMENTAL,
            strength=env_canalization,
            trait_name=reaction_norm.trait_name,
            variance_component=env_variance
        )

    def measure_developmental_canalization(self, mean_phenotype: float,
                                         phenotypic_variance: float,
                                         trait_name: str) -> CanalizationMeasure:
        """
        Measure developmental canalization (buffering against developmental noise).
        
        Args:
            mean_phenotype: Mean phenotype value
            phenotypic_variance: Variance due to developmental noise
            trait_name: Name of trait
            
        Returns:
            CanalizationMeasure for developmental canalization
        """
        # Developmental canalization inversely related to variance
        if mean_phenotype != 0:
            cv = np.sqrt(phenotypic_variance) / abs(mean_phenotype)
            dev_canalization = 1.0 / (1.0 + cv)
        else:
            dev_canalization = 1.0 if phenotypic_variance == 0 else 0.0
        
        return CanalizationMeasure(
            canalization_type=CanalizationType.DEVELOPMENTAL,
            strength=dev_canalization,
            trait_name=trait_name,
            variance_component=phenotypic_variance
        )

    def simulate_canalization_evolution(self, initial_variance: float,
                                      trait_name: str,
                                      generations: int = 1000) -> CanalizationTrajectory:
        """
        Simulate evolution of canalization over time.
        
        Args:
            initial_variance: Starting phenotypic variance
            trait_name: Name of trait being canalized
            generations: Number of generations to simulate
            
        Returns:
            CanalizationTrajectory object
        """
        # Initialize arrays
        gen_array = np.arange(generations)
        canalization_array = np.zeros(generations)
        variance_array = np.zeros(generations)
        sensitivity_array = np.zeros(generations)
        
        # Starting values
        current_variance = initial_variance
        current_canalization = 1.0 / (1.0 + np.sqrt(current_variance))
        
        # Calculate canalization evolution rate
        selection_effect = self.selection_strength * 0.01
        drift_effect = 1.0 / (2 * self.population_size)
        mutation_effect = self.mutation_rate * 0.001
        
        canalization_rate = selection_effect - drift_effect - mutation_effect
        
        # Simulate trajectory
        for gen in range(generations):
            canalization_array[gen] = current_canalization
            variance_array[gen] = current_variance
            sensitivity_array[gen] = 1.0 - current_canalization
            
            # Update canalization (increases with selection, decreases with mutation/drift)
            canalization_change = canalization_rate * (1 - current_canalization)
            noise = np.random.normal(0, 0.001)  # Small random fluctuations
            
            current_canalization = np.clip(current_canalization + canalization_change + noise, 0, 1)
            
            # Variance inversely related to canalization
            current_variance = initial_variance * (1 - current_canalization)
        
        return CanalizationTrajectory(
            generations=gen_array,
            canalization_strength=canalization_array,
            phenotypic_variance=variance_array,
            environmental_sensitivity=sensitivity_array,
            canalization_rate=canalization_rate,
            trait_name=trait_name
        )

    def predict_canalization_trajectory(self, assimilation_trajectory: AssimilationTrajectory) -> CanalizationTrajectory:
        """
        Predict canalization trajectory from genetic assimilation.
        
        Args:
            assimilation_trajectory: AssimilationTrajectory from genetic assimilation
            
        Returns:
            Predicted CanalizationTrajectory
        """
        generations = assimilation_trajectory.generations
        
        # Canalization strength increases as plasticity decreases
        initial_plasticity = assimilation_trajectory.initial_plasticity
        plasticity_levels = assimilation_trajectory.plasticity_levels
        
        # Convert plasticity to canalization (inverse relationship)
        canalization_strength = np.zeros_like(plasticity_levels)
        
        for i, plasticity in enumerate(plasticity_levels):
            if initial_plasticity > 0:
                reduction_fraction = 1 - (plasticity / initial_plasticity)
                canalization_strength[i] = reduction_fraction
            else:
                canalization_strength[i] = 1.0
        
        # Phenotypic variance decreases with canalization
        initial_variance = initial_plasticity / 100.0  # Scaled conversion
        phenotypic_variance = initial_variance * (1 - canalization_strength)
        
        # Environmental sensitivity = 1 - canalization
        environmental_sensitivity = 1.0 - canalization_strength
        
        # Calculate effective canalization rate
        if len(canalization_strength) > 1:
            canalization_rate = np.mean(np.diff(canalization_strength))
        else:
            canalization_rate = 0.0
        
        return CanalizationTrajectory(
            generations=generations,
            canalization_strength=canalization_strength,
            phenotypic_variance=phenotypic_variance,
            environmental_sensitivity=environmental_sensitivity,
            canalization_rate=canalization_rate,
            trait_name=assimilation_trajectory.trait_name
        )

    def compare_canalization_types(self, reaction_norms: List[ReactionNorm],
                                 environment: float,
                                 developmental_variance: float = 0.1) -> Dict[str, CanalizationMeasure]:
        """
        Compare different types of canalization for the same trait.
        
        Args:
            reaction_norms: List of reaction norms (genetic variants)
            environment: Environment for measurement
            developmental_variance: Variance due to developmental noise
            
        Returns:
            Dictionary of canalization measures by type
        """
        measures = {}
        
        if len(reaction_norms) >= 2:
            # Genetic canalization
            genetic_measure = self.measure_genetic_canalization(reaction_norms, environment)
            measures['genetic'] = genetic_measure
        
        # Environmental canalization (use first reaction norm)
        if reaction_norms:
            env_measure = self.measure_environmental_canalization(reaction_norms[0])
            measures['environmental'] = env_measure
            
            # Developmental canalization
            mean_phenotype = reaction_norms[0].predict_phenotype(environment)
            dev_measure = self.measure_developmental_canalization(
                mean_phenotype, developmental_variance, reaction_norms[0].trait_name
            )
            measures['developmental'] = dev_measure
            
            # Overall phenotypic canalization (combination of all types)
            total_variance = (genetic_measure.variance_component if 'genetic' in measures else 0) + \
                           env_measure.variance_component + developmental_variance
            
            phenotypic_measure = CanalizationMeasure(
                canalization_type=CanalizationType.PHENOTYPIC,
                strength=1.0 / (1.0 + np.sqrt(total_variance)) if total_variance > 0 else 1.0,
                trait_name=reaction_norms[0].trait_name,
                environment=environment,
                variance_component=total_variance
            )
            measures['phenotypic'] = phenotypic_measure
        
        return measures

    def assess_canalization_evolution(self, trajectory: CanalizationTrajectory,
                                    time_points: Optional[List[int]] = None) -> Dict[str, Union[float, str]]:
        """
        Assess the evolutionary dynamics of canalization.
        
        Args:
            trajectory: CanalizationTrajectory to analyze
            time_points: Specific time points to analyze (uses quartiles if None)
            
        Returns:
            Dictionary with canalization evolution metrics
        """
        if time_points is None:
            n_gens = len(trajectory.generations)
            time_points = [0, n_gens//4, n_gens//2, 3*n_gens//4, n_gens-1]
        
        # Extract canalization at time points
        canalization_values = [trajectory.canalization_strength[i] for i in time_points if i < len(trajectory.canalization_strength)]
        
        # Calculate metrics
        initial_canalization = canalization_values[0] if canalization_values else 0
        final_canalization = canalization_values[-1] if canalization_values else 0
        max_canalization = np.max(trajectory.canalization_strength)
        mean_canalization = np.mean(trajectory.canalization_strength)
        
        # Evolution rate
        if len(canalization_values) > 1:
            total_change = final_canalization - initial_canalization
            total_time = trajectory.generations[-1] - trajectory.generations[0]
            evolution_rate = total_change / total_time if total_time > 0 else 0
        else:
            evolution_rate = 0
        
        # Stability assessment
        final_third = int(2 * len(trajectory.canalization_strength) / 3)
        final_variance = np.var(trajectory.canalization_strength[final_third:])
        stable = final_variance < 0.001
        
        return {
            'initial_canalization': initial_canalization,
            'final_canalization': final_canalization,
            'max_canalization': max_canalization,
            'mean_canalization': mean_canalization,
            'canalization_evolution_rate': evolution_rate,
            'final_variance_reduction': 1 - trajectory.phenotypic_variance[-1] / trajectory.phenotypic_variance[0] if trajectory.phenotypic_variance[0] > 0 else 0,
            'evolutionary_stability': 'stable' if stable else 'unstable',
            'canalization_efficiency': final_canalization / max_canalization if max_canalization > 0 else 0
        }

    def identify_canalization_drivers(self, trajectory: CanalizationTrajectory) -> Dict[str, float]:
        """
        Identify the main drivers of canalization evolution.
        
        Args:
            trajectory: CanalizationTrajectory to analyze
            
        Returns:
            Dictionary with driver contributions
        """
        # Analyze different components affecting canalization evolution
        selection_contribution = self.selection_strength * 0.8  # Selection is primary driver
        
        drift_effect = 1.0 / (2 * self.population_size)
        drift_contribution = min(0.3, drift_effect)  # Genetic drift opposes canalization
        
        mutation_contribution = self.mutation_rate * 0.2  # Mutation opposes canalization
        
        developmental_contribution = max(0, 0.5 - self.developmental_noise)  # Low noise favors canalization
        
        # Normalize contributions
        total = selection_contribution + drift_contribution + mutation_contribution + developmental_contribution
        
        if total > 0:
            return {
                'selection': selection_contribution / total,
                'genetic_drift': drift_contribution / total,
                'mutation': mutation_contribution / total,
                'developmental_buffering': developmental_contribution / total
            }
        else:
            return {
                'selection': 0.25,
                'genetic_drift': 0.25,
                'mutation': 0.25,
                'developmental_buffering': 0.25
            }

    def to_dict(self) -> Dict:
        """
        Convert engine to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            'selection_strength': self.selection_strength,
            'mutation_rate': self.mutation_rate,
            'developmental_noise': self.developmental_noise,
            'population_size': self.population_size
        }

    def to_json(self) -> str:
        """
        Convert engine to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'CanalizationEngine':
        """
        Create engine from dictionary.
        
        Args:
            data: Dictionary with engine parameters
            
        Returns:
            New CanalizationEngine instance
        """
        return cls(
            selection_strength=data['selection_strength'],
            mutation_rate=data['mutation_rate'],
            developmental_noise=data['developmental_noise'],
            population_size=data['population_size']
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'CanalizationEngine':
        """
        Create engine from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            New CanalizationEngine instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)