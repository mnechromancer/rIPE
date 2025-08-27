"""
Genetic Assimilation Engine

This module implements genetic assimilation and canalization processes,
modeling how plasticity is reduced over evolutionary time and traits
become constitutively expressed.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import numpy as np
import json

from .reaction_norm import ReactionNorm


class AssimilationStage(Enum):
    """Stages of genetic assimilation process"""
    INITIAL = "initial"              # High plasticity, low assimilation
    PARTIAL = "partial"              # Moderate plasticity reduction
    ADVANCED = "advanced"            # Significant plasticity reduction
    COMPLETE = "complete"            # Full assimilation, no plasticity
    CANALIZED = "canalized"          # Beyond assimilation, trait locked


@dataclass
class AssimilationTrajectory:
    """
    Represents a genetic assimilation trajectory over evolutionary time.
    """
    generations: np.ndarray          # Generation numbers
    plasticity_levels: np.ndarray    # Plasticity magnitude over time
    constitutive_values: np.ndarray  # Mean constitutive phenotype over time
    assimilation_rate: float         # Rate of plasticity reduction
    environment: float               # Environment where assimilation occurs
    trait_name: str                  # Trait undergoing assimilation
    initial_plasticity: float       # Starting plasticity level
    final_plasticity: float         # Final plasticity level
    
    def __post_init__(self):
        """Validate trajectory data"""
        if len(self.generations) != len(self.plasticity_levels):
            raise ValueError("generations and plasticity_levels must have same length")
        if len(self.generations) != len(self.constitutive_values):
            raise ValueError("generations and constitutive_values must have same length")
        if not np.all(self.plasticity_levels >= 0):
            raise ValueError("plasticity_levels must be non-negative")


@dataclass
class GeneticAssimilationEngine:
    """
    Engine for modeling genetic assimilation and canalization processes.
    
    Models how plastic responses become genetically fixed over evolutionary time
    in stable environments.
    """
    
    selection_strength: float = 0.1      # Strength of selection for assimilation
    mutation_rate: float = 0.01          # Rate of new mutations affecting plasticity
    population_size: int = 1000          # Effective population size
    environmental_stability: float = 0.9 # Probability environment stays constant
    
    def __post_init__(self):
        """Validate engine parameters"""
        if not 0 < self.selection_strength <= 1:
            raise ValueError("selection_strength must be between 0 and 1")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if not 0 <= self.environmental_stability <= 1:
            raise ValueError("environmental_stability must be between 0 and 1")

    def calculate_assimilation_rate(self, initial_reaction_norm: ReactionNorm,
                                   target_environment: float,
                                   selection_coefficient: float = 0.1) -> float:
        """
        Calculate expected rate of genetic assimilation.
        
        Args:
            initial_reaction_norm: Starting reaction norm
            target_environment: Environment where assimilation occurs
            selection_coefficient: Selection strength for constitutive expression
            
        Returns:
            Assimilation rate (plasticity reduction per generation)
        """
        # Get initial plasticity level
        initial_plasticity = initial_reaction_norm.plasticity_magnitude()
        
        if initial_plasticity == 0:
            return 0.0  # Already no plasticity
        
        # Assimilation rate depends on selection strength, population size, and environmental stability
        effective_selection = selection_coefficient * self.environmental_stability
        genetic_drift_effect = 1.0 / (2 * self.population_size)
        mutation_effect = self.mutation_rate
        
        # Rate equation: depends on selection vs drift balance
        rate = effective_selection * (1 - genetic_drift_effect) - mutation_effect
        
        # Ensure non-negative rate
        return max(0.0, rate)

    def simulate_assimilation_trajectory(self, initial_reaction_norm: ReactionNorm,
                                       target_environment: float,
                                       generations: int = 1000,
                                       selection_coefficient: float = 0.1) -> AssimilationTrajectory:
        """
        Simulate genetic assimilation trajectory over evolutionary time.
        
        Args:
            initial_reaction_norm: Starting reaction norm
            target_environment: Environment where assimilation occurs
            generations: Number of generations to simulate
            selection_coefficient: Selection strength
            
        Returns:
            AssimilationTrajectory object
        """
        # Initialize arrays
        gen_array = np.arange(generations)
        plasticity_array = np.zeros(generations)
        constitutive_array = np.zeros(generations)
        
        # Get initial values
        initial_plasticity = initial_reaction_norm.plasticity_magnitude()
        target_phenotype = initial_reaction_norm.predict_phenotype(target_environment)
        
        # Calculate assimilation rate
        assim_rate = self.calculate_assimilation_rate(
            initial_reaction_norm, target_environment, selection_coefficient
        )
        
        # Simulate trajectory
        current_plasticity = initial_plasticity
        current_constitutive = target_phenotype
        
        for gen in range(generations):
            plasticity_array[gen] = current_plasticity
            constitutive_array[gen] = current_constitutive
            
            # Update plasticity (exponential decay with noise)
            plasticity_reduction = assim_rate * current_plasticity
            mutation_noise = np.random.normal(0, self.mutation_rate * 0.1)
            environmental_noise = (1 - self.environmental_stability) * np.random.normal(0, 0.05)
            
            current_plasticity = max(0, current_plasticity - plasticity_reduction + mutation_noise + environmental_noise)
            
            # Constitutive trait slowly moves toward optimal for target environment
            constitutive_shift = selection_coefficient * 0.01 * np.random.normal(0, 1)
            current_constitutive += constitutive_shift
        
        return AssimilationTrajectory(
            generations=gen_array,
            plasticity_levels=plasticity_array,
            constitutive_values=constitutive_array,
            assimilation_rate=assim_rate,
            environment=target_environment,
            trait_name=initial_reaction_norm.trait_name,
            initial_plasticity=initial_plasticity,
            final_plasticity=current_plasticity
        )

    def predict_time_to_assimilation(self, initial_reaction_norm: ReactionNorm,
                                    target_environment: float,
                                    threshold_plasticity: float = 5.0,
                                    selection_coefficient: float = 0.1) -> Optional[int]:
        """
        Predict number of generations until plasticity drops below threshold.
        
        Args:
            initial_reaction_norm: Starting reaction norm
            target_environment: Environment for assimilation
            threshold_plasticity: Plasticity threshold for "assimilated"
            selection_coefficient: Selection strength
            
        Returns:
            Generations to assimilation (None if never reaches threshold)
        """
        initial_plasticity = initial_reaction_norm.plasticity_magnitude()
        
        if initial_plasticity <= threshold_plasticity:
            return 0  # Already assimilated
        
        assim_rate = self.calculate_assimilation_rate(
            initial_reaction_norm, target_environment, selection_coefficient
        )
        
        if assim_rate <= 0:
            return None  # No assimilation expected
        
        # Exponential decay model: P(t) = P0 * exp(-rate * t)
        # Solve for t when P(t) = threshold
        if assim_rate > 0:
            generations = -np.log(threshold_plasticity / initial_plasticity) / assim_rate
            return int(np.ceil(generations))
        
        return None

    def assess_assimilation_stage(self, current_plasticity: float,
                                 initial_plasticity: float) -> AssimilationStage:
        """
        Assess current stage of genetic assimilation process.
        
        Args:
            current_plasticity: Current plasticity magnitude
            initial_plasticity: Initial plasticity magnitude
            
        Returns:
            AssimilationStage enum value
        """
        if initial_plasticity == 0:
            return AssimilationStage.CANALIZED
        
        reduction_fraction = 1 - (current_plasticity / initial_plasticity)
        
        if reduction_fraction < 0.1:
            return AssimilationStage.INITIAL
        elif reduction_fraction < 0.4:
            return AssimilationStage.PARTIAL
        elif reduction_fraction < 0.8:
            return AssimilationStage.ADVANCED
        elif reduction_fraction < 0.95:
            return AssimilationStage.COMPLETE
        else:
            return AssimilationStage.CANALIZED

    def compare_assimilation_scenarios(self, initial_reaction_norm: ReactionNorm,
                                     environments: List[float],
                                     generations: int = 500) -> Dict[float, AssimilationTrajectory]:
        """
        Compare assimilation trajectories across different environments.
        
        Args:
            initial_reaction_norm: Starting reaction norm
            environments: List of environments to compare
            generations: Simulation length
            
        Returns:
            Dictionary mapping environment to trajectory
        """
        trajectories = {}
        
        for env in environments:
            trajectory = self.simulate_assimilation_trajectory(
                initial_reaction_norm, env, generations
            )
            trajectories[env] = trajectory
        
        return trajectories

    def identify_assimilation_candidates(self, reaction_norm: ReactionNorm,
                                       environment_range: Optional[Tuple[float, float]] = None,
                                       n_environments: int = 10) -> List[Tuple[float, float]]:
        """
        Identify environments most likely to lead to genetic assimilation.
        
        Args:
            reaction_norm: Reaction norm to analyze
            environment_range: Range of environments to test
            n_environments: Number of environments to test
            
        Returns:
            List of (environment, assimilation_rate) tuples sorted by rate
        """
        if environment_range is None:
            min_env = np.min(reaction_norm.environments)
            max_env = np.max(reaction_norm.environments)
            environment_range = (min_env, max_env)
        
        min_env, max_env = environment_range
        test_environments = np.linspace(min_env, max_env, n_environments)
        
        candidates = []
        for env in test_environments:
            rate = self.calculate_assimilation_rate(reaction_norm, env)
            candidates.append((env, rate))
        
        # Sort by assimilation rate (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates

    def model_canalization_strength(self, trajectory: AssimilationTrajectory) -> float:
        """
        Calculate strength of canalization from trajectory.
        
        Args:
            trajectory: Assimilation trajectory to analyze
            
        Returns:
            Canalization strength (0-1, higher = more canalized)
        """
        if len(trajectory.plasticity_levels) < 10:
            return 0.0
        
        # Look at final 10% of trajectory
        final_section = int(0.9 * len(trajectory.plasticity_levels))
        final_plasticity = trajectory.plasticity_levels[final_section:]
        
        # Canalization strength = 1 - (variance in final plasticity)
        variance = np.var(final_plasticity)
        canalization = 1.0 / (1.0 + variance)  # Transform to 0-1 scale
        
        return min(1.0, canalization)

    def predict_evolutionary_endpoint(self, initial_reaction_norm: ReactionNorm,
                                    target_environment: float,
                                    max_generations: int = 5000) -> Dict[str, Union[float, str]]:
        """
        Predict long-term evolutionary endpoint of assimilation.
        
        Args:
            initial_reaction_norm: Starting reaction norm
            target_environment: Target environment
            max_generations: Maximum generations to simulate
            
        Returns:
            Dictionary with endpoint predictions
        """
        trajectory = self.simulate_assimilation_trajectory(
            initial_reaction_norm, target_environment, max_generations
        )
        
        final_stage = self.assess_assimilation_stage(
            trajectory.final_plasticity, trajectory.initial_plasticity
        )
        
        canalization_strength = self.model_canalization_strength(trajectory)
        
        return {
            'final_plasticity': trajectory.final_plasticity,
            'final_constitutive_value': trajectory.constitutive_values[-1],
            'assimilation_stage': final_stage.value,
            'canalization_strength': canalization_strength,
            'time_to_95_percent_reduction': self.predict_time_to_assimilation(
                initial_reaction_norm, target_environment, 
                threshold_plasticity=trajectory.initial_plasticity * 0.05
            ),
            'predicted_stable': trajectory.final_plasticity < 1.0
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
            'population_size': self.population_size,
            'environmental_stability': self.environmental_stability
        }

    def to_json(self) -> str:
        """
        Convert engine to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'GeneticAssimilationEngine':
        """
        Create engine from dictionary.
        
        Args:
            data: Dictionary with engine parameters
            
        Returns:
            New GeneticAssimilationEngine instance
        """
        return cls(
            selection_strength=data['selection_strength'],
            mutation_rate=data['mutation_rate'],
            population_size=data['population_size'],
            environmental_stability=data['environmental_stability']
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'GeneticAssimilationEngine':
        """
        Create engine from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            New GeneticAssimilationEngine instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)