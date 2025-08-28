"""
Maladaptive Plasticity Detection Module

This module implements algorithms for detecting and quantifying maladaptive
plastic responses where plasticity reduces rather than improves fitness.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple, Union
from enum import Enum
import numpy as np
import json

from .reaction_norm import ReactionNorm, PlasticityMagnitude
from ..physiology.state import PhysiologicalState


class MaladaptationType(Enum):
    """Types of maladaptive plasticity"""

    NONE = "none"  # No maladaptive plasticity detected
    SLIGHT = "slight"  # Small fitness reduction < 5%
    MODERATE = "moderate"  # Fitness reduction 5-15%
    SEVERE = "severe"  # Fitness reduction 15-30%
    EXTREME = "extreme"  # Fitness reduction > 30%


@dataclass
class MaladaptiveResponse:
    """
    Represents a detected maladaptive plastic response.
    """

    environment: float  # Environmental condition where maladaptive
    plastic_phenotype: float  # Phenotype expressed by plastic genotype
    optimal_phenotype: float  # Theoretically optimal phenotype
    plastic_fitness: float  # Fitness of plastic response
    optimal_fitness: float  # Fitness of optimal response
    fitness_cost: float  # Fitness reduction due to maladaptation
    relative_fitness_cost: float  # Fitness cost as proportion of optimum
    environment_variable: str  # Name of environmental variable
    trait_name: str  # Name of trait showing maladaptation


@dataclass
class MaladaptiveDetector:
    """
    Detector for maladaptive plasticity in reaction norms.

    Compares plastic responses to theoretical optima and identifies
    environments where plasticity reduces fitness.
    """

    fitness_function: Callable[
        [float, float], float
    ]  # (phenotype, environment) -> fitness
    environment_variable: str  # Environmental variable name
    trait_name: str  # Trait being analyzed
    optimization_tolerance: float = 0.01  # Tolerance for optimization

    def __post_init__(self):
        """Validate detector parameters"""
        if not callable(self.fitness_function):
            raise TypeError("fitness_function must be callable")
        if self.optimization_tolerance <= 0:
            raise ValueError("optimization_tolerance must be positive")

    def compute_optimal_phenotype(
        self, environment: float, phenotype_range: Optional[Tuple[float, float]] = None
    ) -> float:
        """
        Compute theoretically optimal phenotype for given environment.

        Args:
            environment: Environmental condition
            phenotype_range: Optional (min, max) bounds for phenotype optimization

        Returns:
            Optimal phenotype value
        """
        if phenotype_range is None:
            # Default search range
            phenotype_range = (-100, 100)

        min_pheno, max_pheno = phenotype_range

        # Grid search for optimal phenotype
        n_points = int((max_pheno - min_pheno) / self.optimization_tolerance) + 1
        n_points = min(n_points, 10000)  # Limit computation

        phenotypes = np.linspace(min_pheno, max_pheno, n_points)
        fitness_values = np.array(
            [self.fitness_function(p, environment) for p in phenotypes]
        )

        optimal_idx = np.argmax(fitness_values)
        return phenotypes[optimal_idx]

    def detect_maladaptive_responses(
        self,
        reaction_norm: ReactionNorm,
        environments: Optional[np.ndarray] = None,
        phenotype_range: Optional[Tuple[float, float]] = None,
    ) -> List[MaladaptiveResponse]:
        """
        Detect maladaptive responses across environmental range.

        Args:
            reaction_norm: ReactionNorm to analyze
            environments: Environmental conditions to test (uses norm range if None)
            phenotype_range: Search range for optimal phenotypes

        Returns:
            List of detected maladaptive responses
        """
        if environments is None:
            # Use reaction norm's environmental range
            min_env = np.min(reaction_norm.environments)
            max_env = np.max(reaction_norm.environments)
            environments = np.linspace(min_env, max_env, 20)

        maladaptive_responses = []

        for env in environments:
            # Get plastic phenotype from reaction norm
            plastic_phenotype = reaction_norm.predict_phenotype(env)

            # Compute optimal phenotype for this environment
            optimal_phenotype = self.compute_optimal_phenotype(env, phenotype_range)

            # Calculate fitness for both
            plastic_fitness = self.fitness_function(plastic_phenotype, env)
            optimal_fitness = self.fitness_function(optimal_phenotype, env)

            # Check if plastic response is maladaptive
            if plastic_fitness < optimal_fitness:
                fitness_cost = optimal_fitness - plastic_fitness
                relative_cost = (
                    fitness_cost / optimal_fitness if optimal_fitness > 0 else 0
                )

                response = MaladaptiveResponse(
                    environment=env,
                    plastic_phenotype=plastic_phenotype,
                    optimal_phenotype=optimal_phenotype,
                    plastic_fitness=plastic_fitness,
                    optimal_fitness=optimal_fitness,
                    fitness_cost=fitness_cost,
                    relative_fitness_cost=relative_cost,
                    environment_variable=self.environment_variable,
                    trait_name=self.trait_name,
                )

                maladaptive_responses.append(response)

        return maladaptive_responses

    def quantify_maladaptation_severity(
        self, reaction_norm: ReactionNorm, environments: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Quantify overall severity of maladaptive plasticity.

        Args:
            reaction_norm: ReactionNorm to analyze
            environments: Environmental conditions to test

        Returns:
            Dictionary with maladaptation metrics
        """
        responses = self.detect_maladaptive_responses(reaction_norm, environments)

        if not responses:
            return {
                "proportion_maladaptive": 0.0,
                "mean_fitness_cost": 0.0,
                "max_fitness_cost": 0.0,
                "mean_relative_cost": 0.0,
                "max_relative_cost": 0.0,
                "severity_classification": MaladaptationType.NONE.value,
            }

        n_environments = len(environments) if environments is not None else 20

        fitness_costs = [r.fitness_cost for r in responses]
        relative_costs = [r.relative_fitness_cost for r in responses]

        metrics = {
            "proportion_maladaptive": len(responses) / n_environments,
            "mean_fitness_cost": np.mean(fitness_costs),
            "max_fitness_cost": np.max(fitness_costs),
            "mean_relative_cost": np.mean(relative_costs),
            "max_relative_cost": np.max(relative_costs),
        }

        # Classify severity based on mean relative cost
        mean_rel_cost = metrics["mean_relative_cost"] * 100  # Convert to percentage
        if mean_rel_cost < 5.0:
            severity = MaladaptationType.SLIGHT
        elif mean_rel_cost < 15.0:
            severity = MaladaptationType.MODERATE
        elif mean_rel_cost < 30.0:
            severity = MaladaptationType.SEVERE
        else:
            severity = MaladaptationType.EXTREME

        metrics["severity_classification"] = severity.value

        return metrics

    def compare_plastic_vs_constitutive(
        self,
        reaction_norm: ReactionNorm,
        constitutive_phenotype: float,
        environments: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[float, List[float]]]:
        """
        Compare plastic vs constitutive strategy fitness.

        Args:
            reaction_norm: Plastic reaction norm
            constitutive_phenotype: Fixed phenotype for constitutive strategy
            environments: Environmental conditions to test

        Returns:
            Comparison metrics
        """
        if environments is None:
            min_env = np.min(reaction_norm.environments)
            max_env = np.max(reaction_norm.environments)
            environments = np.linspace(min_env, max_env, 20)

        plastic_fitness = []
        constitutive_fitness = []

        for env in environments:
            plastic_pheno = reaction_norm.predict_phenotype(env)

            plastic_fit = self.fitness_function(plastic_pheno, env)
            const_fit = self.fitness_function(constitutive_phenotype, env)

            plastic_fitness.append(plastic_fit)
            constitutive_fitness.append(const_fit)

        plastic_fitness = np.array(plastic_fitness)
        constitutive_fitness = np.array(constitutive_fitness)

        return {
            "mean_plastic_fitness": np.mean(plastic_fitness),
            "mean_constitutive_fitness": np.mean(constitutive_fitness),
            "plastic_advantage": np.mean(plastic_fitness)
            - np.mean(constitutive_fitness),
            "proportion_plastic_better": np.mean(
                plastic_fitness > constitutive_fitness
            ),
            "max_plastic_advantage": np.max(plastic_fitness - constitutive_fitness),
            "max_plastic_disadvantage": np.min(plastic_fitness - constitutive_fitness),
            "environments": environments.tolist(),
            "plastic_fitness_by_env": plastic_fitness.tolist(),
            "constitutive_fitness_by_env": constitutive_fitness.tolist(),
        }

    def identify_maladaptive_environments(
        self,
        reaction_norm: ReactionNorm,
        threshold: float = 0.05,
        environments: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        Identify specific environments where plasticity is maladaptive.

        Args:
            reaction_norm: ReactionNorm to analyze
            threshold: Minimum relative fitness cost to flag as maladaptive
            environments: Environmental conditions to test

        Returns:
            List of environmental values where plasticity is maladaptive
        """
        responses = self.detect_maladaptive_responses(reaction_norm, environments)

        maladaptive_envs = []
        for response in responses:
            if response.relative_fitness_cost >= threshold:
                maladaptive_envs.append(response.environment)

        return sorted(maladaptive_envs)

    def compute_plasticity_cost_function(
        self, reaction_norm: ReactionNorm, environments: Optional[np.ndarray] = None
    ) -> ReactionNorm:
        """
        Compute cost function showing fitness reduction due to maladaptation.

        Args:
            reaction_norm: ReactionNorm to analyze
            environments: Environmental conditions to evaluate

        Returns:
            New ReactionNorm with fitness costs as phenotypes
        """
        if environments is None:
            environments = reaction_norm.environments.copy()

        fitness_costs = []

        for env in environments:
            plastic_phenotype = reaction_norm.predict_phenotype(env)
            optimal_phenotype = self.compute_optimal_phenotype(env)

            plastic_fitness = self.fitness_function(plastic_phenotype, env)
            optimal_fitness = self.fitness_function(optimal_phenotype, env)

            cost = max(0, optimal_fitness - plastic_fitness)  # Only positive costs
            fitness_costs.append(cost)

        return ReactionNorm(
            environments=environments,
            phenotypes=np.array(fitness_costs),
            trait_name=f"{reaction_norm.trait_name}_fitness_cost",
            environmental_variable=reaction_norm.environmental_variable,
            genotype_id=reaction_norm.genotype_id,
        )

    def to_dict(self) -> Dict:
        """
        Convert detector to dictionary (excluding function).

        Returns:
            Dictionary representation
        """
        return {
            "environment_variable": self.environment_variable,
            "trait_name": self.trait_name,
            "optimization_tolerance": self.optimization_tolerance,
        }

    def to_json(self) -> str:
        """
        Convert detector to JSON string (excluding function).

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())


# Predefined fitness functions for common use cases


def quadratic_fitness(
    phenotype: float,
    environment: float,
    optimum_slope: float = 1.0,
    width: float = 10.0,
) -> float:
    """
    Quadratic fitness function with environment-dependent optimum.

    Args:
        phenotype: Phenotypic value
        environment: Environmental value
        optimum_slope: How optimal phenotype changes with environment
        width: Width of fitness peak (smaller = more selection)

    Returns:
        Fitness value
    """
    optimal_phenotype = optimum_slope * environment
    deviation = (phenotype - optimal_phenotype) ** 2
    fitness = np.exp(-deviation / width)
    return fitness


def linear_fitness(
    phenotype: float, environment: float, slope: float = 1.0, intercept: float = 0.0
) -> float:
    """
    Linear fitness function.

    Args:
        phenotype: Phenotypic value
        environment: Environmental value (not used in linear case)
        slope: Slope of fitness-phenotype relationship
        intercept: Fitness intercept

    Returns:
        Fitness value
    """
    return slope * phenotype + intercept


def thermal_fitness(phenotype: float, environment: float) -> float:
    """
    Thermal adaptation fitness function.

    Args:
        phenotype: Body size or thermal trait
        environment: Temperature

    Returns:
        Thermal fitness
    """
    # Optimal phenotype increases with temperature
    optimal = environment * 0.5 + 10
    deviation = abs(phenotype - optimal)
    # Gaussian fitness peak
    fitness = np.exp(-(deviation**2) / 20)
    return fitness
