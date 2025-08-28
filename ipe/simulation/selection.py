"""
Selection Mechanisms

This module implements various selection strategies for evolutionary simulations,
including truncation, proportional, tournament, and frequency-dependent selection.
"""

from typing import List, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

from .population import Individual


@dataclass
class SelectionDifferential:
    """
    Represents selection differential for a trait.

    Selection differential (S) measures the difference between
    the mean trait value before and after selection.
    """

    trait_name: str
    before_mean: float
    after_mean: float
    selection_differential: float
    selection_intensity: float


class SelectionStrategy(ABC):
    """
    Abstract base class for selection strategies.

    All selection strategies must implement the select method
    that takes a list of individuals and returns survivors.
    """

    @abstractmethod
    def select(
        self, individuals: List[Individual], num_survivors: Optional[int] = None
    ) -> List[Individual]:
        """
        Select survivors from population.

        Args:
            individuals: List of individuals to select from
            num_survivors: Number of survivors (if None, uses strategy default)

        Returns:
            List of selected individuals
        """

    def calculate_selection_differential(
        self,
        before: List[Individual],
        after: List[Individual],
        trait_extractor: Callable[[Individual], float],
        trait_name: str = "fitness",
    ) -> SelectionDifferential:
        """
        Calculate selection differential for a trait.

        Args:
            before: Individuals before selection
            after: Individuals after selection
            trait_extractor: Function to extract trait value from individual
            trait_name: Name of the trait

        Returns:
            SelectionDifferential object
        """
        if not before or not after:
            return SelectionDifferential(trait_name, 0.0, 0.0, 0.0, 0.0)

        before_values = [trait_extractor(ind) for ind in before]
        after_values = [trait_extractor(ind) for ind in after]

        before_mean = np.mean(before_values)
        after_mean = np.mean(after_values)
        selection_diff = after_mean - before_mean

        # Selection intensity = S / σ_P (standardized selection differential)
        before_std = np.std(before_values)
        selection_intensity = selection_diff / before_std if before_std > 0 else 0.0

        return SelectionDifferential(
            trait_name=trait_name,
            before_mean=before_mean,
            after_mean=after_mean,
            selection_differential=selection_diff,
            selection_intensity=selection_intensity,
        )


class TruncationSelection(SelectionStrategy):
    """
    Truncation selection: keep top N individuals by fitness.
    """

    def __init__(self, survival_fraction: float = 0.5):
        """
        Initialize truncation selection.

        Args:
            survival_fraction: Fraction of population to keep (0-1)
        """
        if not 0 < survival_fraction <= 1:
            raise ValueError("survival_fraction must be between 0 and 1")
        self.survival_fraction = survival_fraction

    def select(
        self, individuals: List[Individual], num_survivors: Optional[int] = None
    ) -> List[Individual]:
        """Select top individuals by fitness"""
        if not individuals:
            return []

        if num_survivors is None:
            num_survivors = max(1, int(len(individuals) * self.survival_fraction))

        # Sort by fitness (descending) and take top N
        sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_individuals[:num_survivors]


class ProportionalSelection(SelectionStrategy):
    """
    Proportional (fitness proportionate) selection: probability ∝ fitness.
    """

    def __init__(self, replacement: bool = True):
        """
        Initialize proportional selection.

        Args:
            replacement: Whether to allow sampling with replacement
        """
        self.replacement = replacement

    def select(
        self, individuals: List[Individual], num_survivors: Optional[int] = None
    ) -> List[Individual]:
        """Select individuals with probability proportional to fitness"""
        if not individuals:
            return []

        if num_survivors is None:
            num_survivors = len(individuals) // 2

        # Handle negative fitness by shifting
        fitnesses = np.array([ind.fitness for ind in individuals])
        min_fitness = np.min(fitnesses)
        if min_fitness <= 0:
            fitnesses = fitnesses - min_fitness + 1e-6

        # Calculate selection probabilities
        total_fitness = np.sum(fitnesses)
        if total_fitness == 0:
            # Uniform selection if all fitness is zero
            probabilities = np.ones(len(individuals)) / len(individuals)
        else:
            probabilities = fitnesses / total_fitness

        # Sample individuals
        indices = np.random.choice(
            len(individuals),
            size=num_survivors,
            replace=self.replacement,
            p=probabilities,
        )

        return [individuals[i] for i in indices]


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection: randomly sample k individuals, keep best.
    """

    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection.

        Args:
            tournament_size: Number of individuals per tournament
        """
        if tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")
        self.tournament_size = tournament_size

    def select(
        self, individuals: List[Individual], num_survivors: Optional[int] = None
    ) -> List[Individual]:
        """Select individuals via tournaments"""
        if not individuals:
            return []

        if num_survivors is None:
            num_survivors = len(individuals) // 2

        survivors = []
        for _ in range(num_survivors):
            # Random tournament
            tournament_indices = np.random.choice(
                len(individuals),
                size=min(self.tournament_size, len(individuals)),
                replace=False,
            )
            tournament = [individuals[i] for i in tournament_indices]

            # Select best from tournament
            winner = max(tournament, key=lambda x: x.fitness)
            survivors.append(winner)

        return survivors


class FrequencyDependentSelection(SelectionStrategy):
    """
    Frequency-dependent selection: fitness depends on trait frequency.
    """

    def __init__(
        self,
        trait_extractor: Callable[[Individual], Any],
        fitness_function: Callable[[Any, Dict[Any, float]], float],
    ):
        """
        Initialize frequency-dependent selection.

        Args:
            trait_extractor: Function to extract trait from individual
            fitness_function: Function(trait, frequencies) -> fitness
        """
        self.trait_extractor = trait_extractor
        self.fitness_function = fitness_function

    def select(
        self, individuals: List[Individual], num_survivors: Optional[int] = None
    ) -> List[Individual]:
        """Select with frequency-dependent fitness"""
        if not individuals:
            return []

        if num_survivors is None:
            num_survivors = len(individuals) // 2

        # Calculate trait frequencies
        traits = [self.trait_extractor(ind) for ind in individuals]
        unique_traits, counts = np.unique(traits, return_counts=True)
        frequencies = {
            trait: count / len(traits) for trait, count in zip(unique_traits, counts)
        }

        # Recalculate fitness based on frequencies
        for individual in individuals:
            trait = self.trait_extractor(individual)
            individual.fitness = self.fitness_function(trait, frequencies)

        # Use proportional selection with new fitness values
        proportional_selector = ProportionalSelection()
        return proportional_selector.select(individuals, num_survivors)


class MultiTraitSelection(SelectionStrategy):
    """
    Multi-trait selection: select based on multiple correlated traits.
    """

    def __init__(
        self,
        trait_extractors: Dict[str, Callable[[Individual], float]],
        trait_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-trait selection.

        Args:
            trait_extractors: Dict mapping trait names to extraction functions
            trait_weights: Optional weights for each trait (default: equal)
        """
        self.trait_extractors = trait_extractors
        self.trait_weights = trait_weights or {
            name: 1.0 for name in trait_extractors.keys()
        }

    def select(
        self, individuals: List[Individual], num_survivors: Optional[int] = None
    ) -> List[Individual]:
        """Select based on weighted combination of traits"""
        if not individuals:
            return []

        if num_survivors is None:
            num_survivors = len(individuals) // 2

        # Calculate composite fitness
        for individual in individuals:
            composite_fitness = 0.0
            total_weight = 0.0

            for trait_name, extractor in self.trait_extractors.items():
                trait_value = extractor(individual)
                weight = self.trait_weights.get(trait_name, 1.0)
                composite_fitness += trait_value * weight
                total_weight += weight

            individual.fitness = (
                composite_fitness / total_weight if total_weight > 0 else 0.0
            )

        # Use truncation selection with new fitness values
        truncation_selector = TruncationSelection(
            survival_fraction=num_survivors / len(individuals)
        )
        return truncation_selector.select(individuals, num_survivors)


class SelectionAnalyzer:
    """
    Analyzes selection outcomes and provides statistics.
    """

    @staticmethod
    def analyze_selection(
        before: List[Individual], after: List[Individual]
    ) -> Dict[str, Any]:
        """
        Analyze selection event and return statistics.

        Args:
            before: Individuals before selection
            after: Individuals after selection

        Returns:
            Dictionary with selection statistics
        """
        if not before or not after:
            return {
                "selection_intensity": 0.0,
                "survival_rate": 0.0,
                "fitness_change": 0.0,
                "variance_change": 0.0,
            }

        # Basic statistics
        before_fitness = [ind.fitness for ind in before]
        after_fitness = [ind.fitness for ind in after]

        before_mean = np.mean(before_fitness)
        after_mean = np.mean(after_fitness)
        before_var = np.var(before_fitness)
        after_var = np.var(after_fitness)

        # Selection differential and intensity
        selection_diff = after_mean - before_mean
        selection_intensity = (
            selection_diff / np.sqrt(before_var) if before_var > 0 else 0.0
        )

        return {
            "selection_intensity": selection_intensity,
            "survival_rate": len(after) / len(before),
            "fitness_change": selection_diff,
            "variance_change": after_var - before_var,
            "before_mean_fitness": before_mean,
            "after_mean_fitness": after_mean,
            "before_fitness_variance": before_var,
            "after_fitness_variance": after_var,
        }
