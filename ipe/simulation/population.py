"""
Population Dynamics Core

This module implements the Population and Individual classes for tracking
population states and managing evolutionary dynamics in IPE simulations.
"""

from typing import List, Dict, Any, Generator, Optional, Callable
from dataclasses import dataclass
import numpy as np
from ..core.physiology.state import PhysiologicalState


@dataclass
class Individual:
    """
    Represents an individual organism in the population.

    Each individual has a physiological state representing their phenotype
    and optional genetic information for tracking inheritance.
    """

    id: int
    physiological_state: PhysiologicalState
    fitness: float = 0.0
    genetic_values: Optional[Dict[str, float]] = None
    age: int = 0

    def __post_init__(self):
        """Initialize genetic values if not provided"""
        if self.genetic_values is None:
            self.genetic_values = {}


class Population:
    """
    Manages a population of individuals for evolutionary simulation.

    This class provides the core functionality for population dynamics
    including birth-death processes, fitness tracking, and generation
    management.
    """

    def __init__(
        self,
        size: int,
        initial_state: PhysiologicalState,
        carrying_capacity: Optional[int] = None,
    ):
        """
        Initialize population with given size and initial state.

        Args:
            size: Initial population size
            initial_state: Template physiological state for all individuals
            carrying_capacity: Maximum population size (defaults to 2x initial)
        """
        if size <= 0:
            raise ValueError("Population size must be positive")

        self.individuals: List[Individual] = []
        self.generation = 0
        self.carrying_capacity = carrying_capacity or size * 2
        self.size = size

        # Create initial population
        for i in range(size):
            individual = Individual(
                id=i, physiological_state=initial_state, fitness=1.0
            )
            self.individuals.append(individual)

    def current_size(self) -> int:
        """Get current population size"""
        return len(self.individuals)

    def calculate_fitness(
        self, fitness_function: Callable[[PhysiologicalState], float]
    ) -> None:
        """
        Calculate fitness for all individuals using provided function.

        Args:
            fitness_function: Function that maps PhysiologicalState to fitness
        """
        for individual in self.individuals:
            individual.fitness = fitness_function(individual.physiological_state)

    def get_mean_fitness(self) -> float:
        """Calculate mean population fitness"""
        if not self.individuals:
            return 0.0
        return np.mean([ind.fitness for ind in self.individuals])

    def get_fitness_variance(self) -> float:
        """Calculate fitness variance"""
        if not self.individuals:
            return 0.0
        fitnesses = [ind.fitness for ind in self.individuals]
        return np.var(fitnesses)

    def selection(
        self, selection_function: Callable[[List[Individual]], List[Individual]]
    ) -> None:
        """
        Apply selection to population using provided selection function.

        Args:
            selection_function: Function that selects survivors from population
        """
        self.individuals = selection_function(self.individuals)

    def reproduction(
        self, mutation_function: Optional[Callable[[Individual], Individual]] = None
    ) -> None:
        """
        Generate offspring to restore population size.

        Args:
            mutation_function: Optional function to apply mutations to offspring
        """
        current_size = len(self.individuals)
        needed_offspring = min(self.size, self.carrying_capacity) - current_size

        if needed_offspring <= 0:
            return

        # Simple reproduction: randomly sample parents
        parents = np.random.choice(
            self.individuals, size=needed_offspring, replace=True
        )

        new_id_start = max((ind.id for ind in self.individuals), default=-1) + 1

        for i, parent in enumerate(parents):
            # Create offspring with same state as parent
            offspring = Individual(
                id=new_id_start + i,
                physiological_state=parent.physiological_state,
                fitness=0.0,  # Will be calculated in next generation
                genetic_values=(
                    parent.genetic_values.copy() if parent.genetic_values else None
                ),
                age=0,
            )

            # Apply mutation if function provided
            if mutation_function:
                offspring = mutation_function(offspring)

            self.individuals.append(offspring)

    def age_population(self) -> None:
        """Increment age of all individuals"""
        for individual in self.individuals:
            individual.age += 1

    def evolve(
        self,
        generations: int,
        fitness_function: Callable[[PhysiologicalState], float],
        selection_function: Callable[[List[Individual]], List[Individual]],
        mutation_function: Optional[Callable[[Individual], Individual]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run evolution for specified number of generations.

        Args:
            generations: Number of generations to simulate
            fitness_function: Function to calculate individual fitness
            selection_function: Function to select survivors
            mutation_function: Optional mutation function

        Yields:
            Dictionary with generation statistics
        """
        for gen in range(generations):
            # Age population
            self.age_population()

            # Apply selection
            self.selection(selection_function)

            # Reproduction
            self.reproduction(mutation_function)

            # Calculate fitness for new generation
            self.calculate_fitness(fitness_function)

            # Increment generation counter
            self.generation += 1

            # Yield statistics
            yield self.get_statistics()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current population statistics.

        Returns:
            Dictionary with population metrics
        """
        if not self.individuals:
            return {
                "generation": self.generation,
                "size": 0,
                "mean_fitness": 0.0,
                "fitness_variance": 0.0,
                "mean_age": 0.0,
            }

        fitnesses = [ind.fitness for ind in self.individuals]
        ages = [ind.age for ind in self.individuals]

        return {
            "generation": self.generation,
            "size": len(self.individuals),
            "mean_fitness": np.mean(fitnesses),
            "fitness_variance": np.var(fitnesses),
            "mean_age": np.mean(ages),
            "max_fitness": max(fitnesses),
            "min_fitness": min(fitnesses),
        }
