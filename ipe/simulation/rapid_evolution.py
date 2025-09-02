"""
Rapid Evolution Simulator

This module implements rapid evolution simulations for 10-100 generation
timescales, integrating population dynamics, selection, mutation, and
plasticity evolution for contemporary evolutionary studies.
"""

from typing import List, Dict, Any, Optional, Generator, Callable
from dataclasses import dataclass, field
import numpy as np
from ..core.physiology.state import PhysiologicalState
from ..core.plasticity.reaction_norm import ReactionNorm
from ..core.plasticity.maladaptive import MaladaptiveDetector

from .population import Population, Individual
from .selection import SelectionStrategy
from .mutation import MutationEngine
from .genetic_architecture import GeneticArchitecture


@dataclass
class EnvironmentalChange:
    """
    Represents an environmental change scenario.

    Defines how environmental conditions change over
    the course of a rapid evolution simulation.
    """

    name: str
    description: str
    start_generation: int = 0
    duration: int = 10  # generations

    # Environmental gradient parameters
    initial_environment: PhysiologicalState = None
    final_environment: PhysiologicalState = None
    change_type: str = "gradual"  # "gradual", "sudden", "oscillating"

    # Change pattern parameters
    oscillation_period: int = 5  # for oscillating changes
    change_magnitude: float = 1.0  # scaling factor

    def get_environment_at_generation(self, generation: int) -> PhysiologicalState:
        """
        Get environmental state at specific generation.

        Args:
            generation: Generation number

        Returns:
            PhysiologicalState for that generation
        """
        if generation < self.start_generation:
            return self.initial_environment

        if generation >= self.start_generation + self.duration:
            return self.final_environment

        # Calculate progress through change
        progress = (generation - self.start_generation) / self.duration

        if self.change_type == "sudden":
            return (
                self.final_environment if progress > 0.5 else self.initial_environment
            )

        elif self.change_type == "oscillating":
            oscillation_phase = (
                2
                * np.pi
                * (generation - self.start_generation)
                / self.oscillation_period
            )
            oscillation_factor = (1 + np.sin(oscillation_phase)) / 2
            return self._interpolate_environments(oscillation_factor)

        else:  # gradual
            return self._interpolate_environments(progress)

    def _interpolate_environments(self, progress: float) -> PhysiologicalState:
        """Interpolate between initial and final environments"""
        if self.initial_environment is None or self.final_environment is None:
            return self.initial_environment or self.final_environment

        # Linear interpolation of environmental parameters
        initial_dict = self.initial_environment.__dict__
        final_dict = self.final_environment.__dict__

        interpolated = {}
        for key in initial_dict:
            if (
                key in final_dict
                and initial_dict[key] is not None
                and final_dict[key] is not None
            ):
                initial_val = initial_dict[key]
                final_val = final_dict[key]

                # Only interpolate numeric values, copy others as-is
                if isinstance(initial_val, (int, float)) and isinstance(
                    final_val, (int, float)
                ):
                    interpolated[key] = initial_val + progress * (
                        final_val - initial_val
                    )
                else:
                    # For non-numeric values (dicts, etc.), use initial value for 
                    # first half, final for second half
                    interpolated[key] = initial_val if progress < 0.5 else final_val
            else:
                interpolated[key] = initial_dict[key]

        return PhysiologicalState(**interpolated)


@dataclass
class PlasticityEvolutionTracker:
    """
    Tracks evolution of phenotypic plasticity over generations.

    Monitors changes in reaction norms, maladaptive responses,
    and plastic vs. constitutive strategies.
    """

    generation_data: List[Dict[str, Any]] = field(default_factory=list)

    def record_generation(
        self,
        generation: int,
        population: List[Individual],
        environment: PhysiologicalState,
        reaction_norms: Optional[Dict[int, ReactionNorm]] = None,
    ):
        """
        Record plasticity data for a generation.

        Args:
            generation: Current generation
            population: Population individuals
            environment: Current environment
            reaction_norms: Optional reaction norms for individuals
        """
        # Calculate plasticity metrics
        plasticity_data = {
            "generation": generation,
            "population_size": len(population),
            "environmental_conditions": {
                "po2": environment.po2,
                "temperature": environment.temperature,
                "altitude": environment.altitude,
            },
        }

        if reaction_norms:
            # Analyze plasticity evolution
            plasticity_magnitudes = []
            for individual in population:
                if individual.id in reaction_norms:
                    norm = reaction_norms[individual.id]
                    magnitude = norm.plasticity_magnitude()
                    plasticity_magnitudes.append(magnitude)

            if plasticity_magnitudes:
                plasticity_data.update(
                    {
                        "mean_plasticity": np.mean(plasticity_magnitudes),
                        "plasticity_variance": np.var(plasticity_magnitudes),
                        "max_plasticity": np.max(plasticity_magnitudes),
                        "min_plasticity": np.min(plasticity_magnitudes),
                    }
                )

        # Analyze maladaptive responses
        if len(population) > 0:
            try:
                detector = MaladaptiveDetector([environment])
                maladaptive_count = 0

                for individual in population:
                    if reaction_norms and individual.id in reaction_norms:
                        norm = reaction_norms[individual.id]
                        responses = detector.detect_maladaptive_responses([norm])
                        if len(responses) > 0:
                            maladaptive_count += 1

                plasticity_data["maladaptive_fraction"] = maladaptive_count / len(
                    population
                )
            except Exception:
                plasticity_data["maladaptive_fraction"] = 0.0

        self.generation_data.append(plasticity_data)

    def get_plasticity_trajectory(self) -> Dict[str, List[float]]:
        """
        Get trajectory of plasticity evolution over generations.

        Returns:
            Dictionary with time series of plasticity metrics
        """
        if not self.generation_data:
            return {}

        trajectory = {
            "generations": [],
            "mean_plasticity": [],
            "plasticity_variance": [],
            "maladaptive_fraction": [],
        }

        for data in self.generation_data:
            trajectory["generations"].append(data["generation"])
            trajectory["mean_plasticity"].append(data.get("mean_plasticity", 0.0))
            trajectory["plasticity_variance"].append(
                data.get("plasticity_variance", 0.0)
            )
            trajectory["maladaptive_fraction"].append(
                data.get("maladaptive_fraction", 0.0)
            )

        return trajectory


class RapidEvolutionSimulator:
    """
    Rapid evolution simulator for contemporary evolutionary dynamics.

    Integrates population dynamics, selection, mutation, and plasticity
    evolution for 10-100 generation timescales typical of contemporary
    evolution studies.
    """

    def __init__(
        self,
        initial_population: Population,
        genetic_architecture: GeneticArchitecture,
        mutation_engine: MutationEngine,
        selection_strategy: SelectionStrategy,
        fitness_function: Callable[[Individual, PhysiologicalState], float],
    ):
        """
        Initialize rapid evolution simulator.

        Args:
            initial_population: Starting population
            genetic_architecture: Genetic architecture for trait mapping
            mutation_engine: Mutation engine for genetic variation
            selection_strategy: Selection strategy to apply
            fitness_function: Function to calculate individual fitness
        """
        self.population = initial_population
        self.genetic_architecture = genetic_architecture
        self.mutation_engine = mutation_engine
        self.selection_strategy = selection_strategy
        self.fitness_function = fitness_function

        # Evolution tracking
        self.plasticity_tracker = PlasticityEvolutionTracker()
        self.generation_statistics: List[Dict[str, Any]] = []

        # Visualization hooks
        self.generation_callbacks: List[Callable[[int, Population], None]] = []

    def add_visualization_callback(self, callback: Callable[[int, Population], None]):
        """Add callback for real-time visualization"""
        self.generation_callbacks.append(callback)

    def simulate_environmental_change(
        self, environmental_scenario: EnvironmentalChange, total_generations: int
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Simulate evolution under environmental change.

        Args:
            environmental_scenario: Environmental change scenario
            total_generations: Total number of generations to simulate

        Yields:
            Generation statistics and evolution data
        """
        for generation in range(total_generations):
            # Get current environment
            current_env = environmental_scenario.get_environment_at_generation(
                generation
            )

            # Run single generation
            gen_data = self._run_generation(generation, current_env)

            # Track plasticity evolution
            self.plasticity_tracker.record_generation(
                generation, self.population.individuals, current_env
            )

            # Call visualization callbacks
            for callback in self.generation_callbacks:
                callback(generation, self.population)

            yield gen_data

    def simulate_constant_environment(
        self, environment: PhysiologicalState, generations: int
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Simulate evolution in constant environment.

        Args:
            environment: Constant environmental conditions
            generations: Number of generations to simulate

        Yields:
            Generation statistics
        """
        for generation in range(generations):
            gen_data = self._run_generation(generation, environment)

            # Track plasticity
            self.plasticity_tracker.record_generation(
                generation, self.population.individuals, environment
            )

            # Callbacks
            for callback in self.generation_callbacks:
                callback(generation, self.population)

            yield gen_data

    def _run_generation(
        self, generation: int, environment: PhysiologicalState
    ) -> Dict[str, Any]:
        """
        Run a single generation of evolution.

        Args:
            generation: Current generation number
            environment: Current environmental conditions

        Returns:
            Statistics for this generation
        """
        # Age population
        self.population.age_population()

        # Calculate fitness
        for individual in self.population.individuals:
            individual.fitness = self.fitness_function(individual, environment)

        # Apply selection
        before_selection = self.population.individuals.copy()
        survivors = self.selection_strategy.select(self.population.individuals)
        self.population.individuals = survivors

        # Reproduce with mutation
        offspring = []
        target_size = self.population.size
        current_size = len(self.population.individuals)

        for _ in range(target_size - current_size):
            # Select parent
            parent = np.random.choice(self.population.individuals)

            # Create offspring
            child = Individual(
                id=max(ind.id for ind in self.population.individuals)
                + len(offspring)
                + 1,
                physiological_state=parent.physiological_state,
                fitness=0.0,
                genetic_values=(
                    parent.genetic_values.copy() if parent.genetic_values else {}
                ),
                age=0,
            )

            # Apply mutation
            child = self.mutation_engine.mutate_individual(child)
            offspring.append(child)

        self.population.individuals.extend(offspring)

        # Evolution of mutation rates
        self.mutation_engine.evolve_mutation_rates(self.population.individuals)

        # Maintain standing variation
        self.population.individuals = self.mutation_engine.maintain_standing_variation(
            self.population.individuals
        )

        # Update generation counter
        self.population.generation = generation

        # Collect statistics
        gen_stats = self._collect_generation_statistics(
            generation, environment, before_selection, self.population.individuals
        )

        self.generation_statistics.append(gen_stats)
        return gen_stats

    def _collect_generation_statistics(
        self,
        generation: int,
        environment: PhysiologicalState,
        before_selection: List[Individual],
        after_selection: List[Individual],
    ) -> Dict[str, Any]:
        """Collect comprehensive statistics for generation"""

        # Basic population stats
        fitness_values = [ind.fitness for ind in after_selection]

        stats = {
            "generation": generation,
            "population_size": len(after_selection),
            "mean_fitness": np.mean(fitness_values),
            "fitness_variance": np.var(fitness_values),
            "max_fitness": np.max(fitness_values),
            "min_fitness": np.min(fitness_values),
            "environment": {
                "po2": environment.po2,
                "temperature": environment.temperature,
                "altitude": environment.altitude,
            },
        }

        # Selection statistics
        if before_selection:
            before_fitness = [ind.fitness for ind in before_selection]
            selection_differential = np.mean(fitness_values) - np.mean(before_fitness)
            stats.update(
                {
                    "selection_differential": selection_differential,
                    "survival_rate": len(after_selection) / len(before_selection),
                }
            )

        # Genetic statistics
        if after_selection and after_selection[0].genetic_values:
            locus_means = {}
            locus_vars = {}

            for locus_name in self.genetic_architecture.loci.keys():
                values = [
                    ind.genetic_values.get(locus_name, 0.0) for ind in after_selection
                ]
                locus_means[locus_name] = np.mean(values)
                locus_vars[locus_name] = np.var(values)

            stats.update(
                {"genetic_means": locus_means, "genetic_variances": locus_vars}
            )

        # Mutation statistics
        mutation_stats = self.mutation_engine.get_mutation_statistics()
        stats["mutation_rates"] = mutation_stats["current_mutation_rates"]

        return stats

    def get_evolution_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of evolution simulation.

        Returns:
            Summary statistics and trajectories
        """
        if not self.generation_statistics:
            return {}

        # Extract trajectories
        generations = [stat["generation"] for stat in self.generation_statistics]
        mean_fitness = [stat["mean_fitness"] for stat in self.generation_statistics]
        fitness_variance = [
            stat["fitness_variance"] for stat in self.generation_statistics
        ]
        population_sizes = [
            stat["population_size"] for stat in self.generation_statistics
        ]

        # Calculate fitness change
        initial_fitness = mean_fitness[0] if mean_fitness else 0
        final_fitness = mean_fitness[-1] if mean_fitness else 0
        fitness_change = final_fitness - initial_fitness

        # Plasticity trajectory
        plasticity_trajectory = self.plasticity_tracker.get_plasticity_trajectory()

        return {
            "total_generations": len(self.generation_statistics),
            "fitness_trajectory": {
                "generations": generations,
                "mean_fitness": mean_fitness,
                "fitness_variance": fitness_variance,
            },
            "population_trajectory": {
                "generations": generations,
                "population_sizes": population_sizes,
            },
            "plasticity_evolution": plasticity_trajectory,
            "summary_statistics": {
                "initial_fitness": initial_fitness,
                "final_fitness": final_fitness,
                "total_fitness_change": fitness_change,
                "final_population_size": (
                    population_sizes[-1] if population_sizes else 0
                ),
            },
        }


def create_freshwater_invasion_scenario() -> EnvironmentalChange:
    """
    Create an environmental change scenario for freshwater invasion.

    Models the environmental transition from marine to freshwater
    conditions typical of alewife invasion studies.

    Returns:
        EnvironmentalChange scenario for freshwater invasion
    """
    marine_environment = PhysiologicalState(
        po2=18.0, temperature=15.0, altitude=0.0, salinity=35.0
    )

    freshwater_environment = PhysiologicalState(
        po2=18.0, temperature=15.0, altitude=0.0, salinity=0.0
    )

    return EnvironmentalChange(
        name="freshwater_invasion",
        description="Marine to freshwater transition",
        start_generation=0,
        duration=20,
        initial_environment=marine_environment,
        final_environment=freshwater_environment,
        change_type="gradual",
    )


def create_altitude_adaptation_scenario() -> EnvironmentalChange:
    """
    Create environmental change scenario for altitude adaptation.

    Models hypoxic stress from low to high altitude conditions.

    Returns:
        EnvironmentalChange scenario for altitude adaptation
    """
    lowland_environment = PhysiologicalState(po2=21.0, temperature=20.0, altitude=100.0)

    highland_environment = PhysiologicalState(
        po2=14.0,  # Reduced oxygen at altitude
        temperature=10.0,  # Cooler at altitude
        altitude=3000.0,
    )

    return EnvironmentalChange(
        name="altitude_adaptation",
        description="Lowland to highland transition",
        start_generation=0,
        duration=50,
        initial_environment=lowland_environment,
        final_environment=highland_environment,
        change_type="gradual",
    )
