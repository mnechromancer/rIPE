"""
Contemporary Evolution Models

This module provides specialized models for contemporary evolution
studies, focusing on rapid evolutionary responses to environmental
change typical in modern ecological contexts.
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np

from ..core.physiology.state import PhysiologicalState
from .population import Population, Individual
from .rapid_evolution import RapidEvolutionSimulator, EnvironmentalChange
from .genetic_architecture import GeneticArchitecture, create_default_architecture
from .mutation import MutationEngine, create_default_mutation_engine
from .selection import TruncationSelection


class ContemporaryEvolutionModel:
    """
    Specialized model for contemporary evolution scenarios.

    Provides pre-configured setups for common contemporary evolution
    studies including urban adaptation, climate change responses,
    and biological invasions.
    """

    @staticmethod
    def create_urban_adaptation_model(
        population_size: int = 100,
    ) -> RapidEvolutionSimulator:
        """
        Create model for urban adaptation studies.

        Args:
            population_size: Size of starting population

        Returns:
            Configured RapidEvolutionSimulator for urban adaptation
        """
        # Create urban-adapted genetic architecture
        arch = create_default_architecture()

        # Initial population in natural environment
        natural_state = PhysiologicalState(po2=20.0, temperature=15.0, altitude=100.0)

        population = Population(population_size, natural_state)

        # Mutation engine
        mutation_engine = create_default_mutation_engine(arch)

        # Selection strategy favoring survival in urban conditions
        selection_strategy = TruncationSelection(survival_fraction=0.7)

        # Urban fitness function
        def urban_fitness(
            individual: Individual, environment: PhysiologicalState
        ) -> float:
            """Fitness function for urban environments"""
            base_fitness = 0.5

            # Temperature tolerance
            temp_optimum = 18.0
            temp_tolerance = abs(environment.temperature - temp_optimum)
            temp_fitness = max(0.0, 1.0 - temp_tolerance / 10.0)

            # Pollution tolerance (represented by reduced pO2)
            pollution_tolerance = max(0.0, (environment.po2 - 15.0) / 5.0)

            return base_fitness + 0.3 * temp_fitness + 0.2 * pollution_tolerance

        return RapidEvolutionSimulator(
            population, arch, mutation_engine, selection_strategy, urban_fitness
        )

    @staticmethod
    def create_climate_change_model(
        population_size: int = 100,
    ) -> RapidEvolutionSimulator:
        """
        Create model for climate change adaptation studies.

        Args:
            population_size: Size of starting population

        Returns:
            Configured RapidEvolutionSimulator for climate adaptation
        """
        arch = create_default_architecture()

        # Initial population in historical climate
        historical_climate = PhysiologicalState(
            po2=20.0, temperature=12.0, altitude=200.0  # Historical average
        )

        population = Population(population_size, historical_climate)
        mutation_engine = create_default_mutation_engine(arch)
        selection_strategy = TruncationSelection(survival_fraction=0.6)

        def climate_fitness(
            individual: Individual, environment: PhysiologicalState
        ) -> float:
            """Fitness function for climate change scenarios"""
            base_fitness = 0.6

            # Thermal tolerance
            thermal_optimum = 12.0  # Historical optimum
            temp_deviation = abs(environment.temperature - thermal_optimum)
            thermal_fitness = max(0.0, 1.0 - temp_deviation / 15.0)

            # Metabolic efficiency under warming
            warming_stress = max(0.0, environment.temperature - 12.0)
            metabolic_fitness = max(0.1, 1.0 - warming_stress / 20.0)

            return base_fitness * thermal_fitness * metabolic_fitness

        return RapidEvolutionSimulator(
            population, arch, mutation_engine, selection_strategy, climate_fitness
        )

    @staticmethod
    def create_invasion_model(population_size: int = 50) -> RapidEvolutionSimulator:
        """
        Create model for biological invasion studies.

        Args:
            population_size: Size of founding population

        Returns:
            Configured RapidEvolutionSimulator for invasion scenarios
        """
        arch = create_default_architecture()

        # Small founding population (invasion bottleneck)
        source_environment = PhysiologicalState(
            po2=19.0, temperature=20.0, altitude=0.0, salinity=35.0  # Marine source
        )

        population = Population(population_size, source_environment)

        # Higher mutation rate for rapid adaptation
        mutation_engine = create_default_mutation_engine(arch)
        mutation_engine.parameters.base_mutation_rate *= 2  # Double mutation rate

        # Strong selection in novel environment
        selection_strategy = TruncationSelection(survival_fraction=0.5)

        def invasion_fitness(
            individual: Individual, environment: PhysiologicalState
        ) -> float:
            """Fitness function for invasion scenarios"""
            base_fitness = 0.4  # Lower base fitness in novel environment

            # Salinity tolerance
            if environment.salinity is not None:
                salinity_optimum = 35.0  # Marine optimum
                salinity_deviation = abs(environment.salinity - salinity_optimum)
                salinity_fitness = max(0.0, 1.0 - salinity_deviation / 40.0)
            else:
                salinity_fitness = 0.5  # Moderate fitness in freshwater

            # Osmoregulatory efficiency
            osmoreg_efficiency = 0.7 + 0.3 * np.random.random()  # Variable efficiency

            return base_fitness + 0.4 * salinity_fitness + 0.2 * osmoreg_efficiency

        return RapidEvolutionSimulator(
            population, arch, mutation_engine, selection_strategy, invasion_fitness
        )


class ExperimentalEvolutionModel:
    """
    Models for experimental evolution studies.

    Provides setups for common experimental evolution protocols
    including selection experiments and common garden studies.
    """

    @staticmethod
    def create_selection_experiment(
        trait_target: float, population_size: int = 200
    ) -> RapidEvolutionSimulator:
        """
        Create artificial selection experiment model.

        Args:
            trait_target: Target value for selected trait
            population_size: Population size to maintain

        Returns:
            Configured RapidEvolutionSimulator for selection experiment
        """
        arch = create_default_architecture()

        # Standard laboratory conditions
        lab_environment = PhysiologicalState(po2=20.0, temperature=22.0, altitude=100.0)

        population = Population(population_size, lab_environment)
        mutation_engine = create_default_mutation_engine(arch)

        # Directional selection toward target
        selection_strategy = TruncationSelection(survival_fraction=0.8)

        def selection_fitness(
            individual: Individual, environment: PhysiologicalState
        ) -> float:
            """Fitness function for directional selection"""
            # Extract trait value (using heart mass as example)
            trait_value = individual.genetic_values.get("cardio1", 0.0)

            # Fitness decreases with distance from target
            distance_from_target = abs(trait_value - trait_target)
            return max(0.1, 1.0 - distance_from_target)

        return RapidEvolutionSimulator(
            population, arch, mutation_engine, selection_strategy, selection_fitness
        )

    @staticmethod
    def create_common_garden_experiment(
        environments: List[PhysiologicalState], population_size: int = 100
    ) -> Dict[str, RapidEvolutionSimulator]:
        """
        Create common garden experiment with multiple environments.

        Args:
            environments: List of environments to test
            population_size: Population size for each environment

        Returns:
            Dictionary mapping environment names to simulators
        """
        simulators = {}
        arch = create_default_architecture()

        for i, environment in enumerate(environments):
            # Create population for this environment
            population = Population(population_size, environment)
            mutation_engine = create_default_mutation_engine(arch)
            selection_strategy = TruncationSelection(survival_fraction=0.7)

            def env_fitness(individual: Individual, env: PhysiologicalState) -> float:
                """Environment-specific fitness function"""
                base_fitness = 0.5

                # General physiological efficiency
                efficiency = 0.5 + 0.5 * np.random.random()

                # Environment-specific adaptations
                temp_fitness = max(0.0, 1.0 - abs(env.temperature - 20.0) / 20.0)
                oxygen_fitness = max(0.0, env.po2 / 21.0)

                return base_fitness * efficiency * temp_fitness * oxygen_fitness

            simulator = RapidEvolutionSimulator(
                population, arch, mutation_engine, selection_strategy, env_fitness
            )

            simulators[f"environment_{i}"] = simulator

        return simulators


def create_predefined_scenarios() -> Dict[str, EnvironmentalChange]:
    """
    Create library of predefined environmental change scenarios.

    Returns:
        Dictionary mapping scenario names to EnvironmentalChange objects
    """
    scenarios = {}

    # Urban heat island
    scenarios["urban_heat_island"] = EnvironmentalChange(
        name="urban_heat_island",
        description="Gradual warming in urban environment",
        start_generation=0,
        duration=30,
        initial_environment=PhysiologicalState(
            po2=20.0, temperature=15.0, altitude=100.0
        ),
        final_environment=PhysiologicalState(
            po2=18.0, temperature=25.0, altitude=100.0
        ),
        change_type="gradual",
    )

    # Pollution gradient
    scenarios["pollution_gradient"] = EnvironmentalChange(
        name="pollution_gradient",
        description="Increasing pollution (decreasing oxygen)",
        start_generation=0,
        duration=25,
        initial_environment=PhysiologicalState(
            po2=21.0, temperature=20.0, altitude=100.0
        ),
        final_environment=PhysiologicalState(
            po2=16.0, temperature=20.0, altitude=100.0
        ),
        change_type="gradual",
    )

    # Seasonal temperature variation
    scenarios["seasonal_variation"] = EnvironmentalChange(
        name="seasonal_variation",
        description="Oscillating seasonal temperatures",
        start_generation=0,
        duration=100,
        initial_environment=PhysiologicalState(
            po2=20.0, temperature=10.0, altitude=200.0
        ),
        final_environment=PhysiologicalState(
            po2=20.0, temperature=30.0, altitude=200.0
        ),
        change_type="oscillating",
        oscillation_period=10,  # 10-generation "seasons"
    )

    # Sudden habitat fragmentation
    scenarios["habitat_fragmentation"] = EnvironmentalChange(
        name="habitat_fragmentation",
        description="Sudden environmental degradation",
        start_generation=20,
        duration=5,
        initial_environment=PhysiologicalState(
            po2=20.0, temperature=18.0, altitude=150.0
        ),
        final_environment=PhysiologicalState(
            po2=17.0, temperature=22.0, altitude=150.0
        ),
        change_type="sudden",
    )

    return scenarios
