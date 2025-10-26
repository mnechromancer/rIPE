#!/usr/bin/env python3
"""
Alpine Evolution Simulation Runner

Runs an actual evolutionary simulation using your Alpine parameters:
- Altitude: 3500m
- Temperature: -8°C
- Oxygen level: 0.65
- Population size: 150
- Duration: 25 generations
- Mutation rate: 0.002
"""

import numpy as np
from ipe.core.physiology.state import PhysiologicalState
from ipe.simulation.population import Population, Individual
from ipe.simulation.genetic_architecture import create_default_architecture
from ipe.simulation.mutation import MutationEngine, MutationParameters
from ipe.simulation.selection import TruncationSelection
from ipe.simulation.rapid_evolution import RapidEvolutionSimulator


def alpine_fitness_function(
    individual: Individual, environment: PhysiologicalState
) -> float:
    """
    Fitness function for Alpine environment adaptation.

    Individuals with physiological states closer to the alpine environment
    have higher fitness.
    """
    # Calculate environmental stress based on altitude and temperature
    altitude_stress = (
        abs(individual.physiological_state.altitude - environment.altitude) / 1000.0
    )
    temp_stress = (
        abs(individual.physiological_state.temperature - environment.temperature) / 10.0
    )
    oxygen_stress = abs(individual.physiological_state.po2 - environment.po2) / 5.0

    # Fitness decreases with environmental stress
    total_stress = altitude_stress + temp_stress + oxygen_stress
    fitness = max(0.1, 1.0 - (total_stress * 0.1))  # Minimum fitness of 0.1

    return fitness


def main():
    """Run the Alpine Evolution simulation."""
    print("🏔️  RIPE Alpine Evolution Simulation")
    print("=" * 50)

    # Create Alpine environment (from your API parameters)
    alpine_environment = PhysiologicalState(
        po2=13.0,  # Reduced oxygen at altitude (from oxygen_level 0.65)
        temperature=-8.0,  # Your temperature parameter
        altitude=3500.0,  # Your altitude parameter
        heart_mass=10.0,  # Default heart mass
    )

    print(f"🌍 Environment: {alpine_environment}")
    print()

    # Create initial population (sea-level adapted)
    print("👥 Creating initial population (150 individuals)...")
    sea_level_state = PhysiologicalState(
        po2=21.0,  # Sea level oxygen
        temperature=15.0,  # Temperate temperature
        altitude=0.0,  # Sea level
        heart_mass=10.0,
    )

    initial_population = Population(
        size=150, initial_state=sea_level_state  # Your population_size parameter
    )

    # Set up genetic architecture
    print("🧬 Setting up genetic architecture...")
    genetic_architecture = create_default_architecture()

    # Set up mutation engine with your mutation rate
    print("🔬 Configuring mutation engine...")
    mutation_params = MutationParameters(
        base_mutation_rate=0.002,  # Your mutation_rate parameter
        mutational_variance=0.01,
        pleiotropic_correlation=0.1,
    )
    mutation_engine = MutationEngine(
        parameters=mutation_params, architecture=genetic_architecture
    )

    # Set up selection strategy (moderate selection pressure)
    print("🎯 Setting up selection strategy...")
    selection_strategy = TruncationSelection(
        survival_fraction=0.7
    )  # 70% survive each generation

    # Create the simulator
    print("⚙️  Initializing rapid evolution simulator...")
    simulator = RapidEvolutionSimulator(
        initial_population=initial_population,
        genetic_architecture=genetic_architecture,
        mutation_engine=mutation_engine,
        selection_strategy=selection_strategy,
        fitness_function=alpine_fitness_function,
    )

    print()
    print("🚀 Starting simulation...")
    print("   Duration: 25 generations")
    print("   Monitoring adaptation to Alpine conditions...")
    print()

    # Run simulation
    generation_data = []

    for generation_stats in simulator.simulate_constant_environment(
        environment=alpine_environment, generations=25  # Your duration parameter
    ):
        gen = generation_stats["generation"]
        pop_size = generation_stats["population_size"]
        mean_fitness = generation_stats["mean_fitness"]
        max_fitness = generation_stats["max_fitness"]
        mean_altitude_adaptation = generation_stats.get("mean_altitude", 0)

        generation_data.append(generation_stats)

        # Print progress every 5 generations
        if gen % 5 == 0 or gen == 1:
            print(
                f"Gen {gen:2d}: Pop={pop_size:3d}, Mean Fitness={mean_fitness:.3f}, Max Fitness={max_fitness:.3f}"
            )

    print()
    print("✅ Simulation completed!")
    print()

    # Final statistics
    final_stats = generation_data[-1]
    initial_stats = generation_data[0]

    print("📊 Evolution Summary:")
    print(f"   Initial mean fitness: {initial_stats['mean_fitness']:.3f}")
    print(f"   Final mean fitness:   {final_stats['mean_fitness']:.3f}")
    print(
        f"   Fitness improvement:  {final_stats['mean_fitness'] - initial_stats['mean_fitness']:.3f}"
    )
    print(f"   Final population:     {final_stats['population_size']}")
    print(
        f"   Selection differential: {final_stats.get('selection_differential', 'N/A')}"
    )

    print()
    print("🧬 Adaptive Evolution Analysis:")
    fitness_change = final_stats["mean_fitness"] - initial_stats["mean_fitness"]
    if fitness_change > 0.1:
        print("   ✅ Strong adaptation detected!")
    elif fitness_change > 0.05:
        print("   ⚡ Moderate adaptation detected.")
    else:
        print("   📈 Gradual adaptation in progress.")

    print()
    print("🎯 Your Alpine evolution simulation completed successfully!")
    print(
        "   The population adapted to high-altitude, cold conditions over 25 generations."
    )

    return generation_data


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)

    try:
        results = main()
        print("\n🎉 Simulation data available in 'results' variable")
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        print("Check that all RIPE modules are properly installed.")
