"""
Tests for Population Dynamics Core (EVOL-001)

This module tests the Population and Individual classes,
including population management, evolution simulation,
and demographic tracking.
"""

import pytest
import numpy as np
from typing import List

from ipe.core.physiology.state import PhysiologicalState
from ipe.simulation.population import Population, Individual
from ipe.simulation.demographics import Demographics, AgeClass


class TestIndividual:
    """Test Individual class functionality"""
    
    def test_basic_creation(self):
        """Test creating an individual with basic parameters"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        individual = Individual(id=1, physiological_state=state, fitness=0.8)
        
        assert individual.id == 1
        assert individual.physiological_state == state
        assert individual.fitness == 0.8
        assert individual.age == 0
        assert individual.genetic_values == {}
    
    def test_with_genetic_values(self):
        """Test individual with genetic values"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        genetic_values = {"gene1": 0.5, "gene2": -0.2}
        individual = Individual(
            id=1, 
            physiological_state=state,
            fitness=0.8,
            genetic_values=genetic_values
        )
        
        assert individual.genetic_values == genetic_values


class TestPopulation:
    """Test Population class functionality"""
    
    @pytest.fixture
    def basic_state(self):
        """Fixture for basic physiological state"""
        return PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
    
    @pytest.fixture
    def basic_population(self, basic_state):
        """Fixture for basic population"""
        return Population(size=10, initial_state=basic_state)
    
    def test_population_creation(self, basic_state):
        """Test basic population creation"""
        pop = Population(size=10, initial_state=basic_state)
        
        assert pop.current_size() == 10
        assert pop.generation == 0
        assert pop.carrying_capacity == 20  # Default 2x initial
        assert len(pop.individuals) == 10
        
        # Check all individuals have correct initial state
        for i, individual in enumerate(pop.individuals):
            assert individual.id == i
            assert individual.physiological_state == basic_state
            assert individual.fitness == 1.0
            assert individual.age == 0
    
    def test_population_creation_errors(self, basic_state):
        """Test population creation error conditions"""
        with pytest.raises(ValueError, match="Population size must be positive"):
            Population(size=0, initial_state=basic_state)
        
        with pytest.raises(ValueError, match="Population size must be positive"):
            Population(size=-5, initial_state=basic_state)
    
    def test_custom_carrying_capacity(self, basic_state):
        """Test population with custom carrying capacity"""
        pop = Population(size=10, initial_state=basic_state, carrying_capacity=50)
        assert pop.carrying_capacity == 50
    
    def test_fitness_calculation(self, basic_population):
        """Test fitness calculation for population"""
        def simple_fitness(state: PhysiologicalState) -> float:
            return state.po2 / 20.0  # Simple function of oxygen
        
        basic_population.calculate_fitness(simple_fitness)
        
        expected_fitness = 15.0 / 20.0  # 0.75
        for individual in basic_population.individuals:
            assert individual.fitness == expected_fitness
    
    def test_fitness_statistics(self, basic_population):
        """Test population fitness statistics"""
        # Set varying fitness values
        fitness_values = [0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.5, 0.7, 0.9, 0.1]
        for i, fitness in enumerate(fitness_values):
            basic_population.individuals[i].fitness = fitness
        
        assert abs(basic_population.get_mean_fitness() - 0.55) < 1e-10
        assert basic_population.get_fitness_variance() == pytest.approx(0.0825, rel=1e-3)
    
    def test_empty_population_statistics(self):
        """Test statistics for empty population"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        pop = Population(size=1, initial_state=state)
        pop.individuals = []  # Clear population
        
        assert pop.get_mean_fitness() == 0.0
        assert pop.get_fitness_variance() == 0.0
    
    def test_age_population(self, basic_population):
        """Test aging population"""
        # Set initial ages
        for i, individual in enumerate(basic_population.individuals):
            individual.age = i
        
        basic_population.age_population()
        
        for i, individual in enumerate(basic_population.individuals):
            assert individual.age == i + 1
    
    def test_simple_selection(self, basic_population):
        """Test selection mechanism"""
        # Set fitness values - select top 5
        fitness_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i, fitness in enumerate(fitness_values):
            basic_population.individuals[i].fitness = fitness
        
        def top_half_selection(individuals: List[Individual]) -> List[Individual]:
            sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
            return sorted_individuals[:len(individuals)//2]
        
        basic_population.selection(top_half_selection)
        
        assert basic_population.current_size() == 5
        # Check that survivors have highest fitness
        for individual in basic_population.individuals:
            assert individual.fitness >= 0.6
    
    def test_reproduction(self, basic_population):
        """Test reproduction to restore population size"""
        # Reduce population size
        basic_population.individuals = basic_population.individuals[:5]
        assert basic_population.current_size() == 5
        
        # Reproduce to restore size
        basic_population.reproduction()
        
        assert basic_population.current_size() == 10
        
        # Check new individuals have sequential IDs starting from max existing + 1
        new_ids = [ind.id for ind in basic_population.individuals[5:]]
        assert new_ids == list(range(5, 10))
    
    def test_reproduction_with_mutation(self, basic_population):
        """Test reproduction with mutation function"""
        # Reduce population size
        basic_population.individuals = basic_population.individuals[:5]
        
        def simple_mutation(individual: Individual) -> Individual:
            # Simple mutation: add random noise to fitness
            individual.fitness = max(0.0, individual.fitness + np.random.normal(0, 0.1))
            return individual
        
        # Set random seed for reproducible test
        np.random.seed(42)
        basic_population.reproduction(mutation_function=simple_mutation)
        
        assert basic_population.current_size() == 10
    
    def test_carrying_capacity_limit(self, basic_population):
        """Test that reproduction respects carrying capacity"""
        # Set carrying capacity to current size
        basic_population.carrying_capacity = 10
        
        # Try to reproduce beyond capacity
        basic_population.reproduction()
        
        # Should not exceed carrying capacity
        assert basic_population.current_size() <= 10
    
    def test_evolution_single_generation(self, basic_population):
        """Test single generation evolution"""
        def fitness_func(state: PhysiologicalState) -> float:
            return 0.8
        
        def selection_func(individuals: List[Individual]) -> List[Individual]:
            return individuals[:5]  # Keep first half
        
        results = list(basic_population.evolve(
            generations=1,
            fitness_function=fitness_func,
            selection_function=selection_func
        ))
        
        assert len(results) == 1
        stats = results[0]
        
        assert stats['generation'] == 1
        assert stats['size'] == 10  # Restored by reproduction
        assert stats['mean_fitness'] == 0.8
        assert basic_population.generation == 1
    
    def test_evolution_multiple_generations(self, basic_population):
        """Test multi-generation evolution"""
        def fitness_func(state: PhysiologicalState) -> float:
            return np.random.random()  # Random fitness
        
        def selection_func(individuals: List[Individual]) -> List[Individual]:
            return individuals[:8]  # Keep 80%
        
        np.random.seed(42)  # For reproducible results
        results = list(basic_population.evolve(
            generations=3,
            fitness_function=fitness_func,
            selection_function=selection_func
        ))
        
        assert len(results) == 3
        assert basic_population.generation == 3
        
        # Check generation progression
        for i, stats in enumerate(results):
            assert stats['generation'] == i + 1
            assert stats['size'] == 10
    
    def test_get_statistics(self, basic_population):
        """Test population statistics generation"""
        # Set up known fitness and age values
        fitness_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ages = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        
        for i, (fitness, age) in enumerate(zip(fitness_values, ages)):
            basic_population.individuals[i].fitness = fitness
            basic_population.individuals[i].age = age
        
        basic_population.generation = 5
        
        stats = basic_population.get_statistics()
        
        assert stats['generation'] == 5
        assert stats['size'] == 10
        assert stats['mean_fitness'] == 0.55
        assert stats['mean_age'] == 3.0
        assert stats['max_fitness'] == 1.0
        assert stats['min_fitness'] == 0.1


class TestDemographics:
    """Test Demographics analysis functionality"""
    
    @pytest.fixture
    def sample_population(self):
        """Create population with varied ages for testing"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        pop = Population(size=10, initial_state=state)
        
        # Set up age structure: [0,0,1,1,1,2,2,3,4,5]
        ages = [0, 0, 1, 1, 1, 2, 2, 3, 4, 5]
        fitness_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for i, (age, fitness) in enumerate(zip(ages, fitness_values)):
            pop.individuals[i].age = age
            pop.individuals[i].fitness = fitness
        
        return pop
    
    def test_age_structure(self, sample_population):
        """Test age structure calculation"""
        demographics = Demographics(sample_population)
        age_structure = demographics.get_age_structure()
        
        # Expected: age 0 (2 individuals), age 1 (3), age 2 (2), age 3 (1), age 4 (1), age 5 (1)
        assert len(age_structure) == 6
        
        age_counts = {ac.age: ac.count for ac in age_structure}
        assert age_counts[0] == 2
        assert age_counts[1] == 3
        assert age_counts[2] == 2
        assert age_counts[3] == 1
        assert age_counts[4] == 1
        assert age_counts[5] == 1
        
        # Check mean fitness for age 1 group (0.7, 0.8, 0.9)
        age_1_class = next(ac for ac in age_structure if ac.age == 1)
        assert abs(age_1_class.mean_fitness - 0.8) < 1e-10
    
    def test_empty_population_demographics(self):
        """Test demographics with empty population"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        pop = Population(size=1, initial_state=state)
        pop.individuals = []
        
        demographics = Demographics(pop)
        age_structure = demographics.get_age_structure()
        
        assert age_structure == []
    
    def test_life_expectancy(self, sample_population):
        """Test life expectancy calculation"""
        demographics = Demographics(sample_population)
        life_exp = demographics.calculate_life_expectancy()
        
        # Weighted average: (0*2 + 1*3 + 2*2 + 3*1 + 4*1 + 5*1) / 10 = 19/10 = 1.9
        assert life_exp == 1.9
    
    def test_population_growth_rate(self, sample_population):
        """Test population growth rate calculation"""
        sample_population.generation = 5
        # Current size = 10, initial size = 10, so growth rate = ln(10/10)/5 = 0
        
        demographics = Demographics(sample_population)
        growth_rate = demographics.get_population_growth_rate()
        
        assert growth_rate == 0.0
    
    def test_population_growth_rate_with_growth(self, sample_population):
        """Test growth rate with actual population growth"""
        sample_population.generation = 2
        # Add more individuals to simulate growth
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        for i in range(5):  # Add 5 more individuals
            individual = Individual(id=10+i, physiological_state=state)
            sample_population.individuals.append(individual)
        
        demographics = Demographics(sample_population)
        growth_rate = demographics.get_population_growth_rate()
        
        # ln(15/10) / 2 = ln(1.5) / 2 ≈ 0.2027
        expected_rate = np.log(15/10) / 2
        assert abs(growth_rate - expected_rate) < 1e-10
    
    def test_demographic_summary(self, sample_population):
        """Test comprehensive demographic summary"""
        sample_population.generation = 3
        demographics = Demographics(sample_population)
        summary = demographics.get_demographic_summary()
        
        assert summary['total_population'] == 10
        assert summary['age_classes'] == 6
        assert summary['max_age'] == 5
        assert summary['life_expectancy'] == 1.9
        assert summary['juvenile_fraction'] == 0.2  # 2 out of 10 are age 0
        assert summary['adult_fraction'] == 0.8     # 8 out of 10 are age > 0
    
    def test_demographic_summary_empty_population(self):
        """Test demographic summary with empty population"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        pop = Population(size=1, initial_state=state)
        pop.individuals = []
        
        demographics = Demographics(pop)
        summary = demographics.get_demographic_summary()
        
        assert summary['total_population'] == 0
        assert summary['age_classes'] == 0
        assert summary['life_expectancy'] == 0.0
        assert summary['juvenile_fraction'] == 0.0
        assert summary['adult_fraction'] == 0.0