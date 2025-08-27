"""
Tests for Rapid Evolution Mode (EVOL-004)

This module tests the rapid evolution simulator, environmental change scenarios,
plasticity evolution tracking, and contemporary evolution models.
"""

import pytest
import numpy as np
from typing import List, Dict

from ipe.core.physiology.state import PhysiologicalState
from ipe.simulation.population import Population, Individual
from ipe.simulation.genetic_architecture import create_default_architecture
from ipe.simulation.mutation import create_default_mutation_engine
from ipe.simulation.selection import TruncationSelection
from ipe.simulation.rapid_evolution import (
    EnvironmentalChange, PlasticityEvolutionTracker, RapidEvolutionSimulator,
    create_freshwater_invasion_scenario, create_altitude_adaptation_scenario
)
from ipe.simulation.contemporary import (
    ContemporaryEvolutionModel, ExperimentalEvolutionModel,
    create_predefined_scenarios
)


class TestEnvironmentalChange:
    """Test EnvironmentalChange functionality"""
    
    @pytest.fixture
    def sample_environments(self):
        """Create sample initial and final environments"""
        initial = PhysiologicalState(po2=20.0, temperature=15.0, altitude=100.0, salinity=35.0)
        final = PhysiologicalState(po2=18.0, temperature=25.0, altitude=100.0, salinity=0.0)
        return initial, final
    
    def test_basic_creation(self, sample_environments):
        """Test creating environmental change scenario"""
        initial, final = sample_environments
        
        change = EnvironmentalChange(
            name="test_change",
            description="Test environmental change",
            start_generation=5,
            duration=20,
            initial_environment=initial,
            final_environment=final,
            change_type="gradual"
        )
        
        assert change.name == "test_change"
        assert change.start_generation == 5
        assert change.duration == 20
        assert change.change_type == "gradual"
    
    def test_gradual_change(self, sample_environments):
        """Test gradual environmental change"""
        initial, final = sample_environments
        
        change = EnvironmentalChange(
            name="gradual",
            description="Gradual change",
            start_generation=0,
            duration=10,
            initial_environment=initial,
            final_environment=final,
            change_type="gradual"
        )
        
        # Test environments at different generations
        env_0 = change.get_environment_at_generation(0)  # Start
        env_5 = change.get_environment_at_generation(5)  # Middle
        env_10 = change.get_environment_at_generation(10)  # End
        
        assert env_0.temperature == 15.0  # Initial
        assert env_10.temperature == 25.0  # Final
        assert 15.0 < env_5.temperature < 25.0  # Intermediate
        
        assert env_0.salinity == 35.0  # Initial
        assert env_10.salinity == 0.0  # Final
    
    def test_sudden_change(self, sample_environments):
        """Test sudden environmental change"""
        initial, final = sample_environments
        
        change = EnvironmentalChange(
            name="sudden",
            description="Sudden change",
            start_generation=0,
            duration=10,
            initial_environment=initial,
            final_environment=final,
            change_type="sudden"
        )
        
        env_4 = change.get_environment_at_generation(4)
        env_6 = change.get_environment_at_generation(6)
        
        # Should switch at midpoint
        assert env_4.temperature == 15.0  # Still initial
        assert env_6.temperature == 25.0  # Now final
    
    def test_oscillating_change(self, sample_environments):
        """Test oscillating environmental change"""
        initial, final = sample_environments
        
        change = EnvironmentalChange(
            name="oscillating",
            description="Oscillating change",
            start_generation=0,
            duration=20,
            initial_environment=initial,
            final_environment=final,
            change_type="oscillating",
            oscillation_period=10
        )
        
        # Test that oscillation occurs
        envs = [change.get_environment_at_generation(i) for i in range(20)]
        temperatures = [env.temperature for env in envs]
        
        # Should have variation
        assert min(temperatures) < max(temperatures)
    
    def test_before_and_after_change(self, sample_environments):
        """Test environment before start and after end"""
        initial, final = sample_environments
        
        change = EnvironmentalChange(
            name="test",
            description="Test",
            start_generation=5,
            duration=10,
            initial_environment=initial,
            final_environment=final
        )
        
        env_before = change.get_environment_at_generation(3)  # Before start
        env_after = change.get_environment_at_generation(20)  # After end
        
        assert env_before.temperature == 15.0  # Initial
        assert env_after.temperature == 25.0  # Final


class TestPlasticityEvolutionTracker:
    """Test PlasticityEvolutionTracker functionality"""
    
    @pytest.fixture
    def sample_population(self):
        """Create sample population"""
        state = PhysiologicalState(po2=18.0, temperature=20.0, altitude=100.0)
        individuals = []
        
        for i in range(10):
            individual = Individual(
                id=i,
                physiological_state=state,
                genetic_values={"locus1": np.random.normal(0, 1)}
            )
            individuals.append(individual)
        
        return individuals
    
    def test_tracker_creation(self):
        """Test creating plasticity tracker"""
        tracker = PlasticityEvolutionTracker()
        assert len(tracker.generation_data) == 0
    
    def test_record_generation(self, sample_population):
        """Test recording generation data"""
        tracker = PlasticityEvolutionTracker()
        environment = PhysiologicalState(po2=18.0, temperature=20.0, altitude=100.0)
        
        tracker.record_generation(1, sample_population, environment)
        
        assert len(tracker.generation_data) == 1
        data = tracker.generation_data[0]
        
        assert data['generation'] == 1
        assert data['population_size'] == 10
        assert 'environmental_conditions' in data
    
    def test_get_trajectory(self, sample_population):
        """Test getting plasticity trajectory"""
        tracker = PlasticityEvolutionTracker()
        environment = PhysiologicalState(po2=18.0, temperature=20.0, altitude=100.0)
        
        # Record multiple generations
        for gen in range(5):
            tracker.record_generation(gen, sample_population, environment)
        
        trajectory = tracker.get_plasticity_trajectory()
        
        assert 'generations' in trajectory
        assert len(trajectory['generations']) == 5
        assert trajectory['generations'] == [0, 1, 2, 3, 4]


class TestRapidEvolutionSimulator:
    """Test RapidEvolutionSimulator functionality"""
    
    @pytest.fixture
    def basic_simulator(self):
        """Create basic simulator for testing"""
        # Create components
        arch = create_default_architecture()
        initial_state = PhysiologicalState(po2=20.0, temperature=15.0, altitude=100.0)
        population = Population(20, initial_state)
        mutation_engine = create_default_mutation_engine(arch)
        selection_strategy = TruncationSelection(survival_fraction=0.7)
        
        def fitness_function(individual: Individual, environment: PhysiologicalState) -> float:
            return 0.5 + 0.5 * np.random.random()
        
        return RapidEvolutionSimulator(
            population, arch, mutation_engine, selection_strategy, fitness_function
        )
    
    def test_simulator_creation(self, basic_simulator):
        """Test creating rapid evolution simulator"""
        assert basic_simulator.population is not None
        assert basic_simulator.genetic_architecture is not None
        assert basic_simulator.mutation_engine is not None
        assert basic_simulator.selection_strategy is not None
        assert basic_simulator.fitness_function is not None
        assert len(basic_simulator.generation_callbacks) == 0
    
    def test_add_visualization_callback(self, basic_simulator):
        """Test adding visualization callback"""
        def callback(generation, population):
            pass
        
        basic_simulator.add_visualization_callback(callback)
        assert len(basic_simulator.generation_callbacks) == 1
    
    def test_simulate_constant_environment(self, basic_simulator):
        """Test simulation in constant environment"""
        environment = PhysiologicalState(po2=20.0, temperature=15.0, altitude=100.0)
        
        np.random.seed(42)  # For reproducible results
        results = list(basic_simulator.simulate_constant_environment(environment, generations=3))
        
        assert len(results) == 3
        
        # Check that each result has expected structure
        for i, result in enumerate(results):
            assert result['generation'] == i
            assert 'population_size' in result
            assert 'mean_fitness' in result
            assert 'environment' in result
    
    def test_simulate_environmental_change(self, basic_simulator):
        """Test simulation with environmental change"""
        initial_env = PhysiologicalState(po2=20.0, temperature=15.0, altitude=100.0)
        final_env = PhysiologicalState(po2=18.0, temperature=25.0, altitude=100.0)
        
        scenario = EnvironmentalChange(
            name="test_change",
            description="Test",
            start_generation=0,
            duration=5,
            initial_environment=initial_env,
            final_environment=final_env,
            change_type="gradual"
        )
        
        np.random.seed(42)
        results = list(basic_simulator.simulate_environmental_change(scenario, total_generations=3))
        
        assert len(results) == 3
        
        # Check environmental progression
        for i, result in enumerate(results):
            env_conditions = result['environment']
            assert 'po2' in env_conditions
            assert 'temperature' in env_conditions
    
    def test_get_evolution_summary(self, basic_simulator):
        """Test getting evolution summary"""
        environment = PhysiologicalState(po2=20.0, temperature=15.0, altitude=100.0)
        
        # Run short simulation
        np.random.seed(42)
        list(basic_simulator.simulate_constant_environment(environment, generations=3))
        
        summary = basic_simulator.get_evolution_summary()
        
        assert 'total_generations' in summary
        assert 'fitness_trajectory' in summary
        assert 'population_trajectory' in summary
        assert 'summary_statistics' in summary
        
        assert summary['total_generations'] == 3


class TestPredefinedScenarios:
    """Test predefined evolution scenarios"""
    
    def test_freshwater_invasion_scenario(self):
        """Test freshwater invasion scenario"""
        scenario = create_freshwater_invasion_scenario()
        
        assert scenario.name == "freshwater_invasion"
        assert scenario.initial_environment.salinity == 35.0  # Marine
        assert scenario.final_environment.salinity == 0.0  # Freshwater
        assert scenario.change_type == "gradual"
    
    def test_altitude_adaptation_scenario(self):
        """Test altitude adaptation scenario"""
        scenario = create_altitude_adaptation_scenario()
        
        assert scenario.name == "altitude_adaptation"
        assert scenario.initial_environment.altitude == 100.0  # Lowland
        assert scenario.final_environment.altitude == 3000.0  # Highland
        assert scenario.initial_environment.po2 > scenario.final_environment.po2  # Less oxygen at altitude


class TestContemporaryEvolutionModel:
    """Test ContemporaryEvolutionModel functionality"""
    
    def test_create_urban_adaptation_model(self):
        """Test creating urban adaptation model"""
        model = ContemporaryEvolutionModel.create_urban_adaptation_model(population_size=50)
        
        assert model.population.current_size() == 50
        assert model.genetic_architecture is not None
        assert model.mutation_engine is not None
        assert model.selection_strategy is not None
        assert model.fitness_function is not None
    
    def test_create_climate_change_model(self):
        """Test creating climate change model"""
        model = ContemporaryEvolutionModel.create_climate_change_model(population_size=75)
        
        assert model.population.current_size() == 75
        assert model.genetic_architecture is not None
    
    def test_create_invasion_model(self):
        """Test creating invasion model"""
        model = ContemporaryEvolutionModel.create_invasion_model(population_size=30)
        
        assert model.population.current_size() == 30
        assert model.genetic_architecture is not None
        
        # Should have higher mutation rate for rapid adaptation
        base_rate = 1e-4
        assert model.mutation_engine.parameters.base_mutation_rate >= base_rate
    
    def test_urban_fitness_function(self):
        """Test urban adaptation fitness function"""
        model = ContemporaryEvolutionModel.create_urban_adaptation_model()
        
        # Create test individual and environment
        state = PhysiologicalState(po2=18.0, temperature=20.0, altitude=100.0)
        individual = Individual(id=1, physiological_state=state)
        environment = PhysiologicalState(po2=17.0, temperature=22.0, altitude=100.0)
        
        fitness = model.fitness_function(individual, environment)
        
        assert 0.0 <= fitness <= 1.5  # Reasonable fitness range
    
    def test_climate_fitness_function(self):
        """Test climate change fitness function"""
        model = ContemporaryEvolutionModel.create_climate_change_model()
        
        state = PhysiologicalState(po2=20.0, temperature=15.0, altitude=200.0)
        individual = Individual(id=1, physiological_state=state)
        environment = PhysiologicalState(po2=20.0, temperature=18.0, altitude=200.0)
        
        fitness = model.fitness_function(individual, environment)
        
        assert fitness >= 0.0
    
    def test_invasion_fitness_function(self):
        """Test invasion fitness function"""
        model = ContemporaryEvolutionModel.create_invasion_model()
        
        state = PhysiologicalState(po2=19.0, temperature=20.0, altitude=0.0, salinity=35.0)
        individual = Individual(id=1, physiological_state=state)
        environment = PhysiologicalState(po2=19.0, temperature=20.0, altitude=0.0, salinity=10.0)
        
        fitness = model.fitness_function(individual, environment)
        
        assert fitness >= 0.0


class TestExperimentalEvolutionModel:
    """Test ExperimentalEvolutionModel functionality"""
    
    def test_create_selection_experiment(self):
        """Test creating selection experiment"""
        target_value = 0.5
        model = ExperimentalEvolutionModel.create_selection_experiment(
            trait_target=target_value, 
            population_size=100
        )
        
        assert model.population.current_size() == 100
        assert model.genetic_architecture is not None
        
        # Test fitness function
        state = PhysiologicalState(po2=20.0, temperature=22.0, altitude=100.0)
        individual = Individual(id=1, physiological_state=state, genetic_values={'cardio1': 0.5})
        environment = PhysiologicalState(po2=20.0, temperature=22.0, altitude=100.0)
        
        fitness = model.fitness_function(individual, environment)
        
        # Should have high fitness when trait matches target
        assert fitness > 0.8
    
    def test_create_common_garden_experiment(self):
        """Test creating common garden experiment"""
        environments = [
            PhysiologicalState(po2=20.0, temperature=15.0, altitude=100.0),
            PhysiologicalState(po2=18.0, temperature=25.0, altitude=100.0),
            PhysiologicalState(po2=16.0, temperature=30.0, altitude=100.0)
        ]
        
        simulators = ExperimentalEvolutionModel.create_common_garden_experiment(
            environments, population_size=50
        )
        
        assert len(simulators) == 3
        assert "environment_0" in simulators
        assert "environment_1" in simulators
        assert "environment_2" in simulators
        
        # Each simulator should have correct population size
        for simulator in simulators.values():
            assert simulator.population.current_size() == 50


class TestPredefinedScenarioLibrary:
    """Test predefined scenario library"""
    
    def test_create_predefined_scenarios(self):
        """Test creating predefined scenarios"""
        scenarios = create_predefined_scenarios()
        
        assert len(scenarios) > 0
        
        expected_scenarios = [
            'urban_heat_island', 
            'pollution_gradient',
            'seasonal_variation',
            'habitat_fragmentation'
        ]
        
        for scenario_name in expected_scenarios:
            assert scenario_name in scenarios
            scenario = scenarios[scenario_name]
            assert isinstance(scenario, EnvironmentalChange)
            assert scenario.name == scenario_name
    
    def test_urban_heat_island_scenario(self):
        """Test urban heat island scenario specifics"""
        scenarios = create_predefined_scenarios()
        uhi = scenarios['urban_heat_island']
        
        assert uhi.change_type == "gradual"
        assert uhi.initial_environment.temperature < uhi.final_environment.temperature
        assert uhi.initial_environment.po2 > uhi.final_environment.po2  # Pollution effect
    
    def test_seasonal_variation_scenario(self):
        """Test seasonal variation scenario"""
        scenarios = create_predefined_scenarios()
        seasonal = scenarios['seasonal_variation']
        
        assert seasonal.change_type == "oscillating"
        assert seasonal.oscillation_period == 10
        assert seasonal.duration == 100  # Long-term
    
    def test_habitat_fragmentation_scenario(self):
        """Test habitat fragmentation scenario"""
        scenarios = create_predefined_scenarios()
        fragmentation = scenarios['habitat_fragmentation']
        
        assert fragmentation.change_type == "sudden"
        assert fragmentation.start_generation == 20
        assert fragmentation.duration == 5  # Short, sudden change


class TestIntegrationRapidEvolution:
    """Integration tests for rapid evolution components"""
    
    def test_end_to_end_simulation(self):
        """Test complete end-to-end evolution simulation"""
        # Create urban adaptation scenario
        model = ContemporaryEvolutionModel.create_urban_adaptation_model(population_size=30)
        
        # Create environmental change
        scenarios = create_predefined_scenarios()
        urban_scenario = scenarios['urban_heat_island']
        
        # Run short simulation
        np.random.seed(42)
        results = list(model.simulate_environmental_change(urban_scenario, total_generations=5))
        
        assert len(results) == 5
        
        # Check that simulation progressed
        first_gen = results[0]
        last_gen = results[-1]
        
        assert first_gen['generation'] == 0
        assert last_gen['generation'] == 4
        
        # Get summary
        summary = model.get_evolution_summary()
        assert summary['total_generations'] == 5
        
        # Check plasticity tracking
        plasticity_trajectory = summary['plasticity_evolution']
        assert 'generations' in plasticity_trajectory
    
    def test_multiple_model_comparison(self):
        """Test comparing different evolution models"""
        # Create different models
        urban_model = ContemporaryEvolutionModel.create_urban_adaptation_model(50)
        climate_model = ContemporaryEvolutionModel.create_climate_change_model(50)
        invasion_model = ContemporaryEvolutionModel.create_invasion_model(25)
        
        models = [urban_model, climate_model, invasion_model]
        
        # All models should be properly configured
        for model in models:
            assert model.population is not None
            assert model.genetic_architecture is not None
            assert model.mutation_engine is not None
            assert model.selection_strategy is not None
            assert model.fitness_function is not None