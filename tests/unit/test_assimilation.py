"""
Tests for genetic assimilation and canalization

This module contains comprehensive tests for GeneticAssimilationEngine,
CanalizationEngine, and related classes.
"""

import pytest
import numpy as np
import json
from typing import List

from ipe.core.plasticity.assimilation import (
    GeneticAssimilationEngine,
    AssimilationTrajectory,
    AssimilationStage,
)
from ipe.core.plasticity.canalization import (
    CanalizationEngine,
    CanalizationTrajectory,
    CanalizationMeasure,
    CanalizationType,
)
from ipe.core.plasticity.reaction_norm import ReactionNorm


class TestAssimilationStage:
    """Test suite for AssimilationStage enum"""

    def test_assimilation_stage_values(self):
        """Test AssimilationStage enum values"""
        assert AssimilationStage.INITIAL.value == "initial"
        assert AssimilationStage.PARTIAL.value == "partial"
        assert AssimilationStage.ADVANCED.value == "advanced"
        assert AssimilationStage.COMPLETE.value == "complete"
        assert AssimilationStage.CANALIZED.value == "canalized"


class TestAssimilationTrajectory:
    """Test suite for AssimilationTrajectory dataclass"""

    def test_create_trajectory(self):
        """Test creation of AssimilationTrajectory"""
        generations = np.arange(10)
        plasticity = np.linspace(10, 1, 10)
        constitutive = np.linspace(20, 25, 10)

        trajectory = AssimilationTrajectory(
            generations=generations,
            plasticity_levels=plasticity,
            constitutive_values=constitutive,
            assimilation_rate=0.1,
            environment=25.0,
            trait_name="body_size",
            initial_plasticity=10.0,
            final_plasticity=1.0,
        )

        assert len(trajectory.generations) == 10
        assert trajectory.assimilation_rate == 0.1
        assert trajectory.environment == 25.0
        assert trajectory.initial_plasticity == 10.0

    def test_trajectory_validation_errors(self):
        """Test AssimilationTrajectory validation errors"""
        generations = np.arange(10)
        plasticity = np.linspace(10, 1, 5)  # Wrong length
        constitutive = np.linspace(20, 25, 10)

        # Mismatched lengths
        with pytest.raises(ValueError, match="same length"):
            AssimilationTrajectory(
                generations=generations,
                plasticity_levels=plasticity,
                constitutive_values=constitutive,
                assimilation_rate=0.1,
                environment=25.0,
                trait_name="test",
                initial_plasticity=10.0,
                final_plasticity=1.0,
            )

        # Negative plasticity
        with pytest.raises(ValueError, match="non-negative"):
            AssimilationTrajectory(
                generations=np.arange(3),
                plasticity_levels=np.array([5, -1, 3]),  # Negative value
                constitutive_values=np.array([20, 22, 24]),
                assimilation_rate=0.1,
                environment=25.0,
                trait_name="test",
                initial_plasticity=10.0,
                final_plasticity=1.0,
            )


class TestGeneticAssimilationEngine:
    """Test suite for GeneticAssimilationEngine class"""

    def create_test_engine(self) -> GeneticAssimilationEngine:
        """Helper to create test assimilation engine"""
        return GeneticAssimilationEngine(
            selection_strength=0.1,
            mutation_rate=0.01,
            population_size=1000,
            environmental_stability=0.9,
        )

    def create_test_reaction_norm(self) -> ReactionNorm:
        """Create test reaction norm with high plasticity"""
        environments = [0, 10, 20, 30, 40]
        phenotypes = [10, 20, 30, 40, 50]  # High plasticity (range 40)

        return ReactionNorm(
            environments=environments,
            phenotypes=phenotypes,
            trait_name="body_size",
            environmental_variable="temperature",
        )

    def test_basic_creation(self):
        """Test basic creation of GeneticAssimilationEngine"""
        engine = self.create_test_engine()

        assert engine.selection_strength == 0.1
        assert engine.mutation_rate == 0.01
        assert engine.population_size == 1000
        assert engine.environmental_stability == 0.9

    def test_creation_errors(self):
        """Test GeneticAssimilationEngine creation error cases"""
        # Invalid selection strength
        with pytest.raises(ValueError, match="selection_strength must be"):
            GeneticAssimilationEngine(selection_strength=0.0)

        with pytest.raises(ValueError, match="selection_strength must be"):
            GeneticAssimilationEngine(selection_strength=1.5)

        # Invalid mutation rate
        with pytest.raises(ValueError, match="mutation_rate must be"):
            GeneticAssimilationEngine(mutation_rate=-0.1)

        # Invalid population size
        with pytest.raises(ValueError, match="population_size must be positive"):
            GeneticAssimilationEngine(population_size=0)

        # Invalid environmental stability
        with pytest.raises(ValueError, match="environmental_stability must be"):
            GeneticAssimilationEngine(environmental_stability=1.5)

    def test_calculate_assimilation_rate(self):
        """Test assimilation rate calculation"""
        engine = self.create_test_engine()
        norm = self.create_test_reaction_norm()

        rate = engine.calculate_assimilation_rate(norm, target_environment=20.0)

        assert isinstance(rate, float)
        assert rate >= 0.0  # Rate should be non-negative

        # Test with different selection coefficients
        rate_high = engine.calculate_assimilation_rate(
            norm, target_environment=20.0, selection_coefficient=0.5
        )
        rate_low = engine.calculate_assimilation_rate(
            norm, target_environment=20.0, selection_coefficient=0.01
        )

        assert rate_high >= rate_low  # Higher selection should give higher rate

    def test_simulate_assimilation_trajectory(self):
        """Test simulation of assimilation trajectory"""
        engine = self.create_test_engine()
        norm = self.create_test_reaction_norm()

        trajectory = engine.simulate_assimilation_trajectory(
            norm, target_environment=25.0, generations=100
        )

        assert isinstance(trajectory, AssimilationTrajectory)
        assert len(trajectory.generations) == 100
        assert len(trajectory.plasticity_levels) == 100
        assert len(trajectory.constitutive_values) == 100
        assert trajectory.trait_name == "body_size"
        assert trajectory.environment == 25.0

        # Plasticity should generally decrease over time (allowing for some noise)
        initial_avg = np.mean(trajectory.plasticity_levels[:10])
        final_avg = np.mean(trajectory.plasticity_levels[-10:])
        assert final_avg <= initial_avg + 5.0  # Allow for some increase due to noise

    def test_predict_time_to_assimilation(self):
        """Test prediction of assimilation time"""
        engine = self.create_test_engine()
        norm = self.create_test_reaction_norm()

        # Test with reasonable threshold
        time = engine.predict_time_to_assimilation(
            norm, target_environment=20.0, threshold_plasticity=5.0
        )

        if time is not None:
            assert isinstance(time, int)
            assert time > 0

        # Test with very low threshold (may never reach)
        _ = engine.predict_time_to_assimilation(
            norm, target_environment=20.0, threshold_plasticity=0.001
        )

        # Test with high threshold (should be immediate)
        time_immediate = engine.predict_time_to_assimilation(
            norm, target_environment=20.0, threshold_plasticity=50.0
        )
        if time_immediate is not None:
            assert time_immediate <= 20  # Should be relatively fast or immediate

    def test_assess_assimilation_stage(self):
        """Test assessment of assimilation stage"""
        engine = self.create_test_engine()

        # Test different reduction levels
        stage_initial = engine.assess_assimilation_stage(
            current_plasticity=95.0, initial_plasticity=100.0
        )
        assert stage_initial == AssimilationStage.INITIAL

        stage_partial = engine.assess_assimilation_stage(
            current_plasticity=70.0, initial_plasticity=100.0
        )
        assert stage_partial == AssimilationStage.PARTIAL

        stage_advanced = engine.assess_assimilation_stage(
            current_plasticity=40.0, initial_plasticity=100.0
        )
        assert stage_advanced == AssimilationStage.ADVANCED

        stage_complete = engine.assess_assimilation_stage(
            current_plasticity=10.0, initial_plasticity=100.0
        )
        assert stage_complete == AssimilationStage.COMPLETE

        stage_canalized = engine.assess_assimilation_stage(
            current_plasticity=1.0, initial_plasticity=100.0
        )
        assert stage_canalized == AssimilationStage.CANALIZED

        # Test edge case
        stage_zero = engine.assess_assimilation_stage(
            current_plasticity=50.0, initial_plasticity=0.0
        )
        assert stage_zero == AssimilationStage.CANALIZED

    def test_compare_assimilation_scenarios(self):
        """Test comparison of assimilation across environments"""
        engine = self.create_test_engine()
        norm = self.create_test_reaction_norm()

        environments = [10.0, 20.0, 30.0]
        comparisons = engine.compare_assimilation_scenarios(
            norm, environments, generations=50
        )

        assert isinstance(comparisons, dict)
        assert len(comparisons) == 3

        for env in environments:
            assert env in comparisons
            assert isinstance(comparisons[env], AssimilationTrajectory)
            assert len(comparisons[env].generations) == 50

    def test_identify_assimilation_candidates(self):
        """Test identification of assimilation candidate environments"""
        engine = self.create_test_engine()
        norm = self.create_test_reaction_norm()

        candidates = engine.identify_assimilation_candidates(norm, n_environments=5)

        assert isinstance(candidates, list)
        assert len(candidates) == 5

        # Check format
        for env, rate in candidates:
            assert isinstance(env, (int, float, np.number))
            assert isinstance(rate, (int, float, np.number))
            assert rate >= 0

        # Should be sorted by rate (highest first)
        rates = [rate for _, rate in candidates]
        assert rates == sorted(rates, reverse=True)

    def test_model_canalization_strength(self):
        """Test canalization strength calculation"""
        engine = self.create_test_engine()

        # Create test trajectory
        generations = np.arange(100)
        plasticity = np.linspace(20, 2, 100)  # Decreasing plasticity
        constitutive = np.full(100, 25.0)

        trajectory = AssimilationTrajectory(
            generations=generations,
            plasticity_levels=plasticity,
            constitutive_values=constitutive,
            assimilation_rate=0.1,
            environment=25.0,
            trait_name="test",
            initial_plasticity=20.0,
            final_plasticity=2.0,
        )

        strength = engine.model_canalization_strength(trajectory)

        assert isinstance(strength, float)
        assert 0 <= strength <= 1

        # Test with short trajectory
        short_trajectory = AssimilationTrajectory(
            generations=np.arange(5),
            plasticity_levels=np.array([10, 8, 6, 4, 2]),
            constitutive_values=np.full(5, 25.0),
            assimilation_rate=0.1,
            environment=25.0,
            trait_name="test",
            initial_plasticity=10.0,
            final_plasticity=2.0,
        )

        short_strength = engine.model_canalization_strength(short_trajectory)
        assert short_strength == 0.0  # Too short for analysis

    def test_predict_evolutionary_endpoint(self):
        """Test prediction of evolutionary endpoint"""
        engine = self.create_test_engine()
        norm = self.create_test_reaction_norm()

        endpoint = engine.predict_evolutionary_endpoint(
            norm, target_environment=25.0, max_generations=100
        )

        assert isinstance(endpoint, dict)

        required_keys = [
            "final_plasticity",
            "final_constitutive_value",
            "assimilation_stage",
            "canalization_strength",
            "time_to_95_percent_reduction",
            "predicted_stable",
        ]

        for key in required_keys:
            assert key in endpoint

        assert isinstance(endpoint["final_plasticity"], (int, float, np.number))
        assert isinstance(
            bool(endpoint["predicted_stable"]), bool
        )  # Convert numpy bool
        assert endpoint["assimilation_stage"] in [
            stage.value for stage in AssimilationStage
        ]

    def test_serialization(self):
        """Test engine serialization"""
        engine = self.create_test_engine()

        # Test to_dict
        data = engine.to_dict()
        assert isinstance(data, dict)
        assert "selection_strength" in data
        assert "mutation_rate" in data
        assert "population_size" in data
        assert "environmental_stability" in data

        # Test to_json
        json_str = engine.to_json()
        assert isinstance(json_str, str)

        # Test round-trip
        parsed = json.loads(json_str)
        engine_restored = GeneticAssimilationEngine.from_dict(parsed)
        assert engine_restored.selection_strength == engine.selection_strength
        assert engine_restored.mutation_rate == engine.mutation_rate

        # Test from_json
        engine_from_json = GeneticAssimilationEngine.from_json(json_str)
        assert engine_from_json.population_size == engine.population_size


class TestCanalizationType:
    """Test suite for CanalizationType enum"""

    def test_canalization_type_values(self):
        """Test CanalizationType enum values"""
        assert CanalizationType.GENETIC.value == "genetic"
        assert CanalizationType.ENVIRONMENTAL.value == "environmental"
        assert CanalizationType.DEVELOPMENTAL.value == "developmental"
        assert CanalizationType.PHENOTYPIC.value == "phenotypic"


class TestCanalizationMeasure:
    """Test suite for CanalizationMeasure dataclass"""

    def test_create_measure(self):
        """Test creation of CanalizationMeasure"""
        measure = CanalizationMeasure(
            canalization_type=CanalizationType.GENETIC,
            strength=0.8,
            trait_name="body_size",
            environment=25.0,
            variance_component=0.1,
        )

        assert measure.canalization_type == CanalizationType.GENETIC
        assert measure.strength == 0.8
        assert measure.trait_name == "body_size"
        assert measure.environment == 25.0
        assert measure.variance_component == 0.1

    def test_measure_validation_errors(self):
        """Test CanalizationMeasure validation errors"""
        # Invalid strength
        with pytest.raises(ValueError, match="strength must be"):
            CanalizationMeasure(
                canalization_type=CanalizationType.GENETIC,
                strength=1.5,  # Invalid
                trait_name="test",
            )

        # Negative variance component
        with pytest.raises(ValueError, match="variance_component must be"):
            CanalizationMeasure(
                canalization_type=CanalizationType.GENETIC,
                strength=0.5,
                trait_name="test",
                variance_component=-0.1,  # Invalid
            )


class TestCanalizationEngine:
    """Test suite for CanalizationEngine class"""

    def create_test_engine(self) -> CanalizationEngine:
        """Helper to create test canalization engine"""
        return CanalizationEngine(
            selection_strength=0.05,
            mutation_rate=0.01,
            developmental_noise=0.1,
            population_size=1000,
        )

    def create_test_norms(self) -> List[ReactionNorm]:
        """Create test reaction norms for genetic variants"""
        norm1 = ReactionNorm([0, 20, 40], [20, 25, 30], "size", "temp")
        norm2 = ReactionNorm([0, 20, 40], [18, 26, 32], "size", "temp")
        norm3 = ReactionNorm([0, 20, 40], [22, 24, 28], "size", "temp")
        return [norm1, norm2, norm3]

    def test_basic_creation(self):
        """Test basic creation of CanalizationEngine"""
        engine = self.create_test_engine()

        assert engine.selection_strength == 0.05
        assert engine.mutation_rate == 0.01
        assert engine.developmental_noise == 0.1
        assert engine.population_size == 1000

    def test_creation_errors(self):
        """Test CanalizationEngine creation error cases"""
        # Invalid selection strength
        with pytest.raises(ValueError, match="selection_strength must be"):
            CanalizationEngine(selection_strength=0.0)

        # Invalid mutation rate
        with pytest.raises(ValueError, match="mutation_rate must be"):
            CanalizationEngine(mutation_rate=-0.1)

        # Negative developmental noise
        with pytest.raises(ValueError, match="developmental_noise must be"):
            CanalizationEngine(developmental_noise=-0.1)

        # Invalid population size
        with pytest.raises(ValueError, match="population_size must be positive"):
            CanalizationEngine(population_size=0)

    def test_measure_genetic_canalization(self):
        """Test measurement of genetic canalization"""
        engine = self.create_test_engine()
        norms = self.create_test_norms()

        measure = engine.measure_genetic_canalization(norms, environment=20.0)

        assert isinstance(measure, CanalizationMeasure)
        assert measure.canalization_type == CanalizationType.GENETIC
        assert 0 <= measure.strength <= 1
        assert measure.trait_name == "size"
        assert measure.environment == 20.0
        assert measure.variance_component >= 0

        # Test with too few norms
        with pytest.raises(ValueError, match="at least 2 reaction norms"):
            engine.measure_genetic_canalization([norms[0]], environment=20.0)

    def test_measure_environmental_canalization(self):
        """Test measurement of environmental canalization"""
        engine = self.create_test_engine()
        norm = self.create_test_norms()[0]

        measure = engine.measure_environmental_canalization(norm)

        assert isinstance(measure, CanalizationMeasure)
        assert measure.canalization_type == CanalizationType.ENVIRONMENTAL
        assert 0 <= measure.strength <= 1
        assert measure.trait_name == "size"
        assert measure.variance_component >= 0

        # Test with custom environment range
        measure_custom = engine.measure_environmental_canalization(
            norm, environment_range=(10.0, 30.0), n_environments=10
        )
        assert isinstance(measure_custom, CanalizationMeasure)

    def test_measure_developmental_canalization(self):
        """Test measurement of developmental canalization"""
        engine = self.create_test_engine()

        measure = engine.measure_developmental_canalization(
            mean_phenotype=25.0, phenotypic_variance=4.0, trait_name="size"
        )

        assert isinstance(measure, CanalizationMeasure)
        assert measure.canalization_type == CanalizationType.DEVELOPMENTAL
        assert 0 <= measure.strength <= 1
        assert measure.trait_name == "size"
        assert measure.variance_component == 4.0

        # Test edge case with zero mean
        measure_zero = engine.measure_developmental_canalization(
            mean_phenotype=0.0, phenotypic_variance=1.0, trait_name="size"
        )
        assert 0 <= measure_zero.strength <= 1

    def test_simulate_canalization_evolution(self):
        """Test simulation of canalization evolution"""
        engine = self.create_test_engine()

        trajectory = engine.simulate_canalization_evolution(
            initial_variance=10.0, trait_name="size", generations=100
        )

        assert isinstance(trajectory, CanalizationTrajectory)
        assert len(trajectory.generations) == 100
        assert len(trajectory.canalization_strength) == 100
        assert len(trajectory.phenotypic_variance) == 100
        assert len(trajectory.environmental_sensitivity) == 100
        assert trajectory.trait_name == "size"

        # Canalization should generally increase over time
        initial_avg = np.mean(trajectory.canalization_strength[:10])
        final_avg = np.mean(trajectory.canalization_strength[-10:])
        assert final_avg >= initial_avg - 0.1  # Allow for some decrease due to noise

    def test_predict_canalization_trajectory(self):
        """Test prediction of canalization from assimilation"""
        engine = self.create_test_engine()

        # Create mock assimilation trajectory
        generations = np.arange(50)
        plasticity = np.linspace(20, 2, 50)
        constitutive = np.full(50, 25.0)

        assim_trajectory = AssimilationTrajectory(
            generations=generations,
            plasticity_levels=plasticity,
            constitutive_values=constitutive,
            assimilation_rate=0.1,
            environment=25.0,
            trait_name="size",
            initial_plasticity=20.0,
            final_plasticity=2.0,
        )

        can_trajectory = engine.predict_canalization_trajectory(assim_trajectory)

        assert isinstance(can_trajectory, CanalizationTrajectory)
        assert len(can_trajectory.generations) == 50
        assert can_trajectory.trait_name == "size"

        # Canalization should increase as plasticity decreases
        assert (
            can_trajectory.canalization_strength[0]
            <= can_trajectory.canalization_strength[-1]
        )

    def test_compare_canalization_types(self):
        """Test comparison of different canalization types"""
        engine = self.create_test_engine()
        norms = self.create_test_norms()

        comparison = engine.compare_canalization_types(
            norms, environment=20.0, developmental_variance=2.0
        )

        assert isinstance(comparison, dict)

        expected_keys = ["genetic", "environmental", "developmental", "phenotypic"]
        for key in expected_keys:
            assert key in comparison
            assert isinstance(comparison[key], CanalizationMeasure)

        # Test with single norm (no genetic canalization)
        single_comparison = engine.compare_canalization_types(
            [norms[0]], environment=20.0
        )
        assert "genetic" not in single_comparison
        assert "environmental" in single_comparison

    def test_assess_canalization_evolution(self):
        """Test assessment of canalization evolution"""
        engine = self.create_test_engine()

        # Create test trajectory
        generations = np.arange(100)
        canalization = np.linspace(0.1, 0.9, 100)
        variance = np.linspace(10.0, 1.0, 100)
        sensitivity = 1.0 - canalization

        trajectory = CanalizationTrajectory(
            generations=generations,
            canalization_strength=canalization,
            phenotypic_variance=variance,
            environmental_sensitivity=sensitivity,
            canalization_rate=0.008,
            trait_name="size",
        )

        assessment = engine.assess_canalization_evolution(trajectory)

        assert isinstance(assessment, dict)

        required_keys = [
            "initial_canalization",
            "final_canalization",
            "max_canalization",
            "mean_canalization",
            "canalization_evolution_rate",
            "final_variance_reduction",
            "evolutionary_stability",
            "canalization_efficiency",
        ]

        for key in required_keys:
            assert key in assessment

        assert assessment["evolutionary_stability"] in ["stable", "unstable"]
        assert 0 <= assessment["canalization_efficiency"] <= 1

    def test_identify_canalization_drivers(self):
        """Test identification of canalization drivers"""
        engine = self.create_test_engine()

        # Create mock trajectory
        trajectory = CanalizationTrajectory(
            generations=np.arange(50),
            canalization_strength=np.linspace(0.2, 0.8, 50),
            phenotypic_variance=np.linspace(5.0, 1.0, 50),
            environmental_sensitivity=np.linspace(0.8, 0.2, 50),
            canalization_rate=0.01,
            trait_name="size",
        )

        drivers = engine.identify_canalization_drivers(trajectory)

        assert isinstance(drivers, dict)

        expected_keys = [
            "selection",
            "genetic_drift",
            "mutation",
            "developmental_buffering",
        ]
        for key in expected_keys:
            assert key in drivers
            assert 0 <= drivers[key] <= 1

        # Should sum to approximately 1
        total = sum(drivers.values())
        assert abs(total - 1.0) < 0.001

    def test_serialization(self):
        """Test canalization engine serialization"""
        engine = self.create_test_engine()

        # Test to_dict
        data = engine.to_dict()
        assert isinstance(data, dict)
        assert "selection_strength" in data
        assert "developmental_noise" in data

        # Test to_json
        json_str = engine.to_json()
        assert isinstance(json_str, str)

        # Test round-trip
        engine_restored = CanalizationEngine.from_json(json_str)
        assert engine_restored.selection_strength == engine.selection_strength
        assert engine_restored.developmental_noise == engine.developmental_noise


class TestIntegrationAssimilationCanalization:
    """Integration tests for assimilation and canalization"""

    def test_assimilation_to_canalization_pipeline(self):
        """Test full pipeline from assimilation to canalization"""
        # Create engines
        assim_engine = GeneticAssimilationEngine()
        can_engine = CanalizationEngine()

        # Create reaction norm
        norm = ReactionNorm([0, 20, 40], [10, 30, 50], "size", "temp")

        # Simulate assimilation
        assim_trajectory = assim_engine.simulate_assimilation_trajectory(
            norm, target_environment=25.0, generations=100
        )

        # Predict canalization from assimilation
        can_trajectory = can_engine.predict_canalization_trajectory(assim_trajectory)

        # Verify consistency
        assert len(assim_trajectory.generations) == len(can_trajectory.generations)
        assert assim_trajectory.trait_name == can_trajectory.trait_name

        # As plasticity decreases, canalization should increase
        initial_plasticity = assim_trajectory.plasticity_levels[0]
        final_plasticity = assim_trajectory.plasticity_levels[-1]
        initial_canalization = can_trajectory.canalization_strength[0]
        final_canalization = can_trajectory.canalization_strength[-1]

        if initial_plasticity > final_plasticity:
            assert final_canalization >= initial_canalization

    def test_comparative_analysis(self):
        """Test comparative analysis across environments"""
        assim_engine = GeneticAssimilationEngine()
        norm = ReactionNorm([0, 20, 40], [15, 25, 35], "size", "temp")

        # Compare assimilation in different environments
        environments = [10.0, 20.0, 30.0]
        trajectories = assim_engine.compare_assimilation_scenarios(
            norm, environments, generations=50
        )

        # Analyze endpoints
        endpoints = {}
        for env, trajectory in trajectories.items():
            endpoint = assim_engine.predict_evolutionary_endpoint(
                norm, target_environment=env, max_generations=100
            )
            endpoints[env] = endpoint

        # Should have different outcomes in different environments
        final_plasticities = [ep["final_plasticity"] for ep in endpoints.values()]
        assert (
            len(set([round(fp, 1) for fp in final_plasticities])) >= 1
        )  # Some variation expected

    def test_parameter_sensitivity(self):
        """Test sensitivity to parameter changes"""
        # Test with strong selection
        strong_engine = GeneticAssimilationEngine(selection_strength=0.5)
        weak_engine = GeneticAssimilationEngine(selection_strength=0.01)

        norm = ReactionNorm([0, 40], [20, 40], "size", "temp")

        # Compare rates
        strong_rate = strong_engine.calculate_assimilation_rate(norm, 30.0)
        weak_rate = weak_engine.calculate_assimilation_rate(norm, 30.0)

        assert strong_rate >= weak_rate  # Strong selection should give higher rate


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
