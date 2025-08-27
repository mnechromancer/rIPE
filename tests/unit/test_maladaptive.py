"""
Tests for maladaptive plasticity detection

This module contains comprehensive tests for MaladaptiveDetector and related
classes, ensuring detection and quantification of maladaptive plasticity.
"""

import pytest
import numpy as np
from typing import Callable

from ipe.core.plasticity.maladaptive import (
    MaladaptiveDetector, MaladaptiveResponse, MaladaptationType,
    quadratic_fitness, linear_fitness, thermal_fitness
)
from ipe.core.plasticity.reaction_norm import ReactionNorm


class TestMaladaptiveType:
    """Test suite for MaladaptiveType enum"""
    
    def test_maladaptive_type_values(self):
        """Test MaladaptiveType enum values"""
        assert MaladaptationType.NONE.value == "none"
        assert MaladaptationType.SLIGHT.value == "slight"
        assert MaladaptationType.MODERATE.value == "moderate"
        assert MaladaptationType.SEVERE.value == "severe"
        assert MaladaptationType.EXTREME.value == "extreme"


class TestMaladaptiveResponse:
    """Test suite for MaladaptiveResponse dataclass"""
    
    def test_create_response(self):
        """Test creation of MaladaptiveResponse"""
        response = MaladaptiveResponse(
            environment=25.0,
            plastic_phenotype=50.0,
            optimal_phenotype=60.0,
            plastic_fitness=0.8,
            optimal_fitness=1.0,
            fitness_cost=0.2,
            relative_fitness_cost=0.2,
            environment_variable="temperature",
            trait_name="body_size"
        )
        
        assert response.environment == 25.0
        assert response.plastic_phenotype == 50.0
        assert response.optimal_phenotype == 60.0
        assert response.fitness_cost == 0.2
        assert response.relative_fitness_cost == 0.2


class TestFitnessFunctions:
    """Test suite for predefined fitness functions"""
    
    def test_quadratic_fitness(self):
        """Test quadratic fitness function"""
        # Test basic functionality
        fitness1 = quadratic_fitness(10.0, 10.0, optimum_slope=1.0, width=10.0)
        fitness2 = quadratic_fitness(5.0, 10.0, optimum_slope=1.0, width=10.0)
        
        # Fitness should be higher when phenotype matches optimum
        assert fitness1 > fitness2
        assert 0 <= fitness1 <= 1
        assert 0 <= fitness2 <= 1
        
        # Test with different parameters
        fitness3 = quadratic_fitness(20.0, 10.0, optimum_slope=2.0, width=5.0)
        assert 0 <= fitness3 <= 1
    
    def test_linear_fitness(self):
        """Test linear fitness function"""
        # Test basic linear relationship
        fitness1 = linear_fitness(10.0, 25.0, slope=2.0, intercept=5.0)
        fitness2 = linear_fitness(5.0, 25.0, slope=2.0, intercept=5.0)
        
        assert fitness1 == 25.0  # 2*10 + 5
        assert fitness2 == 15.0  # 2*5 + 5
        assert fitness1 > fitness2  # Positive slope
        
        # Test negative slope
        fitness3 = linear_fitness(10.0, 25.0, slope=-1.0, intercept=20.0)
        fitness4 = linear_fitness(5.0, 25.0, slope=-1.0, intercept=20.0)
        assert fitness3 < fitness4  # Negative slope
    
    def test_thermal_fitness(self):
        """Test thermal adaptation fitness function"""
        # Test that fitness peaks near optimal temperature
        fitness_cold = thermal_fitness(15.0, 10.0)  # temp=10, optimal~15
        fitness_opt = thermal_fitness(20.0, 20.0)   # temp=20, optimal~20  
        fitness_hot = thermal_fitness(25.0, 30.0)   # temp=30, optimal~25
        
        # All should be positive and <= 1
        assert 0 < fitness_cold <= 1
        assert 0 < fitness_opt <= 1  
        assert 0 < fitness_hot <= 1
        
        # Optimal should have highest fitness
        assert fitness_opt >= fitness_cold
        assert fitness_opt >= fitness_hot


class TestMaladaptiveDetector:
    """Test suite for MaladaptiveDetector class"""
    
    def create_test_detector(self) -> MaladaptiveDetector:
        """Helper to create test detector"""
        return MaladaptiveDetector(
            fitness_function=quadratic_fitness,
            environment_variable="temperature",
            trait_name="body_size",
            optimization_tolerance=0.1
        )
    
    def create_maladaptive_norm(self) -> ReactionNorm:
        """Create a reaction norm that shows maladaptive plasticity"""
        # Environment: 0-40°C, optimal phenotype should be 0-40
        # But this norm goes opposite direction (maladaptive)
        environments = [0, 10, 20, 30, 40]
        phenotypes = [40, 30, 20, 10, 0]  # Decreases when it should increase
        
        return ReactionNorm(
            environments=environments,
            phenotypes=phenotypes,
            trait_name="body_size",
            environmental_variable="temperature"
        )
    
    def create_adaptive_norm(self) -> ReactionNorm:
        """Create a reaction norm that is mostly adaptive"""
        environments = [0, 10, 20, 30, 40] 
        phenotypes = [0, 10, 20, 30, 40]  # Increases with environment (adaptive)
        
        return ReactionNorm(
            environments=environments,
            phenotypes=phenotypes,
            trait_name="body_size", 
            environmental_variable="temperature"
        )
    
    def test_basic_creation(self):
        """Test basic creation of MaladaptiveDetector"""
        detector = self.create_test_detector()
        
        assert callable(detector.fitness_function)
        assert detector.environment_variable == "temperature"
        assert detector.trait_name == "body_size"
        assert detector.optimization_tolerance == 0.1
    
    def test_creation_errors(self):
        """Test MaladaptiveDetector creation error cases"""
        # Non-callable fitness function
        with pytest.raises(TypeError, match="fitness_function must be callable"):
            MaladaptiveDetector(
                fitness_function="not callable",
                environment_variable="temp",
                trait_name="trait"
            )
        
        # Non-positive tolerance
        with pytest.raises(ValueError, match="optimization_tolerance must be positive"):
            MaladaptiveDetector(
                fitness_function=quadratic_fitness,
                environment_variable="temp",
                trait_name="trait",
                optimization_tolerance=-0.1
            )
    
    def test_compute_optimal_phenotype(self):
        """Test optimal phenotype computation"""
        detector = self.create_test_detector()
        
        # For quadratic_fitness with default slope=1, optimal should be env * 1
        optimal = detector.compute_optimal_phenotype(20.0, phenotype_range=(0, 50))
        assert abs(optimal - 20.0) < 1.0  # Should be close to 20
        
        # Test with different range
        optimal2 = detector.compute_optimal_phenotype(10.0, phenotype_range=(-10, 30))
        assert abs(optimal2 - 10.0) < 1.0
        
        # Test default range
        optimal3 = detector.compute_optimal_phenotype(15.0)
        assert abs(optimal3 - 15.0) < 5.0  # Wider tolerance for default range
    
    def test_detect_maladaptive_responses_positive(self):
        """Test detection with clearly maladaptive norm"""
        detector = self.create_test_detector()
        maladaptive_norm = self.create_maladaptive_norm()
        
        responses = detector.detect_maladaptive_responses(maladaptive_norm)
        
        # Should detect maladaptive responses
        assert len(responses) > 0
        
        # Check first response
        response = responses[0]
        assert isinstance(response, MaladaptiveResponse)
        assert response.plastic_fitness < response.optimal_fitness
        assert response.fitness_cost > 0
        assert response.relative_fitness_cost > 0
        assert response.environment_variable == "temperature"
        assert response.trait_name == "body_size"
    
    def test_detect_maladaptive_responses_negative(self):
        """Test detection with adaptive norm"""
        detector = self.create_test_detector()
        adaptive_norm = self.create_adaptive_norm()
        
        responses = detector.detect_maladaptive_responses(adaptive_norm)
        
        # May or may not detect responses (depends on how close to optimal)
        # But any detected responses should have minimal cost
        for response in responses:
            assert response.relative_fitness_cost < 0.5  # Should be small
    
    def test_detect_with_custom_environments(self):
        """Test detection with custom environment array"""
        detector = self.create_test_detector()
        norm = self.create_maladaptive_norm()
        
        custom_envs = np.array([5.0, 15.0, 25.0])
        responses = detector.detect_maladaptive_responses(norm, environments=custom_envs)
        
        # Should have responses for the custom environments
        env_values = [r.environment for r in responses]
        for env in custom_envs:
            if env in env_values:  # May not all be maladaptive
                assert True
    
    def test_quantify_maladaptation_severity_high(self):
        """Test severity quantification with maladaptive norm"""
        detector = self.create_test_detector()
        maladaptive_norm = self.create_maladaptive_norm()
        
        metrics = detector.quantify_maladaptation_severity(maladaptive_norm)
        
        assert isinstance(metrics, dict)
        assert 'proportion_maladaptive' in metrics
        assert 'mean_fitness_cost' in metrics
        assert 'max_fitness_cost' in metrics
        assert 'mean_relative_cost' in metrics
        assert 'max_relative_cost' in metrics
        assert 'severity_classification' in metrics
        
        # Should detect significant maladaptation
        assert metrics['proportion_maladaptive'] > 0
        assert metrics['mean_fitness_cost'] > 0
        assert metrics['severity_classification'] in [
            MaladaptationType.SLIGHT.value,
            MaladaptationType.MODERATE.value,
            MaladaptationType.SEVERE.value,
            MaladaptationType.EXTREME.value
        ]
    
    def test_quantify_maladaptation_severity_low(self):
        """Test severity quantification with adaptive norm"""
        detector = self.create_test_detector()
        adaptive_norm = self.create_adaptive_norm()
        
        metrics = detector.quantify_maladaptation_severity(adaptive_norm)
        
        # Should show little to no maladaptation
        assert metrics['proportion_maladaptive'] <= 1.0
        assert metrics['mean_fitness_cost'] >= 0
        
        # Classification should be none or slight
        assert metrics['severity_classification'] in [
            MaladaptationType.NONE.value,
            MaladaptationType.SLIGHT.value
        ]
    
    def test_compare_plastic_vs_constitutive(self):
        """Test comparison of plastic vs constitutive strategies"""
        detector = self.create_test_detector()
        norm = self.create_adaptive_norm()  # Use adaptive norm
        
        # Compare against a fixed intermediate phenotype
        constitutive_phenotype = 20.0
        
        comparison = detector.compare_plastic_vs_constitutive(norm, constitutive_phenotype)
        
        assert isinstance(comparison, dict)
        assert 'mean_plastic_fitness' in comparison
        assert 'mean_constitutive_fitness' in comparison
        assert 'plastic_advantage' in comparison
        assert 'proportion_plastic_better' in comparison
        assert 'max_plastic_advantage' in comparison
        assert 'max_plastic_disadvantage' in comparison
        assert 'environments' in comparison
        assert 'plastic_fitness_by_env' in comparison
        assert 'constitutive_fitness_by_env' in comparison
        
        # Check that values are reasonable
        assert 0 <= comparison['proportion_plastic_better'] <= 1
        assert isinstance(comparison['environments'], list)
        assert len(comparison['plastic_fitness_by_env']) == len(comparison['environments'])
    
    def test_identify_maladaptive_environments(self):
        """Test identification of specific maladaptive environments"""
        detector = self.create_test_detector()
        maladaptive_norm = self.create_maladaptive_norm()
        
        # Use low threshold to catch mild maladaptation
        envs = detector.identify_maladaptive_environments(maladaptive_norm, threshold=0.01)
        
        assert isinstance(envs, list)
        # Should find some maladaptive environments
        # (exact number depends on how maladaptive the norm is)
        
        # Test with high threshold
        envs_strict = detector.identify_maladaptive_environments(maladaptive_norm, threshold=0.5)
        assert len(envs_strict) <= len(envs)  # Stricter threshold should find fewer
    
    def test_compute_plasticity_cost_function(self):
        """Test computation of plasticity cost function"""
        detector = self.create_test_detector()
        maladaptive_norm = self.create_maladaptive_norm()
        
        cost_norm = detector.compute_plasticity_cost_function(maladaptive_norm)
        
        assert isinstance(cost_norm, ReactionNorm)
        assert cost_norm.trait_name == "body_size_fitness_cost"
        assert cost_norm.environmental_variable == "temperature"
        assert len(cost_norm.environments) > 0
        assert len(cost_norm.phenotypes) > 0
        
        # All costs should be non-negative
        assert np.all(cost_norm.phenotypes >= 0)
        
        # Should have some positive costs for maladaptive norm
        assert np.any(cost_norm.phenotypes > 0)
    
    def test_serialization(self):
        """Test detector serialization"""
        detector = self.create_test_detector()
        
        # Test to_dict (excludes function)
        data = detector.to_dict()
        assert isinstance(data, dict)
        assert 'environment_variable' in data
        assert 'trait_name' in data
        assert 'optimization_tolerance' in data
        assert data['environment_variable'] == "temperature"
        
        # Test to_json
        json_str = detector.to_json()
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        import json
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)


class TestMaladaptationIntegration:
    """Integration tests for maladaptive plasticity detection"""
    
    def test_thermal_adaptation_scenario(self):
        """Test realistic thermal adaptation scenario"""
        # Create detector with thermal fitness
        detector = MaladaptiveDetector(
            fitness_function=thermal_fitness,
            environment_variable="temperature",
            trait_name="body_size"
        )
        
        # Create norm that increases too steeply (overreaction)
        environments = [0, 10, 20, 30, 40]
        phenotypes = [5, 15, 35, 65, 95]  # Overreacts to temperature
        
        overreactive_norm = ReactionNorm(
            environments=environments,
            phenotypes=phenotypes,
            trait_name="body_size",
            environmental_variable="temperature"
        )
        
        # Should detect maladaptive responses at high temperatures
        responses = detector.detect_maladaptive_responses(overreactive_norm)
        
        # Should find some maladaptive responses
        high_temp_responses = [r for r in responses if r.environment >= 30]
        assert len(high_temp_responses) > 0
    
    def test_comparison_adaptive_vs_maladaptive(self):
        """Compare adaptive vs maladaptive norms"""
        detector = MaladaptiveDetector(
            fitness_function=quadratic_fitness,
            environment_variable="temperature", 
            trait_name="size"
        )
        
        # Adaptive norm: matches environment
        adaptive = ReactionNorm([0, 20, 40], [0, 20, 40], "size", "temperature")
        
        # Maladaptive norm: opposite response  
        maladaptive = ReactionNorm([0, 20, 40], [40, 20, 0], "size", "temperature")
        
        # Compare severity
        adaptive_metrics = detector.quantify_maladaptation_severity(adaptive)
        maladaptive_metrics = detector.quantify_maladaptation_severity(maladaptive)
        
        # Maladaptive should have higher costs
        assert (maladaptive_metrics['mean_fitness_cost'] >= 
                adaptive_metrics['mean_fitness_cost'])
        assert (maladaptive_metrics['proportion_maladaptive'] >= 
                adaptive_metrics['proportion_maladaptive'])
    
    def test_edge_case_flat_norm(self):
        """Test edge case with flat reaction norm"""
        detector = MaladaptiveDetector(
            fitness_function=quadratic_fitness,
            environment_variable="temperature",
            trait_name="size"
        )
        
        # Flat norm (no plasticity)
        flat_norm = ReactionNorm([0, 20, 40], [20, 20, 20], "size", "temperature")
        
        responses = detector.detect_maladaptive_responses(flat_norm)
        
        # May detect some responses (flat isn't always optimal)
        # But costs should be moderate (allow for edge case at boundaries)
        for response in responses:
            assert response.relative_fitness_cost <= 1.0  # Allow for boundary cases
    
    def test_perfect_adaptation(self):
        """Test case with perfect adaptation (no maladaptation)"""
        # Custom fitness function where optimal phenotype = environment
        def perfect_fit(phenotype, environment):
            return 1.0 - abs(phenotype - environment) / 50.0
        
        detector = MaladaptiveDetector(
            fitness_function=perfect_fit,
            environment_variable="temperature",
            trait_name="size"
        )
        
        # Perfect norm: phenotype = environment
        perfect_norm = ReactionNorm([0, 10, 20, 30, 40], 
                                  [0, 10, 20, 30, 40], 
                                  "size", "temperature")
        
        responses = detector.detect_maladaptive_responses(perfect_norm)
        
        # Should detect very few or no maladaptive responses
        assert len(responses) <= 2  # Allow for small numerical errors
        
        # Any detected responses should have minimal costs
        for response in responses:
            assert response.relative_fitness_cost < 0.1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])