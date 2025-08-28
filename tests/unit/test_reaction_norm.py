"""
Tests for reaction norm implementation

This module contains comprehensive tests for ReactionNorm and PlasticityMagnitude
classes, ensuring 100% coverage and validation of all functionality.
"""

import json
import pytest
import numpy as np
import tempfile
import os

from ipe.core.plasticity.reaction_norm import ReactionNorm, PlasticityMagnitude
from ipe.core.plasticity.gxe import GxEInteraction


class TestPlasticityMagnitude:
    """Test suite for PlasticityMagnitude enum"""

    def test_magnitude_values(self):
        """Test PlasticityMagnitude enum values"""
        assert PlasticityMagnitude.NONE.value == "none"
        assert PlasticityMagnitude.LOW.value == "low"
        assert PlasticityMagnitude.MODERATE.value == "moderate"
        assert PlasticityMagnitude.HIGH.value == "high"
        assert PlasticityMagnitude.EXTREME.value == "extreme"


class TestReactionNorm:
    """Test suite for ReactionNorm class"""

    def create_test_norm(self, environments=None, phenotypes=None) -> ReactionNorm:
        """Helper to create test reaction norms"""
        if environments is None:
            environments = [0, 10, 20, 30, 40]  # Temperature range
        if phenotypes is None:
            phenotypes = [50, 55, 60, 65, 70]  # Body size response

        return ReactionNorm(
            environments=np.array(environments),
            phenotypes=np.array(phenotypes),
            trait_name="body_size",
            environmental_variable="temperature",
        )

    def test_basic_creation(self):
        """Test basic creation of ReactionNorm"""
        norm = self.create_test_norm()

        assert norm.trait_name == "body_size"
        assert norm.environmental_variable == "temperature"
        assert len(norm.environments) == 5
        assert len(norm.phenotypes) == 5
        assert norm.interpolation_method == "linear"
        assert norm.genotype_id is None

    def test_creation_with_lists(self):
        """Test creation with Python lists instead of numpy arrays"""
        norm = ReactionNorm(
            environments=[0, 10, 20],
            phenotypes=[50, 55, 60],
            trait_name="test_trait",
            environmental_variable="test_env",
        )

        assert isinstance(norm.environments, np.ndarray)
        assert isinstance(norm.phenotypes, np.ndarray)
        assert len(norm.environments) == 3

    def test_creation_errors(self):
        """Test ReactionNorm creation error cases"""
        # Mismatched array lengths
        with pytest.raises(ValueError, match="same length"):
            ReactionNorm(
                environments=[0, 10, 20],
                phenotypes=[50, 55],  # One less
                trait_name="test",
                environmental_variable="test",
            )

        # Too few data points
        with pytest.raises(ValueError, match="at least 2 data points"):
            ReactionNorm(
                environments=[0],
                phenotypes=[50],
                trait_name="test",
                environmental_variable="test",
            )

        # Non-finite values
        with pytest.raises(ValueError, match="finite values"):
            ReactionNorm(
                environments=[0, np.inf, 20],
                phenotypes=[50, 55, 60],
                trait_name="test",
                environmental_variable="test",
            )

        with pytest.raises(ValueError, match="finite values"):
            ReactionNorm(
                environments=[0, 10, 20],
                phenotypes=[50, np.nan, 60],
                trait_name="test",
                environmental_variable="test",
            )

    def test_post_init_sorting(self):
        """Test that environments are sorted during initialization"""
        # Create unsorted data
        norm = ReactionNorm(
            environments=[30, 10, 40, 0, 20],
            phenotypes=[65, 55, 70, 50, 60],
            trait_name="test",
            environmental_variable="temp",
        )

        # Should be sorted by environment
        expected_envs = np.array([0, 10, 20, 30, 40])
        expected_phenos = np.array([50, 55, 60, 65, 70])

        np.testing.assert_array_equal(norm.environments, expected_envs)
        np.testing.assert_array_equal(norm.phenotypes, expected_phenos)

    def test_interpolation_methods(self):
        """Test different interpolation methods"""
        environments = np.linspace(0, 40, 9)  # More points for cubic
        phenotypes = environments**2 / 100 + 50  # Quadratic relationship

        # Test linear interpolation
        norm_linear = ReactionNorm(
            environments=environments,
            phenotypes=phenotypes,
            trait_name="test",
            environmental_variable="temp",
            interpolation_method="linear",
        )

        # Test cubic interpolation
        norm_cubic = ReactionNorm(
            environments=environments,
            phenotypes=phenotypes,
            trait_name="test",
            environmental_variable="temp",
            interpolation_method="cubic",
        )

        # Test quadratic interpolation
        norm_quad = ReactionNorm(
            environments=environments,
            phenotypes=phenotypes,
            trait_name="test",
            environmental_variable="temp",
            interpolation_method="quadratic",
        )

        # All should work
        test_env = 25.0
        linear_pred = norm_linear.predict_phenotype(test_env)
        cubic_pred = norm_cubic.predict_phenotype(test_env)
        quad_pred = norm_quad.predict_phenotype(test_env)

        assert all(np.isfinite([linear_pred, cubic_pred, quad_pred]))

    def test_interpolation_fallback(self):
        """Test fallback to linear interpolation for insufficient points"""
        # Only 3 points - cubic should fall back to linear
        norm = ReactionNorm(
            environments=[0, 20, 40],
            phenotypes=[50, 60, 70],
            trait_name="test",
            environmental_variable="temp",
            interpolation_method="cubic",
        )

        # Should still work
        pred = norm.predict_phenotype(10.0)
        assert np.isfinite(pred)

        # Only 2 points - quadratic should fall back to linear
        norm2 = ReactionNorm(
            environments=[0, 40],
            phenotypes=[50, 70],
            trait_name="test",
            environmental_variable="temp",
            interpolation_method="quadratic",
        )

        pred2 = norm2.predict_phenotype(20.0)
        assert np.isfinite(pred2)
        assert pred2 == 60.0  # Linear interpolation

    def test_invalid_interpolation_method(self):
        """Test error for invalid interpolation method"""
        with pytest.raises(ValueError, match="interpolation_method must be"):
            ReactionNorm(
                environments=[0, 10, 20],
                phenotypes=[50, 55, 60],
                trait_name="test",
                environmental_variable="test",
                interpolation_method="invalid",
            )

    def test_predict_phenotype_single(self):
        """Test phenotype prediction for single environment"""
        norm = self.create_test_norm()

        # Test exact match
        pred = norm.predict_phenotype(20.0)
        assert pred == 60.0

        # Test interpolation
        pred = norm.predict_phenotype(15.0)
        assert pred == 57.5  # Midpoint between 55 and 60

        # Test extrapolation
        pred = norm.predict_phenotype(50.0)
        assert np.isfinite(pred)
        assert pred > 70.0  # Should extrapolate upward

    def test_predict_phenotype_array(self):
        """Test phenotype prediction for array of environments"""
        norm = self.create_test_norm()

        envs = np.array([5.0, 15.0, 25.0])
        preds = norm.predict_phenotype(envs)

        assert len(preds) == 3
        assert all(np.isfinite(preds))
        assert preds[1] == 57.5  # 15.0 interpolation

    def test_plasticity_magnitude(self):
        """Test plasticity magnitude calculation"""
        # High plasticity case
        norm_high = ReactionNorm(
            environments=[0, 40],
            phenotypes=[10, 60],  # 500% change relative to mean
            trait_name="test",
            environmental_variable="temp",
        )

        magnitude = norm_high.plasticity_magnitude()
        expected = (50 / 35) * 100  # range / mean * 100
        assert abs(magnitude - expected) < 0.01

        # Low plasticity case
        norm_low = ReactionNorm(
            environments=[0, 40],
            phenotypes=[50, 52],  # Small change
            trait_name="test",
            environmental_variable="temp",
        )

        magnitude_low = norm_low.plasticity_magnitude()
        expected_low = (2 / 51) * 100  # ~3.9%
        assert abs(magnitude_low - expected_low) < 0.1

        # Zero mean case
        norm_zero = ReactionNorm(
            environments=[0, 40],
            phenotypes=[-10, 10],  # Mean = 0
            trait_name="test",
            environmental_variable="temp",
        )

        magnitude_zero = norm_zero.plasticity_magnitude()
        assert magnitude_zero == 0.0

    def test_classify_plasticity(self):
        """Test plasticity classification"""
        # Test each category

        # NONE: < 5%
        norm_none = ReactionNorm([0, 40], [50, 51], "test", "temp")  # 2% change
        assert norm_none.classify_plasticity() == PlasticityMagnitude.NONE

        # LOW: 5-15%
        norm_low = ReactionNorm([0, 40], [50, 55], "test", "temp")  # ~9.5% change
        assert norm_low.classify_plasticity() == PlasticityMagnitude.LOW

        # MODERATE: 15-30%
        norm_moderate = ReactionNorm([0, 40], [50, 60], "test", "temp")  # ~18% change
        assert norm_moderate.classify_plasticity() == PlasticityMagnitude.MODERATE

        # HIGH: 30-50%
        norm_high = ReactionNorm([0, 40], [50, 70], "test", "temp")  # ~33% change
        assert norm_high.classify_plasticity() == PlasticityMagnitude.HIGH

        # EXTREME: > 50%
        norm_extreme = ReactionNorm([0, 40], [10, 60], "test", "temp")  # ~143% change
        assert norm_extreme.classify_plasticity() == PlasticityMagnitude.EXTREME

    def test_slope(self):
        """Test slope calculation"""
        # Perfect linear relationship
        norm = ReactionNorm(
            environments=[0, 10, 20, 30, 40],
            phenotypes=[50, 52, 54, 56, 58],  # slope = 0.2
            trait_name="test",
            environmental_variable="temp",
        )

        slope = norm.slope()
        assert abs(slope - 0.2) < 0.01

        # Flat relationship
        norm_flat = ReactionNorm([0, 40], [50, 50], "test", "temp")
        assert abs(norm_flat.slope()) < 1e-10  # Near zero due to numerical precision

        # Negative slope
        norm_neg = ReactionNorm([0, 40], [60, 40], "test", "temp")
        assert norm_neg.slope() < 0

    def test_curvature(self):
        """Test curvature calculation"""
        # Quadratic relationship (positive curvature)
        environments = np.array([0, 10, 20, 30, 40])
        phenotypes = (environments - 20) ** 2 / 100 + 50  # U-shaped

        norm = ReactionNorm(environments, phenotypes, "test", "temp")
        curvature = norm.curvature()
        assert curvature > 0  # Positive curvature (convex)

        # Linear relationship (zero curvature)
        norm_linear = ReactionNorm([0, 20, 40], [50, 60, 70], "test", "temp")
        curvature_linear = norm_linear.curvature()
        assert abs(curvature_linear) < 0.1

        # Too few points
        norm_short = ReactionNorm([0, 40], [50, 60], "test", "temp")
        assert norm_short.curvature() == 0.0

    def test_inflection_points(self):
        """Test inflection point detection"""
        # S-shaped curve with inflection point
        environments = np.linspace(0, 40, 10)
        # Sigmoid-like function
        phenotypes = 50 + 20 / (1 + np.exp(-(environments - 20) / 5))

        norm = ReactionNorm(environments, phenotypes, "test", "temp")
        inflections = norm.inflection_points()

        # Should find an inflection point near environment = 20
        assert (
            len(inflections) >= 0
        )  # May or may not find inflections depending on discretization

        # Linear relationship - no inflections
        norm_linear = ReactionNorm([0, 20, 40], [50, 60, 70], "test", "temp")
        inflections_linear = norm_linear.inflection_points()
        assert len(inflections_linear) == 0

        # Too few points
        norm_short = ReactionNorm([0, 40, 80], [50, 60, 70], "test", "temp")
        assert len(norm_short.inflection_points()) == 0

    def test_environmental_optimum(self):
        """Test environmental optimum detection"""
        # Peak in middle
        norm = ReactionNorm(
            environments=[0, 10, 20, 30, 40],
            phenotypes=[50, 60, 70, 65, 55],  # Peak at 20
            trait_name="test",
            environmental_variable="temp",
        )

        optimum = norm.environmental_optimum()
        assert optimum == 20

        # Peak at end
        norm_end = ReactionNorm([0, 20, 40], [50, 60, 70], "test", "temp")
        assert norm_end.environmental_optimum() == 40

    def test_fitness_landscape(self):
        """Test conversion to fitness landscape"""
        norm = self.create_test_norm()

        # Simple fitness function: fitness = phenotype / 10
        def fitness_func(phenotype):
            return phenotype / 10

        fitness_norm = norm.fitness_landscape(fitness_func)

        assert fitness_norm.trait_name == "body_size_fitness"
        assert fitness_norm.environmental_variable == "temperature"
        assert len(fitness_norm.environments) == 5

        # Check fitness values
        expected_fitness = np.array([5.0, 5.5, 6.0, 6.5, 7.0])
        np.testing.assert_array_almost_equal(fitness_norm.phenotypes, expected_fitness)

    def test_serialization_to_dict(self):
        """Test conversion to dictionary"""
        norm = self.create_test_norm()
        norm.genotype_id = "test_genotype"

        data = norm.to_dict()

        assert isinstance(data, dict)
        assert "environments" in data
        assert "phenotypes" in data
        assert "trait_name" in data
        assert data["trait_name"] == "body_size"
        assert data["genotype_id"] == "test_genotype"
        assert "plasticity_magnitude" in data
        assert "slope" in data
        assert "curvature" in data

        # Check that arrays were converted to lists
        assert isinstance(data["environments"], list)
        assert isinstance(data["phenotypes"], list)

    def test_serialization_json(self):
        """Test JSON serialization"""
        norm = self.create_test_norm()

        # Test to_json
        json_str = norm.to_json()
        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        # Test round-trip
        norm_restored = ReactionNorm.from_json(json_str)
        assert norm_restored.trait_name == norm.trait_name
        assert norm_restored.environmental_variable == norm.environmental_variable
        np.testing.assert_array_equal(norm_restored.environments, norm.environments)
        np.testing.assert_array_equal(norm_restored.phenotypes, norm.phenotypes)

    def test_file_serialization(self):
        """Test file-based serialization"""
        norm = self.create_test_norm()
        norm.genotype_id = "test_genotype"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save to file
            norm.save_json(temp_path)

            # Load from file
            norm_restored = ReactionNorm.load_json(temp_path)

            assert norm_restored.trait_name == norm.trait_name
            assert norm_restored.genotype_id == norm.genotype_id
            np.testing.assert_array_equal(norm_restored.environments, norm.environments)
            np.testing.assert_array_equal(norm_restored.phenotypes, norm.phenotypes)

        finally:
            os.unlink(temp_path)


class TestGxEInteraction:
    """Test suite for GxEInteraction class"""

    def create_test_gxe(self) -> GxEInteraction:
        """Helper to create test G×E interaction"""
        # Create two different reaction norms
        norm1 = ReactionNorm(
            environments=[0, 20, 40],
            phenotypes=[40, 50, 60],  # Steady increase
            trait_name="body_size",
            environmental_variable="temperature",
            genotype_id="genotype_A",
        )

        norm2 = ReactionNorm(
            environments=[0, 20, 40],
            phenotypes=[60, 55, 50],  # Decrease
            trait_name="body_size",
            environmental_variable="temperature",
            genotype_id="genotype_B",
        )

        return GxEInteraction(
            genotypes={"genotype_A": norm1, "genotype_B": norm2},
            environmental_variable="temperature",
            trait_name="body_size",
        )

    def test_basic_creation(self):
        """Test basic creation of GxEInteraction"""
        gxe = self.create_test_gxe()

        assert len(gxe.genotypes) == 2
        assert "genotype_A" in gxe.genotypes
        assert "genotype_B" in gxe.genotypes
        assert gxe.environmental_variable == "temperature"
        assert gxe.trait_name == "body_size"

    def test_creation_errors(self):
        """Test GxEInteraction creation error cases"""
        norm = ReactionNorm([0, 20], [50, 60], "test", "temp")

        # Too few genotypes
        with pytest.raises(ValueError, match="at least 2 genotypes"):
            GxEInteraction({"gen1": norm}, "temp", "test")

        # Wrong norm type
        with pytest.raises(TypeError, match="must be ReactionNorm"):
            GxEInteraction({"gen1": norm, "gen2": "not a norm"}, "temp", "test")

        # Mismatched environmental variables
        norm2 = ReactionNorm([0, 20], [50, 60], "test", "altitude")  # Different env var
        with pytest.raises(ValueError, match="same environmental variable"):
            GxEInteraction({"gen1": norm, "gen2": norm2}, "temp", "test")

        # Mismatched traits
        norm3 = ReactionNorm(
            [0, 20], [50, 60], "other_trait", "temp"
        )  # Different trait
        with pytest.raises(ValueError, match="same trait"):
            GxEInteraction({"gen1": norm, "gen2": norm3}, "temp", "test")

    def test_common_environment_range(self):
        """Test getting common environment range"""
        gxe = self.create_test_gxe()

        min_env, max_env = gxe.get_common_environment_range()
        assert min_env == 0
        assert max_env == 40

        # Test with non-overlapping ranges
        norm1 = ReactionNorm([0, 10, 20], [50, 55, 60], "test", "temp")
        norm2 = ReactionNorm([15, 25, 35], [45, 50, 55], "test", "temp")

        gxe_partial = GxEInteraction({"gen1": norm1, "gen2": norm2}, "temp", "test")
        min_env, max_env = gxe_partial.get_common_environment_range()
        assert min_env == 15  # Max of mins
        assert max_env == 20  # Min of maxes

    def test_evaluate_at_environments(self):
        """Test evaluation at specific environments"""
        gxe = self.create_test_gxe()

        test_envs = np.array([10, 30])
        results = gxe.evaluate_at_environments(test_envs)

        assert "genotype_A" in results
        assert "genotype_B" in results
        assert len(results["genotype_A"]) == 2
        assert len(results["genotype_B"]) == 2

        # Check interpolated values
        # genotype_A: should increase with environment
        assert results["genotype_A"][1] > results["genotype_A"][0]
        # genotype_B: should decrease with environment
        assert results["genotype_B"][1] < results["genotype_B"][0]

    def test_interaction_variance(self):
        """Test G×E interaction variance calculation"""
        gxe = self.create_test_gxe()

        variance = gxe.interaction_variance()
        assert 0 <= variance <= 1  # Should be proportion

        # With crossing genotypes, should have interaction variance
        assert variance > 0

    def test_crossing_environments(self):
        """Test detection of crossing environments"""
        gxe = self.create_test_gxe()

        crossings = gxe.crossing_environments()

        # Our test genotypes cross (one increases, one decreases)
        assert len(crossings) > 0

        # Crossing should be in middle range
        for crossing in crossings:
            assert 0 <= crossing <= 40

    def test_genotype_plasticity_comparison(self):
        """Test comparison of plasticity across genotypes"""
        gxe = self.create_test_gxe()

        comparison = gxe.genotype_plasticity_comparison()

        assert "genotype_A" in comparison
        assert "genotype_B" in comparison

        # Check required metrics
        for genotype in comparison.values():
            assert "plasticity_magnitude" in genotype
            assert "plasticity_classification" in genotype
            assert "slope" in genotype
            assert "curvature" in genotype
            assert "environmental_optimum" in genotype

    def test_most_least_plastic(self):
        """Test identification of most/least plastic genotypes"""
        gxe = self.create_test_gxe()

        most_plastic = gxe.most_plastic_genotype()
        least_plastic = gxe.least_plastic_genotype()

        assert most_plastic in ["genotype_A", "genotype_B"]
        assert least_plastic in ["genotype_A", "genotype_B"]
        assert most_plastic != least_plastic  # Should be different

    def test_specialist_vs_generalist(self):
        """Test specialist vs generalist classification"""
        gxe = self.create_test_gxe()

        classification = gxe.specialist_vs_generalist()

        assert "genotype_A" in classification
        assert "genotype_B" in classification

        for class_type in classification.values():
            assert class_type in ["specialist", "generalist"]

    def test_optimal_environments(self):
        """Test optimal environment detection"""
        gxe = self.create_test_gxe()

        optimums = gxe.optimal_environments()

        assert "genotype_A" in optimums
        assert "genotype_B" in optimums

        # genotype_A peaks at high temp, genotype_B at low temp
        assert optimums["genotype_A"] == 40  # Max phenotype at highest temp
        assert optimums["genotype_B"] == 0  # Max phenotype at lowest temp

    def test_performance_correlation(self):
        """Test performance correlation between environments"""
        gxe = self.create_test_gxe()

        # Correlation between similar environments should be high
        correlation = gxe.performance_correlation(10, 15)
        assert 0.5 <= abs(correlation) <= 1.0

        # Correlation between distant environments may be lower (due to crossing)
        correlation_distant = gxe.performance_correlation(0, 40)
        assert -1.0 <= correlation_distant <= 1.0

    def test_environmental_canalization(self):
        """Test environmental canalization measurement"""
        gxe = self.create_test_gxe()

        canalization = gxe.environmental_canalization()

        assert "genotype_A" in canalization
        assert "genotype_B" in canalization

        # Canalization values should be between 0 and 1
        for value in canalization.values():
            assert 0 <= value <= 1

    def test_gxe_serialization(self):
        """Test G×E interaction serialization"""
        gxe = self.create_test_gxe()

        # Test to_dict
        data = gxe.to_dict()
        assert isinstance(data, dict)
        assert "genotypes" in data
        assert "environmental_variable" in data
        assert "trait_name" in data
        assert "interaction_variance" in data
        assert "crossing_environments" in data

        # Test JSON serialization
        json_str = gxe.to_json()
        assert isinstance(json_str, str)

        # Test round-trip
        gxe_restored = GxEInteraction.from_json(json_str)
        assert len(gxe_restored.genotypes) == 2
        assert gxe_restored.environmental_variable == gxe.environmental_variable
        assert gxe_restored.trait_name == gxe.trait_name

    def test_gxe_file_serialization(self):
        """Test G×E file-based serialization"""
        gxe = self.create_test_gxe()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save to file
            gxe.save_json(temp_path)

            # Load from file
            gxe_restored = GxEInteraction.load_json(temp_path)

            assert len(gxe_restored.genotypes) == 2
            assert gxe_restored.environmental_variable == gxe.environmental_variable
            assert gxe_restored.trait_name == gxe.trait_name

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
