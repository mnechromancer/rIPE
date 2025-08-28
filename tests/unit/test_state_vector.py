"""
Tests for physiological state vector implementation

This module contains comprehensive tests for PhysiologicalState and StateVector
classes, ensuring 100% coverage and validation of all functionality.
"""

import json
import pytest
import numpy as np
import tempfile
import os

from ipe.core.physiology.state import PhysiologicalState, Tissue
from ipe.core.physiology.state_vector import StateVector


class TestPhysiologicalState:
    """Test suite for PhysiologicalState class"""

    def test_basic_creation(self):
        """Test basic creation of PhysiologicalState"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)

        assert state.po2 == 15.0
        assert state.temperature == 25.0
        assert state.altitude == 1000.0

        # Check defaults
        assert state.heart_mass == 8.0
        assert state.hematocrit == 45.0
        assert state.salinity is None

    def test_post_init_defaults(self):
        """Test that __post_init__ sets correct defaults"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)

        # Check mitochondrial density defaults
        assert state.mitochondrial_density is not None
        assert state.mitochondrial_density[Tissue.BRAIN] == 1.8
        assert state.mitochondrial_density[Tissue.HEART] == 1.5
        assert state.mitochondrial_density[Tissue.BROWN_FAT] == 2.0

        # Check tissue perfusion defaults
        assert state.tissue_perfusion is not None
        assert abs(sum(state.tissue_perfusion.values()) - 1.0) < 0.01
        assert state.tissue_perfusion[Tissue.BRAIN] == 0.15
        assert state.tissue_perfusion[Tissue.HEART] == 0.04

    def test_immutability(self):
        """Test that PhysiologicalState is immutable"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)

        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            state.po2 = 20.0

    def test_equality_and_hashing(self):
        """Test __eq__ and __hash__ methods work correctly"""
        state1 = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)

        state2 = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)

        state3 = PhysiologicalState(
            po2=16.0, temperature=25.0, altitude=1000.0  # Different
        )

        assert state1 == state2
        assert state1 != state3
        assert hash(state1) == hash(state2)
        assert hash(state1) != hash(state3)

        # Should be able to use in sets
        state_set = {state1, state2, state3}
        assert len(state_set) == 2  # state1 and state2 are equal

    def test_validation_success(self):
        """Test validation passes for valid parameters"""
        state = PhysiologicalState(
            po2=15.0,
            temperature=25.0,
            altitude=1000.0,
            salinity=35.0,
            heart_mass=10.0,
            hematocrit=50.0,
            hemoglobin=15.0,
            blood_volume=80.0,
            cardiac_output=200.0,
            lung_volume=60.0,
            diffusion_capacity=2.0,
            ventilation_rate=30.0,
            tidal_volume=8.0,
            bmr=50.0,
            vo2max=80.0,
            respiratory_exchange_ratio=0.8,
            thermal_conductance=2.0,
            lower_critical_temp=10.0,
            upper_critical_temp=35.0,
            max_thermogenesis=150.0,
            plasma_osmolality=300.0,
            gill_na_k_atpase=10.0,
            drinking_rate=25.0,
        )

        # Should not raise any exceptions
        state.validate()

    def test_validation_environmental_errors(self):
        """Test validation catches environmental parameter errors"""
        # Test po2 out of range
        with pytest.raises(ValueError, match="po2.*outside valid range"):
            state = PhysiologicalState(po2=25.0, temperature=25.0, altitude=1000.0)
            state.validate()

        with pytest.raises(ValueError, match="po2.*outside valid range"):
            state = PhysiologicalState(po2=3.0, temperature=25.0, altitude=1000.0)
            state.validate()

        # Test temperature out of range
        with pytest.raises(ValueError, match="temperature.*outside valid range"):
            state = PhysiologicalState(po2=15.0, temperature=60.0, altitude=1000.0)
            state.validate()

        # Test altitude out of range
        with pytest.raises(ValueError, match="altitude.*outside valid range"):
            state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=6000.0)
            state.validate()

        # Test salinity out of range
        with pytest.raises(ValueError, match="salinity.*outside valid range"):
            state = PhysiologicalState(
                po2=15.0, temperature=25.0, altitude=1000.0, salinity=40.0
            )
            state.validate()

    def test_validation_cardiovascular_errors(self):
        """Test validation catches cardiovascular parameter errors"""
        with pytest.raises(ValueError, match="heart_mass.*outside valid range"):
            state = PhysiologicalState(
                po2=15.0, temperature=25.0, altitude=1000.0, heart_mass=20.0
            )
            state.validate()

        with pytest.raises(ValueError, match="hematocrit.*outside valid range"):
            state = PhysiologicalState(
                po2=15.0, temperature=25.0, altitude=1000.0, hematocrit=80.0
            )
            state.validate()

    def test_validation_temperature_logic(self):
        """Test validation catches temperature logic errors"""
        with pytest.raises(
            ValueError, match="lower_critical_temp.*must be.*upper_critical_temp"
        ):
            state = PhysiologicalState(
                po2=15.0,
                temperature=25.0,
                altitude=1000.0,
                lower_critical_temp=25.0,  # At upper bound of range
                upper_critical_temp=25.0,  # At lower bound of range - equal not allowed
            )
            state.validate()

    def test_validation_mitochondrial_density(self):
        """Test validation of mitochondrial density"""
        with pytest.raises(
            ValueError, match="mitochondrial_density.*outside valid range"
        ):
            state = PhysiologicalState(
                po2=15.0,
                temperature=25.0,
                altitude=1000.0,
                mitochondrial_density={Tissue.BRAIN: 3.0},  # Too high
            )
            state.validate()

    def test_validation_tissue_perfusion(self):
        """Test validation of tissue perfusion"""
        # Test individual perfusion out of range first
        with pytest.raises(ValueError, match="tissue_perfusion.*outside valid range"):
            state = PhysiologicalState(
                po2=15.0,
                temperature=25.0,
                altitude=1000.0,
                tissue_perfusion={
                    # Too high - should trigger individual range error first
                    Tissue.BRAIN: 1.5,
                },
            )
            state.validate()

        # Test perfusion doesn't sum to 1
        with pytest.raises(ValueError, match="tissue_perfusion fractions sum"):
            state = PhysiologicalState(
                po2=15.0,
                temperature=25.0,
                altitude=1000.0,
                tissue_perfusion={
                    Tissue.BRAIN: 0.5,
                    Tissue.HEART: 0.6,  # All values within range but sum > 1
                },
            )
            state.validate()

    def test_compute_aerobic_scope(self):
        """Test aerobic scope calculation"""
        state = PhysiologicalState(
            po2=15.0,
            temperature=25.0,
            altitude=1000.0,
            bmr=60.0,  # mL O2/hr/kg
            vo2max=120.0,  # mL O2/min/kg
        )

        expected = 120.0 - (60.0 / 60)  # 120 - 1 = 119 mL O2/min/kg
        assert state.compute_aerobic_scope() == expected

    def test_oxygen_delivery(self):
        """Test oxygen delivery calculation"""
        state = PhysiologicalState(
            po2=21.0,  # Full atmospheric O2
            temperature=25.0,
            altitude=0.0,
            hemoglobin=15.0,  # g/dL
            cardiac_output=200.0,  # mL/min/kg
            tissue_perfusion={
                Tissue.BRAIN: 0.15,
                Tissue.HEART: 0.04,
                Tissue.MUSCLE: 0.40,
                Tissue.BROWN_FAT: 0.02,
                Tissue.KIDNEY: 0.22,
                Tissue.LUNG: 0.12,
                Tissue.GILL: 0.05,
            },
        )

        # Test brain O2 delivery
        brain_delivery = state.oxygen_delivery(Tissue.BRAIN)
        assert brain_delivery > 0

        # Test that higher perfusion tissues get more O2
        muscle_delivery = state.oxygen_delivery(Tissue.MUSCLE)
        assert muscle_delivery > brain_delivery  # Muscle has higher perfusion

        # Test error for missing tissue
        state_incomplete = PhysiologicalState(
            po2=15.0,
            temperature=25.0,
            altitude=1000.0,
            tissue_perfusion={Tissue.BRAIN: 1.0},  # Only brain
        )

        with pytest.raises(ValueError, match="not found in perfusion data"):
            state_incomplete.oxygen_delivery(Tissue.HEART)

    def test_thermal_neutral_zone_width(self):
        """Test thermal neutral zone width calculation"""
        state = PhysiologicalState(
            po2=15.0,
            temperature=25.0,
            altitude=1000.0,
            lower_critical_temp=10.0,
            upper_critical_temp=35.0,
        )

        assert state.thermal_neutral_zone_width() == 25.0

    def test_is_in_thermal_neutral_zone(self):
        """Test thermal neutral zone check"""
        state = PhysiologicalState(
            po2=15.0,
            temperature=25.0,
            altitude=1000.0,
            lower_critical_temp=10.0,
            upper_critical_temp=35.0,
        )

        # Using state's own temperature
        assert state.is_in_thermal_neutral_zone() is True

        # Using custom temperatures
        assert state.is_in_thermal_neutral_zone(15.0) is True
        assert state.is_in_thermal_neutral_zone(5.0) is False
        assert state.is_in_thermal_neutral_zone(40.0) is False


class TestStateVector:
    """Test suite for StateVector class"""

    def create_test_state(self, po2: float = 15.0) -> PhysiologicalState:
        """Helper to create test states"""
        return PhysiologicalState(po2=po2, temperature=25.0, altitude=1000.0)

    def test_basic_creation(self):
        """Test basic StateVector creation"""
        state1 = self.create_test_state(15.0)
        state2 = self.create_test_state(16.0)

        # Single state
        vector1 = StateVector(state1)
        assert len(vector1) == 1
        assert vector1[0] == state1

        # Multiple states
        vector2 = StateVector([state1, state2])
        assert len(vector2) == 2
        assert vector2[0] == state1
        assert vector2[1] == state2

    def test_creation_errors(self):
        """Test StateVector creation error cases"""
        # Empty list
        with pytest.raises(ValueError, match="must contain at least one state"):
            StateVector([])

        # Non-state objects
        with pytest.raises(TypeError, match="not a PhysiologicalState"):
            StateVector([self.create_test_state(), "not a state"])

    def test_iteration(self):
        """Test StateVector iteration"""
        states = [self.create_test_state(15.0), self.create_test_state(16.0)]
        vector = StateVector(states)

        collected = list(vector)
        assert collected == states

    def test_euclidean_distance_single(self):
        """Test Euclidean distance between single states"""
        state1 = self.create_test_state(15.0)
        state2 = self.create_test_state(16.0)

        vector1 = StateVector(state1)
        vector2 = StateVector(state2)

        distance = vector1.euclidean_distance(vector2)
        assert isinstance(distance, float)
        assert distance > 0

        # Distance to self should be 0
        self_distance = vector1.euclidean_distance(vector1)
        assert self_distance == 0.0

    def test_euclidean_distance_multiple(self):
        """Test Euclidean distance with multiple states"""
        states1 = [self.create_test_state(15.0), self.create_test_state(16.0)]
        states2 = [self.create_test_state(17.0), self.create_test_state(18.0)]

        vector1 = StateVector(states1)
        vector2 = StateVector(states2)

        distance_matrix = vector1.euclidean_distance(vector2)
        assert distance_matrix.shape == (2, 2)
        assert np.all(distance_matrix > 0)

    def test_manhattan_distance(self):
        """Test Manhattan distance calculation"""
        state1 = self.create_test_state(15.0)
        state2 = self.create_test_state(16.0)

        vector1 = StateVector(state1)

        manhattan_dist = vector1.manhattan_distance(state2)
        euclidean_dist = vector1.euclidean_distance(state2)

        assert isinstance(manhattan_dist, float)
        assert manhattan_dist > 0
        # Manhattan distance should be >= Euclidean distance
        assert manhattan_dist >= euclidean_dist

    def test_mahalanobis_distance(self):
        """Test Mahalanobis distance calculation"""
        # Need multiple states to compute covariance
        states = [
            self.create_test_state(15.0),
            self.create_test_state(16.0),
            self.create_test_state(17.0),
        ]
        vector = StateVector(states)
        test_state = self.create_test_state(18.0)

        distances = vector.mahalanobis_distance(test_state)
        assert len(distances) == 3
        assert np.all(distances > 0)

        # Test with insufficient states for covariance
        single_vector = StateVector(self.create_test_state(15.0))
        with pytest.raises(ValueError, match="Need at least 2 states"):
            single_vector.mahalanobis_distance(test_state)

    def test_serialization_to_dict(self):
        """Test conversion to dictionary"""
        state = self.create_test_state()
        vector = StateVector(state)

        dict_list = vector.to_dict()
        assert len(dict_list) == 1
        assert isinstance(dict_list[0], dict)

        # Check key fields are present
        state_dict = dict_list[0]
        assert "po2" in state_dict
        assert "temperature" in state_dict
        assert "mitochondrial_density" in state_dict
        assert "tissue_perfusion" in state_dict

        # Check that tissue enums were converted to strings
        assert isinstance(list(state_dict["mitochondrial_density"].keys())[0], str)

    def test_serialization_json(self):
        """Test JSON serialization"""
        state = self.create_test_state()
        vector = StateVector(state)

        # Test to_json
        json_str = vector.to_json()
        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

        # Test round-trip
        vector_restored = StateVector.from_json(json_str)
        assert len(vector_restored) == 1
        assert vector_restored[0] == state

    def test_file_serialization(self):
        """Test file-based serialization"""
        states = [self.create_test_state(15.0), self.create_test_state(16.0)]
        vector = StateVector(states)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save to file
            vector.save_json(temp_path)

            # Load from file
            vector_restored = StateVector.load_json(temp_path)

            assert len(vector_restored) == 2
            assert vector_restored[0] == states[0]
            assert vector_restored[1] == states[1]

        finally:
            os.unlink(temp_path)

    def test_mean_state(self):
        """Test mean state calculation"""
        # Single state - should return the state itself
        state1 = self.create_test_state(15.0)
        vector1 = StateVector(state1)
        mean1 = vector1.mean_state()
        assert mean1 == state1

        # Multiple states - should return mean
        state2 = self.create_test_state(17.0)  # po2 = 17
        vector2 = StateVector([state1, state2])
        mean2 = vector2.mean_state()

        # Mean po2 should be 16.0
        assert mean2.po2 == 16.0
        assert mean2.temperature == 25.0  # Same for both

    def test_state_vector_conversion(self):
        """Test internal state-to-vector conversion"""
        state = PhysiologicalState(
            po2=15.0,
            temperature=25.0,
            altitude=1000.0,
            salinity=30.0,
            plasma_osmolality=300.0,
        )

        vector = StateVector(state)
        numerical_matrix = vector._to_numerical_matrix()

        assert numerical_matrix.shape == (1, 37)  # 1 state, 37 features
        assert numerical_matrix[0, 0] == 15.0  # po2
        assert numerical_matrix[0, 1] == 25.0  # temperature
        assert numerical_matrix[0, 2] == 1000.0  # altitude

        # Test vector-to-state conversion
        reconstructed = vector._vector_to_state(numerical_matrix[0])
        assert reconstructed.po2 == state.po2
        assert reconstructed.temperature == state.temperature
        assert reconstructed.salinity == state.salinity
        assert reconstructed.plasma_osmolality == state.plasma_osmolality


class TestTissue:
    """Test suite for Tissue enum"""

    def test_tissue_values(self):
        """Test Tissue enum values"""
        assert Tissue.BRAIN.value == "brain"
        assert Tissue.HEART.value == "heart"
        assert Tissue.MUSCLE.value == "muscle"
        assert Tissue.BROWN_FAT.value == "brown_fat"
        assert Tissue.KIDNEY.value == "kidney"
        assert Tissue.LUNG.value == "lung"
        assert Tissue.GILL.value == "gill"

    def test_tissue_string_conversion(self):
        """Test conversion from string to Tissue"""
        assert Tissue("brain") == Tissue.BRAIN
        assert Tissue("heart") == Tissue.HEART


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
