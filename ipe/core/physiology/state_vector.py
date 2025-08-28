"""
Physiological State Vector Operations

This module provides the StateVector class for working with collections
of PhysiologicalState objects, including distance calculations and
serialization/deserialization capabilities.
"""

import json
import numpy as np
from typing import List, Dict, Any, Union, Optional
from dataclasses import asdict
from .state import PhysiologicalState, Tissue


class StateVector:
    """
    A collection of PhysiologicalState objects with vector operations.

    This class provides utilities for working with collections of physiological
    states, including distance calculations, serialization, and statistical operations.
    """

    def __init__(self, states: Union[List[PhysiologicalState], PhysiologicalState]):
        """
        Initialize StateVector with one or more PhysiologicalState objects.

        Args:
            states: Single state or list of states
        """
        if isinstance(states, PhysiologicalState):
            self.states = [states]
        else:
            self.states = list(states)

        if not self.states:
            raise ValueError("StateVector must contain at least one state")

        # Validate all states
        for i, state in enumerate(self.states):
            if not isinstance(state, PhysiologicalState):
                raise TypeError(f"Item {i} is not a PhysiologicalState")
            state.validate()

    def __len__(self) -> int:
        """Return number of states in vector"""
        return len(self.states)

    def __getitem__(self, index: int) -> PhysiologicalState:
        """Get state at index"""
        return self.states[index]

    def __iter__(self):
        """Iterate over states"""
        return iter(self.states)

    def euclidean_distance(
        self, other: Union["StateVector", PhysiologicalState]
    ) -> np.ndarray:
        """
        Calculate Euclidean distance between states.

        Args:
            other: Another StateVector or single PhysiologicalState

        Returns:
            np.ndarray: Distance matrix or distance vector
        """
        if isinstance(other, PhysiologicalState):
            other = StateVector(other)

        # Get numerical vectors for all states
        self_vectors = self._to_numerical_matrix()
        other_vectors = other._to_numerical_matrix()

        # Calculate pairwise distances
        distances = np.zeros((len(self), len(other)))
        for i in range(len(self)):
            for j in range(len(other)):
                diff = self_vectors[i] - other_vectors[j]
                distances[i, j] = np.sqrt(np.sum(diff**2))

        # Return scalar if both are single states, otherwise return matrix/vector
        if len(self) == 1 and len(other) == 1:
            return distances[0, 0]
        elif len(self) == 1:
            return distances[0, :]
        elif len(other) == 1:
            return distances[:, 0]
        else:
            return distances

    def manhattan_distance(
        self, other: Union["StateVector", PhysiologicalState]
    ) -> np.ndarray:
        """
        Calculate Manhattan (L1) distance between states.

        Args:
            other: Another StateVector or single PhysiologicalState

        Returns:
            np.ndarray: Distance matrix or distance vector
        """
        if isinstance(other, PhysiologicalState):
            other = StateVector(other)

        # Get numerical vectors for all states
        self_vectors = self._to_numerical_matrix()
        other_vectors = other._to_numerical_matrix()

        # Calculate pairwise distances
        distances = np.zeros((len(self), len(other)))
        for i in range(len(self)):
            for j in range(len(other)):
                diff = np.abs(self_vectors[i] - other_vectors[j])
                distances[i, j] = np.sum(diff)

        # Return scalar if both are single states, otherwise return matrix/vector
        if len(self) == 1 and len(other) == 1:
            return distances[0, 0]
        elif len(self) == 1:
            return distances[0, :]
        elif len(other) == 1:
            return distances[:, 0]
        else:
            return distances

    def mahalanobis_distance(
        self,
        other: Union["StateVector", PhysiologicalState],
        cov_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate Mahalanobis distance between states.

        Args:
            other: Another StateVector or single PhysiologicalState
            cov_matrix: Covariance matrix (computed from self if None)

        Returns:
            np.ndarray: Distance matrix or distance vector
        """
        if isinstance(other, PhysiologicalState):
            other = StateVector(other)

        # Get numerical vectors
        self_vectors = self._to_numerical_matrix()
        other_vectors = other._to_numerical_matrix()

        # Calculate or use provided covariance matrix
        if cov_matrix is None:
            if len(self) < 2:
                raise ValueError("Need at least 2 states to compute covariance matrix")
            cov_matrix = np.cov(self_vectors, rowvar=False)

        # Handle singular covariance matrix
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            inv_cov = np.linalg.pinv(cov_matrix)

        # Calculate pairwise distances
        distances = np.zeros((len(self), len(other)))
        for i in range(len(self)):
            for j in range(len(other)):
                diff = self_vectors[i] - other_vectors[j]
                distances[i, j] = np.sqrt(diff.T @ inv_cov @ diff)

        # Return scalar if both are single states, otherwise return matrix/vector
        if len(self) == 1 and len(other) == 1:
            return distances[0, 0]
        elif len(self) == 1:
            return distances[0, :]
        elif len(other) == 1:
            return distances[:, 0]
        else:
            return distances

    def _to_numerical_matrix(self) -> np.ndarray:
        """
        Convert states to numerical matrix for distance calculations.

        Returns:
            np.ndarray: Matrix where each row is a state vector
        """
        vectors = []
        for state in self.states:
            vector = self._state_to_vector(state)
            vectors.append(vector)
        return np.array(vectors)

    def _state_to_vector(self, state: PhysiologicalState) -> np.ndarray:
        """
        Convert single PhysiologicalState to numerical vector.

        Args:
            state: PhysiologicalState to convert

        Returns:
            np.ndarray: Numerical vector representation
        """
        vector_parts = []

        # Basic numerical fields
        numeric_fields = [
            "po2",
            "temperature",
            "altitude",
            "heart_mass",
            "hematocrit",
            "hemoglobin",
            "blood_volume",
            "cardiac_output",
            "lung_volume",
            "diffusion_capacity",
            "ventilation_rate",
            "tidal_volume",
            "bmr",
            "vo2max",
            "respiratory_exchange_ratio",
            "thermal_conductance",
            "lower_critical_temp",
            "upper_critical_temp",
            "max_thermogenesis",
        ]

        for field in numeric_fields:
            value = getattr(state, field)
            vector_parts.append(value)

        # Optional numerical fields (use 0 if None)
        optional_fields = [
            "salinity",
            "plasma_osmolality",
            "gill_na_k_atpase",
            "drinking_rate",
        ]
        for field in optional_fields:
            value = getattr(state, field)
            vector_parts.append(value if value is not None else 0.0)

        # Mitochondrial density (convert dict to fixed-order vector)
        tissue_order = [
            Tissue.BRAIN,
            Tissue.HEART,
            Tissue.MUSCLE,
            Tissue.BROWN_FAT,
            Tissue.KIDNEY,
            Tissue.LUNG,
            Tissue.GILL,
        ]
        for tissue in tissue_order:
            density = state.mitochondrial_density.get(tissue, 1.0)
            vector_parts.append(density)

        # Tissue perfusion (convert dict to fixed-order vector)
        for tissue in tissue_order:
            perfusion = state.tissue_perfusion.get(tissue, 0.0)
            vector_parts.append(perfusion)

        return np.array(vector_parts)

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Convert StateVector to list of dictionaries for serialization.

        Returns:
            List[Dict[str, Any]]: List of state dictionaries
        """
        return [self._state_to_dict(state) for state in self.states]

    def _state_to_dict(self, state: PhysiologicalState) -> Dict[str, Any]:
        """
        Convert single PhysiologicalState to dictionary.

        Args:
            state: PhysiologicalState to convert

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        # Use dataclasses.asdict but handle Enums properly
        state_dict = asdict(state)

        # Convert Tissue enums to strings in dictionaries
        if (
            "mitochondrial_density" in state_dict
            and state_dict["mitochondrial_density"]
        ):
            state_dict["mitochondrial_density"] = {
                tissue.value: density
                for tissue, density in state.mitochondrial_density.items()
            }

        if "tissue_perfusion" in state_dict and state_dict["tissue_perfusion"]:
            state_dict["tissue_perfusion"] = {
                tissue.value: perfusion
                for tissue, perfusion in state.tissue_perfusion.items()
            }

        return state_dict

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Serialize StateVector to JSON string.

        Args:
            indent: JSON indentation level (None for compact)

        Returns:
            str: JSON representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, filepath: str, indent: Optional[int] = 2) -> None:
        """
        Save StateVector to JSON file.

        Args:
            filepath: Path to output file
            indent: JSON indentation level
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_dict(cls, data: List[Dict[str, Any]]) -> "StateVector":
        """
        Create StateVector from list of dictionaries.

        Args:
            data: List of state dictionaries

        Returns:
            StateVector: New StateVector instance
        """
        states = []
        for state_dict in data:
            state = cls._state_from_dict(state_dict)
            states.append(state)
        return cls(states)

    @classmethod
    def _state_from_dict(cls, state_dict: Dict[str, Any]) -> PhysiologicalState:
        """
        Create PhysiologicalState from dictionary.

        Args:
            state_dict: Dictionary representation of state

        Returns:
            PhysiologicalState: New state instance
        """
        # Make a copy to avoid modifying original
        data = state_dict.copy()

        # Convert tissue string keys back to Tissue enums
        if "mitochondrial_density" in data and data["mitochondrial_density"]:
            data["mitochondrial_density"] = {
                Tissue(tissue_str): density
                for tissue_str, density in data["mitochondrial_density"].items()
            }

        if "tissue_perfusion" in data and data["tissue_perfusion"]:
            data["tissue_perfusion"] = {
                Tissue(tissue_str): perfusion
                for tissue_str, perfusion in data["tissue_perfusion"].items()
            }

        return PhysiologicalState(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "StateVector":
        """
        Create StateVector from JSON string.

        Args:
            json_str: JSON representation

        Returns:
            StateVector: New StateVector instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load_json(cls, filepath: str) -> "StateVector":
        """
        Load StateVector from JSON file.

        Args:
            filepath: Path to input file

        Returns:
            StateVector: New StateVector instance
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def mean_state(self) -> PhysiologicalState:
        """
        Calculate mean PhysiologicalState across all states in vector.

        Returns:
            PhysiologicalState: State with mean values
        """
        if len(self.states) == 1:
            return self.states[0]

        # Get all numerical values
        numerical_matrix = self._to_numerical_matrix()
        mean_vector = np.mean(numerical_matrix, axis=0)

        # Reconstruct state from mean vector
        return self._vector_to_state(mean_vector)

    def _vector_to_state(self, vector: np.ndarray) -> PhysiologicalState:
        """
        Convert numerical vector back to PhysiologicalState.

        Args:
            vector: Numerical vector

        Returns:
            PhysiologicalState: Reconstructed state
        """
        i = 0

        # Basic numerical fields
        po2 = vector[i]
        i += 1
        temperature = vector[i]
        i += 1
        altitude = vector[i]
        i += 1
        heart_mass = vector[i]
        i += 1
        hematocrit = vector[i]
        i += 1
        hemoglobin = vector[i]
        i += 1
        blood_volume = vector[i]
        i += 1
        cardiac_output = vector[i]
        i += 1
        lung_volume = vector[i]
        i += 1
        diffusion_capacity = vector[i]
        i += 1
        ventilation_rate = vector[i]
        i += 1
        tidal_volume = vector[i]
        i += 1
        bmr = vector[i]
        i += 1
        vo2max = vector[i]
        i += 1
        respiratory_exchange_ratio = vector[i]
        i += 1
        thermal_conductance = vector[i]
        i += 1
        lower_critical_temp = vector[i]
        i += 1
        upper_critical_temp = vector[i]
        i += 1
        max_thermogenesis = vector[i]
        i += 1

        # Optional fields
        salinity = vector[i] if vector[i] != 0 else None
        i += 1
        plasma_osmolality = vector[i] if vector[i] != 0 else None
        i += 1
        gill_na_k_atpase = vector[i] if vector[i] != 0 else None
        i += 1
        drinking_rate = vector[i] if vector[i] != 0 else None
        i += 1

        # Mitochondrial density
        tissue_order = [
            Tissue.BRAIN,
            Tissue.HEART,
            Tissue.MUSCLE,
            Tissue.BROWN_FAT,
            Tissue.KIDNEY,
            Tissue.LUNG,
            Tissue.GILL,
        ]
        mitochondrial_density = {}
        for tissue in tissue_order:
            mitochondrial_density[tissue] = vector[i]
            i += 1

        # Tissue perfusion
        tissue_perfusion = {}
        for tissue in tissue_order:
            tissue_perfusion[tissue] = vector[i]
            i += 1

        return PhysiologicalState(
            po2=po2,
            temperature=temperature,
            altitude=altitude,
            salinity=salinity,
            heart_mass=heart_mass,
            hematocrit=hematocrit,
            hemoglobin=hemoglobin,
            blood_volume=blood_volume,
            cardiac_output=cardiac_output,
            lung_volume=lung_volume,
            diffusion_capacity=diffusion_capacity,
            ventilation_rate=ventilation_rate,
            tidal_volume=tidal_volume,
            bmr=bmr,
            vo2max=vo2max,
            respiratory_exchange_ratio=respiratory_exchange_ratio,
            mitochondrial_density=mitochondrial_density,
            thermal_conductance=thermal_conductance,
            lower_critical_temp=lower_critical_temp,
            upper_critical_temp=upper_critical_temp,
            max_thermogenesis=max_thermogenesis,
            tissue_perfusion=tissue_perfusion,
            plasma_osmolality=plasma_osmolality,
            gill_na_k_atpase=gill_na_k_atpase,
            drinking_rate=drinking_rate,
        )
