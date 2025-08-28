"""
Reaction Norm Representation

This module implements reaction norm data structures for modeling
phenotypic plasticity across environmental gradients.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Union
from enum import Enum
import numpy as np
from scipy import interpolate
import json


class PlasticityMagnitude(Enum):
    """Classifications for plasticity magnitude"""

    NONE = "none"  # < 5% phenotypic variation
    LOW = "low"  # 5-15% variation
    MODERATE = "moderate"  # 15-30% variation
    HIGH = "high"  # 30-50% variation
    EXTREME = "extreme"  # > 50% variation


@dataclass
class ReactionNorm:
    """
    Reaction norm representation for phenotypic plasticity.

    Models the relationship between environmental conditions and
    phenotypic expression, supporting interpolation and analysis.
    """

    environments: np.ndarray  # Environmental values (e.g., temperature, altitude)
    phenotypes: np.ndarray  # Corresponding phenotypic values
    trait_name: str  # Name of the trait being modeled
    environmental_variable: str  # Name of environmental variable
    genotype_id: Optional[str] = None  # Optional genotype identifier
    interpolation_method: str = "linear"  # 'linear', 'cubic', 'quadratic'

    def __post_init__(self):
        """Initialize and validate reaction norm data"""
        # Convert to numpy arrays if needed
        if not isinstance(self.environments, np.ndarray):
            self.environments = np.array(self.environments, dtype=float)
        if not isinstance(self.phenotypes, np.ndarray):
            self.phenotypes = np.array(self.phenotypes, dtype=float)

        # Validate arrays
        if len(self.environments) != len(self.phenotypes):
            raise ValueError("environments and phenotypes must have same length")
        if len(self.environments) < 2:
            raise ValueError("Need at least 2 data points for reaction norm")
        if not np.all(np.isfinite(self.environments)):
            raise ValueError("environments must be finite values")
        if not np.all(np.isfinite(self.phenotypes)):
            raise ValueError("phenotypes must be finite values")

        # Sort by environment for interpolation
        sort_idx = np.argsort(self.environments)
        object.__setattr__(self, "environments", self.environments[sort_idx])
        object.__setattr__(self, "phenotypes", self.phenotypes[sort_idx])

        # Initialize interpolator
        self._create_interpolator()

    def _create_interpolator(self):
        """Create scipy interpolator based on method"""
        valid_methods = ["linear", "cubic", "quadratic"]
        if self.interpolation_method not in valid_methods:
            raise ValueError(f"interpolation_method must be one of {valid_methods}")

        try:
            if self.interpolation_method == "linear":
                self.interpolator = interpolate.interp1d(
                    self.environments,
                    self.phenotypes,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            elif self.interpolation_method == "cubic":
                if len(self.environments) < 4:
                    # Fall back to linear for insufficient points
                    self.interpolator = interpolate.interp1d(
                        self.environments,
                        self.phenotypes,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                else:
                    self.interpolator = interpolate.interp1d(
                        self.environments,
                        self.phenotypes,
                        kind="cubic",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
            elif self.interpolation_method == "quadratic":
                if len(self.environments) < 3:
                    # Fall back to linear for insufficient points
                    self.interpolator = interpolate.interp1d(
                        self.environments,
                        self.phenotypes,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                else:
                    self.interpolator = interpolate.interp1d(
                        self.environments,
                        self.phenotypes,
                        kind="quadratic",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
        except Exception:
            # Fall back to linear interpolation if anything goes wrong
            self.interpolator = interpolate.interp1d(
                self.environments,
                self.phenotypes,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

    def predict_phenotype(
        self, environment: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Interpolate phenotype for given environment(s).

        Args:
            environment: Environmental value(s) to predict phenotype for

        Returns:
            Predicted phenotype value(s)
        """
        return self.interpolator(environment)

    def plasticity_magnitude(self) -> float:
        """
        Calculate range of phenotypic variation as percentage.

        Returns:
            Plasticity magnitude as percentage of mean phenotype
        """
        phenotype_range = np.ptp(self.phenotypes)  # peak-to-peak range
        mean_phenotype = np.mean(self.phenotypes)

        if mean_phenotype == 0:
            return 0.0

        return (phenotype_range / abs(mean_phenotype)) * 100

    def classify_plasticity(self) -> PlasticityMagnitude:
        """
        Classify plasticity magnitude into categories.

        Returns:
            PlasticityMagnitude enum value
        """
        magnitude = self.plasticity_magnitude()

        if magnitude < 5.0:
            return PlasticityMagnitude.NONE
        elif magnitude < 15.0:
            return PlasticityMagnitude.LOW
        elif magnitude < 30.0:
            return PlasticityMagnitude.MODERATE
        elif magnitude < 50.0:
            return PlasticityMagnitude.HIGH
        else:
            return PlasticityMagnitude.EXTREME

    def slope(self) -> float:
        """
        Calculate overall slope of reaction norm.

        Returns:
            Slope coefficient (phenotype change per environment change)
        """
        # Linear regression slope
        env_range = np.ptp(self.environments)
        if env_range == 0:
            return 0.0

        # Use numpy polyfit for linear slope
        coefficients = np.polyfit(self.environments, self.phenotypes, 1)
        return coefficients[0]

    def curvature(self) -> float:
        """
        Calculate curvature (second derivative) of reaction norm.

        Returns:
            Curvature measure (0 = linear, >0 = convex, <0 = concave)
        """
        if len(self.environments) < 3:
            return 0.0

        # Second derivative from quadratic fit
        try:
            coefficients = np.polyfit(self.environments, self.phenotypes, 2)
            return 2 * coefficients[0]  # Second derivative coefficient
        except Exception:
            return 0.0

    def inflection_points(self) -> List[float]:
        """
        Find inflection points in the reaction norm.

        Returns:
            List of environmental values at inflection points
        """
        if len(self.environments) < 4:
            return []

        # Find points where second derivative changes sign
        # Use numerical differentiation
        try:
            # First derivative
            first_deriv = np.gradient(self.phenotypes, self.environments)
            # Second derivative
            second_deriv = np.gradient(first_deriv, self.environments)

            # Find sign changes in second derivative
            inflections = []
            for i in range(len(second_deriv) - 1):
                if second_deriv[i] * second_deriv[i + 1] < 0:
                    # Interpolate to find zero crossing
                    env_interp = self.environments[i] + (
                        self.environments[i + 1] - self.environments[i]
                    ) * abs(second_deriv[i]) / (
                        abs(second_deriv[i]) + abs(second_deriv[i + 1])
                    )
                    inflections.append(env_interp)

            return inflections
        except Exception:
            return []

    def environmental_optimum(self) -> Optional[float]:
        """
        Find environmental condition that maximizes phenotype.

        Returns:
            Environmental value at phenotypic optimum (None if no clear optimum)
        """
        max_idx = np.argmax(self.phenotypes)
        return self.environments[max_idx]

    def fitness_landscape(self, fitness_function) -> "ReactionNorm":
        """
        Convert phenotype reaction norm to fitness landscape.

        Args:
            fitness_function: Function that converts phenotype to fitness

        Returns:
            New ReactionNorm with fitness values
        """
        fitness_values = np.array([fitness_function(p) for p in self.phenotypes])

        return ReactionNorm(
            environments=self.environments.copy(),
            phenotypes=fitness_values,
            trait_name=f"{self.trait_name}_fitness",
            environmental_variable=self.environmental_variable,
            genotype_id=self.genotype_id,
            interpolation_method=self.interpolation_method,
        )

    def to_dict(self) -> Dict:
        """
        Convert reaction norm to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "environments": self.environments.tolist(),
            "phenotypes": self.phenotypes.tolist(),
            "trait_name": self.trait_name,
            "environmental_variable": self.environmental_variable,
            "genotype_id": self.genotype_id,
            "interpolation_method": self.interpolation_method,
            "plasticity_magnitude": self.plasticity_magnitude(),
            "plasticity_classification": self.classify_plasticity().value,
            "slope": self.slope(),
            "curvature": self.curvature(),
        }

    def to_json(self) -> str:
        """
        Convert reaction norm to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> "ReactionNorm":
        """
        Create ReactionNorm from dictionary.

        Args:
            data: Dictionary containing reaction norm data

        Returns:
            New ReactionNorm instance
        """
        return cls(
            environments=np.array(data["environments"]),
            phenotypes=np.array(data["phenotypes"]),
            trait_name=data["trait_name"],
            environmental_variable=data["environmental_variable"],
            genotype_id=data.get("genotype_id"),
            interpolation_method=data.get("interpolation_method", "linear"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ReactionNorm":
        """
        Create ReactionNorm from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            New ReactionNorm instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_json(self, filepath: str):
        """
        Save reaction norm to JSON file.

        Args:
            filepath: Path to output file
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load_json(cls, filepath: str) -> "ReactionNorm":
        """
        Load reaction norm from JSON file.

        Args:
            filepath: Path to input file

        Returns:
            New ReactionNorm instance
        """
        with open(filepath, "r") as f:
            return cls.from_json(f.read())
