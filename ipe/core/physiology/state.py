"""
Physiological State Representation

This module defines the core PhysiologicalState dataclass that represents
the complete physiological state of an organism in the IPE simulation.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import numpy as np


class Tissue(Enum):
    """Enumeration of tissue types for physiological modeling"""

    HEART = "heart"
    LUNG = "lung"
    MUSCLE = "muscle"
    BROWN_FAT = "brown_fat"
    BRAIN = "brain"
    KIDNEY = "kidney"
    GILL = "gill"  # For fish


@dataclass(frozen=True)
class PhysiologicalState:
    """
    Complete physiological state of an organism.

    This immutable dataclass represents all physiological parameters
    needed for IPE simulations, including environmental conditions
    and organ system characteristics.
    """

    # Environmental conditions
    po2: float  # Partial pressure O2 (kPa), range: 5-21
    temperature: float  # °C, range: -40 to 50
    altitude: float  # meters, range: 0-5000
    salinity: Optional[float] = None  # ppt for aquatic, range: 0-35

    # Cardiovascular parameters
    heart_mass: float = 8.0  # g/kg body mass, range: 3-15
    hematocrit: float = 45.0  # %, range: 20-70
    hemoglobin: float = 15.0  # g/dL, range: 10-25
    blood_volume: float = 80.0  # mL/kg, range: 50-120
    cardiac_output: float = 200.0  # mL/min/kg, range: 100-500

    # Respiratory parameters
    lung_volume: float = 60.0  # mL/kg, range: 40-100
    diffusion_capacity: float = 2.0  # mL O2/min/mmHg/kg, range: 1-5
    ventilation_rate: float = 30.0  # breaths/min, range: 10-100
    tidal_volume: float = 8.0  # mL/kg, range: 5-15

    # Metabolic parameters
    bmr: float = 50.0  # mL O2/hr/kg, range: 20-200
    vo2max: float = 80.0  # mL O2/min/kg, range: 30-200
    respiratory_exchange_ratio: float = 0.8  # unitless, range: 0.7-1.0
    mitochondrial_density: Optional[Dict[Tissue, float]] = (
        None  # relative, range: 0.5-2.0
    )

    # Thermoregulation parameters
    thermal_conductance: float = 2.0  # mL O2/hr/°C, range: 1-5
    lower_critical_temp: float = 10.0  # °C, range: -10 to 25
    upper_critical_temp: float = 35.0  # °C, range: 25 to 45
    max_thermogenesis: float = 150.0  # mL O2/hr/kg, range: 50-300

    # Tissue perfusion fractions (sum should equal 1.0)
    tissue_perfusion: Optional[Dict[Tissue, float]] = None  # fraction, range: 0-1

    # Osmoregulation parameters (for aquatic organisms)
    plasma_osmolality: Optional[float] = None  # mOsm/kg, range: 250-400
    gill_na_k_atpase: Optional[float] = None  # μmol ADP/mg protein/h, range: 1-20
    drinking_rate: Optional[float] = None  # mL/hr/kg, range: 0-50

    def __post_init__(self):
        """Initialize default values for optional dict fields"""
        if self.mitochondrial_density is None:
            # Default mitochondrial densities
            object.__setattr__(
                self,
                "mitochondrial_density",
                {
                    Tissue.HEART: 1.5,
                    Tissue.MUSCLE: 1.0,
                    Tissue.BRAIN: 1.8,
                    Tissue.BROWN_FAT: 2.0,
                    Tissue.KIDNEY: 1.3,
                    Tissue.LUNG: 0.8,
                    Tissue.GILL: 1.2,
                },
            )

        if self.tissue_perfusion is None:
            # Default tissue perfusion fractions
            object.__setattr__(
                self,
                "tissue_perfusion",
                {
                    Tissue.BRAIN: 0.15,
                    Tissue.HEART: 0.04,
                    Tissue.KIDNEY: 0.22,
                    Tissue.MUSCLE: 0.40,
                    Tissue.LUNG: 0.12,
                    Tissue.BROWN_FAT: 0.02,
                    Tissue.GILL: 0.05,
                },
            )

    def __hash__(self):
        """
        Custom hash implementation that handles dict fields.

        Returns:
            int: Hash value for the state
        """
        # Convert dict fields to sorted tuples for hashing (sort by tissue name)
        mito_tuple = (
            tuple(sorted(self.mitochondrial_density.items(), key=lambda x: x[0].value))
            if self.mitochondrial_density
            else None
        )
        perfusion_tuple = (
            tuple(sorted(self.tissue_perfusion.items(), key=lambda x: x[0].value))
            if self.tissue_perfusion
            else None
        )

        # Create tuple of all hashable values
        hash_tuple = (
            self.po2,
            self.temperature,
            self.altitude,
            self.salinity,
            self.heart_mass,
            self.hematocrit,
            self.hemoglobin,
            self.blood_volume,
            self.cardiac_output,
            self.lung_volume,
            self.diffusion_capacity,
            self.ventilation_rate,
            self.tidal_volume,
            self.bmr,
            self.vo2max,
            self.respiratory_exchange_ratio,
            self.thermal_conductance,
            self.lower_critical_temp,
            self.upper_critical_temp,
            self.max_thermogenesis,
            mito_tuple,
            perfusion_tuple,
            self.plasma_osmolality,
            self.gill_na_k_atpase,
            self.drinking_rate,
        )

        return hash(hash_tuple)

    def validate(self) -> None:
        """
        Validate physiological parameters are within expected bounds.

        Raises:
            ValueError: If any parameter is outside its valid range
        """
        # Environmental validation
        if not (5.0 <= self.po2 <= 21.0):
            raise ValueError(f"po2 {self.po2} kPa outside valid range 5-21 kPa")
        if not (-40.0 <= self.temperature <= 50.0):
            raise ValueError(
                f"temperature {self.temperature}°C outside valid range -40 to 50°C"
            )
        if not (0.0 <= self.altitude <= 5000.0):
            raise ValueError(f"altitude {self.altitude}m outside valid range 0-5000m")
        if self.salinity is not None and not (0.0 <= self.salinity <= 35.0):
            raise ValueError(
                f"salinity {self.salinity} ppt outside valid range 0-35 ppt"
            )

        # Cardiovascular validation
        if not (3.0 <= self.heart_mass <= 15.0):
            raise ValueError(
                f"heart_mass {self.heart_mass} g/kg outside valid range 3-15 g/kg"
            )
        if not (20.0 <= self.hematocrit <= 70.0):
            raise ValueError(
                f"hematocrit {self.hematocrit}% outside valid range 20-70%"
            )
        if not (10.0 <= self.hemoglobin <= 25.0):
            raise ValueError(
                f"hemoglobin {self.hemoglobin} g/dL outside valid range 10-25 g/dL"
            )
        if not (50.0 <= self.blood_volume <= 120.0):
            raise ValueError(
                f"blood_volume {self.blood_volume} mL/kg outside valid range 50-120 mL/kg"
            )
        if not (100.0 <= self.cardiac_output <= 500.0):
            raise ValueError(
                f"cardiac_output {self.cardiac_output} mL/min/kg outside valid range 100-500 mL/min/kg"
            )

        # Respiratory validation
        if not (40.0 <= self.lung_volume <= 100.0):
            raise ValueError(
                f"lung_volume {self.lung_volume} mL/kg outside valid range 40-100 mL/kg"
            )
        if not (1.0 <= self.diffusion_capacity <= 5.0):
            raise ValueError(
                f"diffusion_capacity {self.diffusion_capacity} outside valid range 1-5"
            )
        if not (10.0 <= self.ventilation_rate <= 100.0):
            raise ValueError(
                f"ventilation_rate {self.ventilation_rate} breaths/min outside valid range 10-100"
            )
        if not (5.0 <= self.tidal_volume <= 15.0):
            raise ValueError(
                f"tidal_volume {self.tidal_volume} mL/kg outside valid range 5-15 mL/kg"
            )

        # Metabolic validation
        if not (20.0 <= self.bmr <= 200.0):
            raise ValueError(f"bmr {self.bmr} mL O2/hr/kg outside valid range 20-200")
        if not (30.0 <= self.vo2max <= 200.0):
            raise ValueError(
                f"vo2max {self.vo2max} mL O2/min/kg outside valid range 30-200"
            )
        if not (0.7 <= self.respiratory_exchange_ratio <= 1.0):
            raise ValueError(
                f"respiratory_exchange_ratio {self.respiratory_exchange_ratio} outside valid range 0.7-1.0"
            )

        # Thermoregulation validation
        if not (1.0 <= self.thermal_conductance <= 5.0):
            raise ValueError(
                f"thermal_conductance {self.thermal_conductance} outside valid range 1-5"
            )
        if not (-10.0 <= self.lower_critical_temp <= 25.0):
            raise ValueError(
                f"lower_critical_temp {self.lower_critical_temp}°C outside valid range -10 to 25°C"
            )
        if not (25.0 <= self.upper_critical_temp <= 45.0):
            raise ValueError(
                f"upper_critical_temp {self.upper_critical_temp}°C outside valid range 25 to 45°C"
            )
        if not (50.0 <= self.max_thermogenesis <= 300.0):
            raise ValueError(
                f"max_thermogenesis {self.max_thermogenesis} outside valid range 50-300"
            )

        # Temperature logic validation
        if self.lower_critical_temp >= self.upper_critical_temp:
            raise ValueError(
                f"lower_critical_temp {self.lower_critical_temp}°C must be < upper_critical_temp {self.upper_critical_temp}°C"
            )

        # Mitochondrial density validation
        if self.mitochondrial_density:
            for tissue, density in self.mitochondrial_density.items():
                if not (0.5 <= density <= 2.0):
                    raise ValueError(
                        f"mitochondrial_density for {tissue.value} outside valid range 0.5-2.0"
                    )

        # Tissue perfusion validation
        if self.tissue_perfusion:
            # First check individual perfusion values
            for tissue, fraction in self.tissue_perfusion.items():
                if not (0.0 <= fraction <= 1.0):
                    raise ValueError(
                        f"tissue_perfusion for {tissue.value} outside valid range 0-1"
                    )

            # Then check if they sum to ~1.0
            total_perfusion = sum(self.tissue_perfusion.values())
            if not (0.95 <= total_perfusion <= 1.05):  # Allow small rounding errors
                raise ValueError(
                    f"tissue_perfusion fractions sum to {total_perfusion}, should sum to ~1.0"
                )

        # Osmoregulation validation (if present)
        if self.plasma_osmolality is not None and not (
            250.0 <= self.plasma_osmolality <= 400.0
        ):
            raise ValueError(
                f"plasma_osmolality {self.plasma_osmolality} mOsm/kg outside valid range 250-400"
            )
        if self.gill_na_k_atpase is not None and not (
            1.0 <= self.gill_na_k_atpase <= 20.0
        ):
            raise ValueError(
                f"gill_na_k_atpase {self.gill_na_k_atpase} outside valid range 1-20"
            )
        if self.drinking_rate is not None and not (0.0 <= self.drinking_rate <= 50.0):
            raise ValueError(
                f"drinking_rate {self.drinking_rate} mL/hr/kg outside valid range 0-50"
            )

    def compute_aerobic_scope(self) -> float:
        """
        Calculate aerobic scope (VO2max - BMR).

        Returns:
            float: Aerobic scope in mL O2/min/kg
        """
        return self.vo2max - (self.bmr / 60)  # Convert BMR from hr to min

    def oxygen_delivery(self, tissue: Tissue) -> float:
        """
        Calculate O2 delivery to specific tissue.

        Args:
            tissue: Target tissue type

        Returns:
            float: O2 delivery rate in mL O2/min/kg

        Raises:
            ValueError: If tissue not found in perfusion data
        """
        if tissue not in self.tissue_perfusion:
            raise ValueError(f"Tissue {tissue.value} not found in perfusion data")

        blood_flow_fraction = self.tissue_perfusion[tissue]

        # O2 content calculation: Hemoglobin capacity × saturation
        # Using simplified saturation based on PO2
        saturation = min(1.0, self.po2 / 13.3)  # Simplified O2 saturation curve
        o2_content = self.hemoglobin * 1.34 * saturation  # mL O2/dL blood

        # Convert to mL O2/mL blood and calculate delivery
        o2_content_ml_per_ml = o2_content / 100  # Convert dL to mL
        tissue_blood_flow = self.cardiac_output * blood_flow_fraction

        return tissue_blood_flow * o2_content_ml_per_ml

    def thermal_neutral_zone_width(self) -> float:
        """
        Calculate the width of the thermal neutral zone.

        Returns:
            float: TNZ width in °C
        """
        return self.upper_critical_temp - self.lower_critical_temp

    def is_in_thermal_neutral_zone(self, temperature: Optional[float] = None) -> bool:
        """
        Check if organism is in thermal neutral zone.

        Args:
            temperature: Temperature to check (uses self.temperature if None)

        Returns:
            bool: True if in thermal neutral zone
        """
        temp = temperature if temperature is not None else self.temperature
        return self.lower_critical_temp <= temp <= self.upper_critical_temp
