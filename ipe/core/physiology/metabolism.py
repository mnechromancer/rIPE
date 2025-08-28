"""
Metabolic Calculator Module

Implements BMR, VO2max, thermal performance, and aerobic scope calculations.
"""

from typing import Optional
import numpy as np
from ipe.core.physiology.state import PhysiologicalState
from ipe.core.physiology.allometry import kleiber_bmr

Q10 = 2.0  # Default Q10 coefficient


class MetabolicCalculator:
    """
    Provides metabolic calculations for physiological states.
    """

    @staticmethod
    def bmr(state: PhysiologicalState, body_mass: float) -> float:
        """Calculate BMR using Kleiber's law (W)"""
        return kleiber_bmr(body_mass)

    @staticmethod
    def vo2max(state: PhysiologicalState, body_mass: float) -> float:
        """Estimate VO2max from physiological parameters (mL O2/min)"""
        # Example formula: cardiac_output * (arterial O2 - venous O2)
        return state.cardiac_output * (state.hemoglobin * 1.34 * state.hematocrit / 100)

    @staticmethod
    def thermal_performance(state: PhysiologicalState, t_ref: float = 37.0) -> float:
        """Thermal performance curve using Q10"""
        return Q10 ** ((state.temperature - t_ref) / 10)

    @staticmethod
    def aerobic_scope(state: PhysiologicalState, body_mass: float) -> float:
        """Calculate aerobic scope (VO2max/BMR)"""
        bmr = MetabolicCalculator.bmr(state, body_mass)
        vo2 = MetabolicCalculator.vo2max(state, body_mass)
        return vo2 / bmr if bmr > 0 else np.nan

    @staticmethod
    def altitude_correction(state: PhysiologicalState) -> float:
        """Apply altitude correction for O2 availability"""
        # Example: linear decrease in O2 with altitude
        return max(0.5, 1.0 - state.altitude / 10000)
