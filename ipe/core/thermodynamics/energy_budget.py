"""
Energy Budget Calculations

Implements heat transfer and energy flow calculations for physiological systems.
"""

from typing import Dict


class EnergyBudget:
    """
    Calculates energy flows and heat transfer.
    """

    @staticmethod
    def heat_transfer(temp_body: float, temp_env: float, conductance: float) -> float:
        """Calculate heat transfer (W) from body to environment."""
        return conductance * (temp_body - temp_env)

    @staticmethod
    def total_energy_flow(flows: Dict[str, float]) -> float:
        """Sum all energy flows (W)."""
        return sum(flows.values())

    # Extend with more detailed energy budget calculations as needed
