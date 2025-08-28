"""
Tests for Thermodynamic Constraints Engine

Ensures energy balance, heat transfer, efficiency limits, and constraint violation detection work as expected.
"""

import numpy as np
from ipe.core.thermodynamics.constraints import ThermodynamicConstraints
from ipe.core.thermodynamics.energy_budget import EnergyBudget


class TestThermodynamicConstraints:
    def test_validate_energy_balance(self):
        inputs = {"food": 100.0, "O2": 50.0}
        outputs = {"work": 80.0, "heat": 70.0}
        assert ThermodynamicConstraints.validate_energy_balance(inputs, outputs)

    def test_efficiency_limit(self):
        eff = ThermodynamicConstraints.efficiency_limit(50.0, 100.0)
        assert np.isclose(eff, 0.5)

    def test_detect_constraint_violation(self):
        assert ThermodynamicConstraints.detect_constraint_violation(120.0, 100.0)
        assert not ThermodynamicConstraints.detect_constraint_violation(80.0, 100.0)


class TestEnergyBudget:
    def test_heat_transfer(self):
        q = EnergyBudget.heat_transfer(37.0, 25.0, 0.5)
        assert np.isclose(q, 6.0)

    def test_total_energy_flow(self):
        flows = {"work": 50.0, "heat": 30.0}
        total = EnergyBudget.total_energy_flow(flows)
        assert np.isclose(total, 80.0)
