"""
Tests for Metabolic Calculator Module

Ensures BMR, VO2max, thermal performance, and aerobic scope calculations are correct.
"""
import numpy as np
import pytest
from ipe.core.physiology.state import PhysiologicalState
from ipe.core.physiology.metabolism import MetabolicCalculator
from ipe.core.physiology.allometry import kleiber_bmr

class TestMetabolicCalculator:
    def test_bmr(self):
        bmr = kleiber_bmr(50.0)
        assert np.isclose(bmr, 3.4 * (50.0 ** 0.75))

    def test_vo2max(self):
        state = PhysiologicalState(po2=15.0, temperature=37.0, altitude=0.0)
        vo2 = MetabolicCalculator.vo2max(state, 50.0)
        assert vo2 > 0

    def test_thermal_performance(self):
        state = PhysiologicalState(po2=15.0, temperature=40.0, altitude=0.0)
        perf = MetabolicCalculator.thermal_performance(state)
        assert perf > 0

    def test_aerobic_scope(self):
        state = PhysiologicalState(po2=15.0, temperature=37.0, altitude=0.0)
        scope = MetabolicCalculator.aerobic_scope(state, 50.0)
        assert scope > 0

    def test_altitude_correction(self):
        state = PhysiologicalState(po2=15.0, temperature=37.0, altitude=5000.0)
        corr = MetabolicCalculator.altitude_correction(state)
        assert 0.5 <= corr <= 1.0
