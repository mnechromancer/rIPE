"""
Evolution Simulator Package

This package implements the evolution simulation components for IPE,
including population dynamics, selection mechanisms, mutation models,
and rapid evolution scenarios.
"""

from .population import Population, Individual
from .demographics import Demographics

__all__ = [
    'Population',
    'Individual', 
    'Demographics'
]