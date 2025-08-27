"""
Plasticity Module for IPE

This module implements phenotypic plasticity modeling including:
- Reaction norm representation and G×E interactions
- Maladaptive plasticity detection 
- Genetic assimilation and canalization processes
"""

from .reaction_norm import ReactionNorm, PlasticityMagnitude
from .gxe import GxEInteraction
from .maladaptive import MaladaptiveDetector, MaladaptiveResponse, MaladaptationType
from .maladaptive import quadratic_fitness, linear_fitness, thermal_fitness
from .assimilation import GeneticAssimilationEngine, AssimilationTrajectory, AssimilationStage
from .canalization import CanalizationEngine, CanalizationTrajectory, CanalizationMeasure, CanalizationType

__all__ = [
    "ReactionNorm", 
    "PlasticityMagnitude",
    "GxEInteraction",
    "MaladaptiveDetector",
    "MaladaptiveResponse", 
    "MaladaptationType",
    "quadratic_fitness",
    "linear_fitness", 
    "thermal_fitness",
    "GeneticAssimilationEngine",
    "AssimilationTrajectory",
    "AssimilationStage",
    "CanalizationEngine",
    "CanalizationTrajectory", 
    "CanalizationMeasure",
    "CanalizationType"
]