"""
Plasticity Module for IPE

This module implements phenotypic plasticity modeling including:
- Reaction norm representation and G×E interactions
- Maladaptive plasticity detection 
- Genetic assimilation and canalization processes
"""

from .reaction_norm import ReactionNorm, PlasticityMagnitude
from .gxe import GxEInteraction

__all__ = [
    "ReactionNorm", 
    "PlasticityMagnitude",
    "GxEInteraction"
]