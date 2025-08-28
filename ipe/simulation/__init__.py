"""
Evolution Simulator Package

This package implements the evolution simulation components for IPE,
including population dynamics, selection mechanisms, mutation models,
and rapid evolution scenarios.
"""

from .population import Population, Individual
from .demographics import Demographics
from .selection import (
    SelectionStrategy,
    TruncationSelection,
    ProportionalSelection,
    TournamentSelection,
    FrequencyDependentSelection,
    MultiTraitSelection,
    SelectionAnalyzer,
    SelectionDifferential,
)
from .genetic_architecture import (
    GeneticLocus,
    TraitArchitecture,
    TraitType,
    GeneticArchitecture,
    create_default_architecture,
)
from .mutation import (
    MutationParameters,
    MutationStrategy,
    GaussianMutation,
    PleiotopicMutation,
    MutationEngine,
    create_default_mutation_engine,
)
from .rapid_evolution import (
    EnvironmentalChange,
    PlasticityEvolutionTracker,
    RapidEvolutionSimulator,
    create_freshwater_invasion_scenario,
    create_altitude_adaptation_scenario,
)
from .contemporary import (
    ContemporaryEvolutionModel,
    ExperimentalEvolutionModel,
    create_predefined_scenarios,
)

__all__ = [
    "Population",
    "Individual",
    "Demographics",
    "SelectionStrategy",
    "TruncationSelection",
    "ProportionalSelection",
    "TournamentSelection",
    "FrequencyDependentSelection",
    "MultiTraitSelection",
    "SelectionAnalyzer",
    "SelectionDifferential",
    "GeneticLocus",
    "TraitArchitecture",
    "TraitType",
    "GeneticArchitecture",
    "create_default_architecture",
    "MutationParameters",
    "MutationStrategy",
    "GaussianMutation",
    "PleiotopicMutation",
    "MutationEngine",
    "create_default_mutation_engine",
    "EnvironmentalChange",
    "PlasticityEvolutionTracker",
    "RapidEvolutionSimulator",
    "create_freshwater_invasion_scenario",
    "create_altitude_adaptation_scenario",
    "ContemporaryEvolutionModel",
    "ExperimentalEvolutionModel",
    "create_predefined_scenarios",
]
