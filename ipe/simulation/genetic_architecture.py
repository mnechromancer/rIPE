"""
Genetic Architecture

This module defines the genetic architecture and trait mapping
for evolutionary simulations, including pleiotropic effects
and genetic-phenotype relationships.
"""

from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ..core.physiology.state import PhysiologicalState


class TraitType(Enum):
    """Types of traits in the genetic architecture"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    THRESHOLD = "threshold"


@dataclass
class GeneticLocus:
    """
    Represents a genetic locus affecting one or more traits.
    
    Each locus has allelic effects, mutation rate, and
    pleiotropic effects on multiple traits.
    """
    
    name: str
    trait_effects: Dict[str, float]  # Effect on each trait
    mutation_rate: float = 1e-4
    allelic_variance: float = 0.01
    dominance_coefficient: float = 0.0  # 0 = additive, 0.5 = dominant, -0.5 = recessive
    epistatic_interactions: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate genetic locus parameters"""
        if self.mutation_rate < 0:
            raise ValueError("Mutation rate must be non-negative")
        if self.allelic_variance < 0:
            raise ValueError("Allelic variance must be non-negative")
        if not -1 <= self.dominance_coefficient <= 1:
            raise ValueError("Dominance coefficient must be between -1 and 1")


@dataclass 
class TraitArchitecture:
    """
    Defines the genetic architecture of a quantitative trait.
    
    Includes information about contributing loci, environmental
    effects, and trait constraints.
    """
    
    name: str
    trait_type: TraitType
    contributing_loci: List[str]  # Names of loci affecting this trait
    environmental_variance: float = 0.01
    heritability: float = 0.5
    phenotypic_bounds: Optional[Tuple[float, float]] = None
    transformation_function: Optional[Callable[[float], float]] = None
    
    def __post_init__(self):
        """Validate trait architecture"""
        if not 0 <= self.heritability <= 1:
            raise ValueError("Heritability must be between 0 and 1")
        if self.environmental_variance < 0:
            raise ValueError("Environmental variance must be non-negative")


class GeneticArchitecture:
    """
    Manages the complete genetic architecture for a simulation.
    
    This class handles the mapping between genetic loci and
    phenotypic traits, including pleiotropic effects and
    genetic-environment interactions.
    """
    
    def __init__(self):
        """Initialize empty genetic architecture"""
        self.loci: Dict[str, GeneticLocus] = {}
        self.traits: Dict[str, TraitArchitecture] = {}
        self.pleiotropy_matrix: Optional[np.ndarray] = None
        
    def add_locus(self, locus: GeneticLocus) -> None:
        """Add a genetic locus to the architecture"""
        self.loci[locus.name] = locus
        self._update_pleiotropy_matrix()
    
    def add_trait(self, trait: TraitArchitecture) -> None:
        """Add a trait to the architecture"""
        self.traits[trait.name] = trait
        self._update_pleiotropy_matrix()
    
    def _update_pleiotropy_matrix(self) -> None:
        """Update the pleiotropy matrix after changes"""
        if not self.loci or not self.traits:
            return
        
        locus_names = list(self.loci.keys())
        trait_names = list(self.traits.keys())
        
        matrix = np.zeros((len(locus_names), len(trait_names)))
        
        for i, locus_name in enumerate(locus_names):
            locus = self.loci[locus_name]
            for j, trait_name in enumerate(trait_names):
                effect = locus.trait_effects.get(trait_name, 0.0)
                matrix[i, j] = effect
        
        self.pleiotropy_matrix = matrix
    
    def get_locus_effects(self, locus_name: str) -> Dict[str, float]:
        """Get all trait effects for a locus"""
        if locus_name not in self.loci:
            return {}
        return self.loci[locus_name].trait_effects.copy()
    
    def get_trait_architecture(self, trait_name: str) -> Optional[TraitArchitecture]:
        """Get architecture for a specific trait"""
        return self.traits.get(trait_name)
    
    def calculate_breeding_values(self, 
                                genotype: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate breeding values for all traits given a genotype.
        
        Args:
            genotype: Dict mapping locus names to genetic values
            
        Returns:
            Dict mapping trait names to breeding values
        """
        breeding_values = {}
        
        for trait_name, trait_arch in self.traits.items():
            breeding_value = 0.0
            
            for locus_name in trait_arch.contributing_loci:
                if locus_name in genotype and locus_name in self.loci:
                    genetic_value = genotype[locus_name]
                    effect_size = self.loci[locus_name].trait_effects.get(trait_name, 0.0)
                    breeding_value += genetic_value * effect_size
            
            breeding_values[trait_name] = breeding_value
        
        return breeding_values
    
    def calculate_phenotype(self, 
                          genotype: Dict[str, float],
                          environment: Optional[PhysiologicalState] = None) -> Dict[str, float]:
        """
        Calculate phenotypic values from genotype and environment.
        
        Args:
            genotype: Dict mapping locus names to genetic values
            environment: Optional environmental state for G×E effects
            
        Returns:
            Dict mapping trait names to phenotypic values
        """
        breeding_values = self.calculate_breeding_values(genotype)
        phenotypes = {}
        
        for trait_name, breeding_value in breeding_values.items():
            trait_arch = self.traits[trait_name]
            
            # Add environmental noise
            env_noise = np.random.normal(0, np.sqrt(trait_arch.environmental_variance))
            phenotype = breeding_value + env_noise
            
            # Apply transformation function if provided
            if trait_arch.transformation_function:
                phenotype = trait_arch.transformation_function(phenotype)
            
            # Apply phenotypic bounds if provided
            if trait_arch.phenotypic_bounds:
                min_val, max_val = trait_arch.phenotypic_bounds
                phenotype = max(min_val, min(max_val, phenotype))
            
            phenotypes[trait_name] = phenotype
        
        return phenotypes
    
    def get_genetic_correlations(self) -> np.ndarray:
        """
        Calculate genetic correlations between traits based on pleiotropy.
        
        Returns:
            Correlation matrix between traits
        """
        if self.pleiotropy_matrix is None or self.pleiotropy_matrix.size == 0:
            return np.array([])
        
        # Genetic covariance matrix = G' * V_mut * G
        # where G is pleiotropy matrix, V_mut is diagonal matrix of mutational variances
        
        mut_variances = np.array([
            self.loci[locus_name].allelic_variance 
            for locus_name in self.loci.keys()
        ])
        
        G = self.pleiotropy_matrix
        V_mut = np.diag(mut_variances)
        
        # Genetic covariance matrix
        G_cov = G.T @ V_mut @ G
        
        # Convert to correlations
        G_std = np.sqrt(np.diag(G_cov))
        correlations = np.outer(G_std, G_std)
        correlations = np.divide(G_cov, correlations, 
                               out=np.zeros_like(G_cov), where=correlations!=0)
        
        return correlations
    
    def serialize(self) -> Dict:
        """Serialize genetic architecture to dictionary"""
        return {
            'loci': {name: {
                'name': locus.name,
                'trait_effects': locus.trait_effects,
                'mutation_rate': locus.mutation_rate,
                'allelic_variance': locus.allelic_variance,
                'dominance_coefficient': locus.dominance_coefficient,
                'epistatic_interactions': locus.epistatic_interactions
            } for name, locus in self.loci.items()},
            'traits': {name: {
                'name': trait.name,
                'trait_type': trait.trait_type.value,
                'contributing_loci': trait.contributing_loci,
                'environmental_variance': trait.environmental_variance,
                'heritability': trait.heritability,
                'phenotypic_bounds': trait.phenotypic_bounds
            } for name, trait in self.traits.items()}
        }


def create_default_architecture() -> GeneticArchitecture:
    """
    Create a default genetic architecture for physiological traits.
    
    Returns:
        GeneticArchitecture with basic physiological trait mapping
    """
    arch = GeneticArchitecture()
    
    # Add basic physiological loci
    loci = [
        GeneticLocus(
            name="cardio1",
            trait_effects={"heart_mass": 0.5, "cardiac_output": 0.3},
            mutation_rate=1e-4,
            allelic_variance=0.01
        ),
        GeneticLocus(
            name="resp1", 
            trait_effects={"lung_volume": 0.4, "diffusion_capacity": 0.6},
            mutation_rate=1e-4,
            allelic_variance=0.01
        ),
        GeneticLocus(
            name="hemo1",
            trait_effects={"hematocrit": 0.7, "hemoglobin": 0.5},
            mutation_rate=1e-4,
            allelic_variance=0.01
        ),
        GeneticLocus(
            name="metab1",
            trait_effects={"bmr": 0.6, "vo2max": 0.4},
            mutation_rate=1e-4,
            allelic_variance=0.01
        )
    ]
    
    for locus in loci:
        arch.add_locus(locus)
    
    # Add trait architectures
    traits = [
        TraitArchitecture(
            name="heart_mass",
            trait_type=TraitType.CONTINUOUS,
            contributing_loci=["cardio1"],
            heritability=0.6,
            phenotypic_bounds=(3.0, 15.0)
        ),
        TraitArchitecture(
            name="cardiac_output", 
            trait_type=TraitType.CONTINUOUS,
            contributing_loci=["cardio1"],
            heritability=0.5,
            phenotypic_bounds=(100.0, 500.0)
        ),
        TraitArchitecture(
            name="lung_volume",
            trait_type=TraitType.CONTINUOUS,
            contributing_loci=["resp1"],
            heritability=0.7,
            phenotypic_bounds=(40.0, 100.0)
        ),
        TraitArchitecture(
            name="hematocrit",
            trait_type=TraitType.CONTINUOUS,
            contributing_loci=["hemo1"],
            heritability=0.8,
            phenotypic_bounds=(20.0, 70.0)
        )
    ]
    
    for trait in traits:
        arch.add_trait(trait)
    
    return arch