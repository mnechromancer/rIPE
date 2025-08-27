"""
Mutation Model

This module implements mutation mechanisms for evolutionary simulations,
including mutational variance, pleiotropic effects, and mutation rate evolution.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

from .population import Individual
from .genetic_architecture import GeneticArchitecture, GeneticLocus


@dataclass
class MutationParameters:
    """
    Parameters for mutation model configuration.
    
    Controls mutation rates, effect sizes, and evolutionary
    dynamics of the mutation process itself.
    """
    
    base_mutation_rate: float = 1e-4
    mutational_variance: float = 0.01
    mutation_rate_variance: float = 1e-6  # Variance in mutation rate evolution
    pleiotropic_correlation: float = 0.1
    
    # Mutation rate evolution parameters  
    mutation_rate_heritability: float = 0.3
    optimal_mutation_rate: float = 1e-4
    mutation_rate_selection_strength: float = 0.1
    
    def __post_init__(self):
        """Validate mutation parameters"""
        if self.base_mutation_rate < 0:
            raise ValueError("Base mutation rate must be non-negative")
        if self.mutational_variance < 0:
            raise ValueError("Mutational variance must be non-negative") 
        if not 0 <= self.pleiotropic_correlation <= 1:
            raise ValueError("Pleiotropic correlation must be between 0 and 1")


class MutationStrategy(ABC):
    """Abstract base class for mutation strategies"""
    
    @abstractmethod
    def mutate(self, individual: Individual, 
               architecture: GeneticArchitecture) -> Individual:
        """
        Apply mutations to an individual.
        
        Args:
            individual: Individual to mutate
            architecture: Genetic architecture defining mutation effects
            
        Returns:
            Mutated individual
        """
        pass


class GaussianMutation(MutationStrategy):
    """
    Gaussian mutation: add normally distributed random effects.
    
    This is the standard mutation model for quantitative traits,
    where mutations have small additive effects drawn from
    a normal distribution.
    """
    
    def __init__(self, parameters: MutationParameters):
        """Initialize Gaussian mutation with parameters"""
        self.parameters = parameters
    
    def mutate(self, individual: Individual, 
               architecture: GeneticArchitecture) -> Individual:
        """Apply Gaussian mutations to individual"""
        # Create copy of individual for mutation
        mutated_individual = Individual(
            id=individual.id,
            physiological_state=individual.physiological_state,
            fitness=individual.fitness,
            genetic_values=individual.genetic_values.copy() if individual.genetic_values else {},
            age=individual.age
        )
        
        # Apply mutations to each locus
        for locus_name, locus in architecture.loci.items():
            if np.random.random() < locus.mutation_rate:
                # Mutational effect size
                effect_size = np.random.normal(0, locus.allelic_variance)
                
                # Apply mutation
                current_value = mutated_individual.genetic_values.get(locus_name, 0.0)
                mutated_individual.genetic_values[locus_name] = current_value + effect_size
        
        return mutated_individual


class PleiotopicMutation(MutationStrategy):
    """
    Pleiotropic mutation: correlated effects across multiple traits.
    
    Mutations at a locus affect multiple traits simultaneously,
    with correlations determined by the genetic architecture.
    """
    
    def __init__(self, parameters: MutationParameters):
        """Initialize pleiotropic mutation"""
        self.parameters = parameters
    
    def mutate(self, individual: Individual, 
               architecture: GeneticArchitecture) -> Individual:
        """Apply pleiotropic mutations"""
        # Create mutated copy
        mutated_individual = Individual(
            id=individual.id,
            physiological_state=individual.physiological_state,
            fitness=individual.fitness,
            genetic_values=individual.genetic_values.copy() if individual.genetic_values else {},
            age=individual.age
        )
        
        # Get pleiotropy matrix
        if architecture.pleiotropy_matrix is None:
            # Fall back to independent mutations
            gaussian_mutator = GaussianMutation(self.parameters)
            return gaussian_mutator.mutate(individual, architecture)
        
        locus_names = list(architecture.loci.keys())
        trait_names = list(architecture.traits.keys())
        
        # Generate correlated mutational effects
        n_loci = len(locus_names)
        if n_loci > 0:
            # Independent mutations at each locus
            mutations = np.random.normal(0, self.parameters.mutational_variance, n_loci)
            
            # Apply mutations based on whether locus mutates
            for i, locus_name in enumerate(locus_names):
                locus = architecture.loci[locus_name]
                if np.random.random() < locus.mutation_rate:
                    current_value = mutated_individual.genetic_values.get(locus_name, 0.0)
                    mutated_individual.genetic_values[locus_name] = current_value + mutations[i]
        
        return mutated_individual


class MutationEngine:
    """
    Main mutation engine managing mutation processes.
    
    Coordinates different mutation strategies, tracks mutation rates,
    and handles evolutionary dynamics of the mutation process.
    """
    
    def __init__(self, 
                 parameters: MutationParameters,
                 architecture: GeneticArchitecture,
                 strategy: Optional[MutationStrategy] = None):
        """
        Initialize mutation engine.
        
        Args:
            parameters: Mutation parameters
            architecture: Genetic architecture
            strategy: Mutation strategy (defaults to Gaussian)
        """
        self.parameters = parameters
        self.architecture = architecture
        self.strategy = strategy or GaussianMutation(parameters)
        
        # Track mutation statistics
        self.mutation_counts: Dict[str, int] = {}
        self.effect_sizes: Dict[str, List[float]] = {}
        
        # Evolution of mutation rate itself
        self.current_mutation_rates: Dict[str, float] = {}
        self._initialize_mutation_rates()
    
    def _initialize_mutation_rates(self):
        """Initialize mutation rates for all loci"""
        for locus_name in self.architecture.loci.keys():
            self.current_mutation_rates[locus_name] = self.parameters.base_mutation_rate
    
    def mutate_individual(self, individual: Individual) -> Individual:
        """
        Mutate a single individual.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        return self.strategy.mutate(individual, self.architecture)
    
    def mutate_population(self, individuals: List[Individual]) -> List[Individual]:
        """
        Apply mutations to entire population.
        
        Args:
            individuals: List of individuals to mutate
            
        Returns:
            List of mutated individuals
        """
        return [self.mutate_individual(ind) for ind in individuals]
    
    def evolve_mutation_rates(self, population: List[Individual]) -> None:
        """
        Evolve mutation rates based on fitness effects.
        
        Implements evolution of evolvability - mutation rates can
        evolve based on their fitness consequences.
        
        Args:
            population: Current population for rate evolution
        """
        if not population or not self.parameters.mutation_rate_heritability > 0:
            return
        
        # Calculate fitness effects of different mutation rates
        for locus_name in self.architecture.loci.keys():
            current_rate = self.current_mutation_rates[locus_name]
            
            # Selection on mutation rate: penalize rates far from optimum
            optimal_rate = self.parameters.optimal_mutation_rate
            rate_deviation = current_rate - optimal_rate
            
            # Fitness effect proportional to deviation from optimum
            rate_fitness_effect = -self.parameters.mutation_rate_selection_strength * rate_deviation**2
            
            # Evolve mutation rate
            rate_change = np.random.normal(
                rate_fitness_effect * self.parameters.mutation_rate_heritability,
                self.parameters.mutation_rate_variance
            )
            
            new_rate = max(0, current_rate + rate_change)
            self.current_mutation_rates[locus_name] = new_rate
            
            # Update locus mutation rate
            if locus_name in self.architecture.loci:
                self.architecture.loci[locus_name].mutation_rate = new_rate
    
    def maintain_standing_variation(self, 
                                   population: List[Individual],
                                   target_variance: float = 0.01) -> List[Individual]:
        """
        Maintain standing genetic variation in population.
        
        Adds mutations to maintain a target level of genetic
        variance, preventing complete fixation.
        
        Args:
            population: Current population
            target_variance: Target genetic variance to maintain
            
        Returns:
            Population with added variation if needed
        """
        if not population:
            return population
        
        # Calculate current genetic variance for each locus
        for locus_name in self.architecture.loci.keys():
            genetic_values = [
                ind.genetic_values.get(locus_name, 0.0) 
                for ind in population
            ]
            
            current_variance = np.var(genetic_values)
            
            # Add variation if below target
            if current_variance < target_variance:
                variance_deficit = target_variance - current_variance
                mutation_variance = np.sqrt(variance_deficit / len(population))
                
                # Add mutations to restore variance
                for individual in population:
                    if np.random.random() < 0.1:  # 10% chance per individual
                        mutation_effect = np.random.normal(0, mutation_variance)
                        current_value = individual.genetic_values.get(locus_name, 0.0)
                        individual.genetic_values[locus_name] = current_value + mutation_effect
        
        return population
    
    def get_mutation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about mutation process.
        
        Returns:
            Dictionary with mutation statistics
        """
        stats = {
            'current_mutation_rates': self.current_mutation_rates.copy(),
            'mean_mutation_rate': np.mean(list(self.current_mutation_rates.values())),
            'mutation_rate_variance': np.var(list(self.current_mutation_rates.values())),
            'total_mutations': sum(self.mutation_counts.values()),
            'mutations_per_locus': self.mutation_counts.copy()
        }
        
        if self.effect_sizes:
            stats['mean_effect_sizes'] = {
                locus: np.mean(effects) 
                for locus, effects in self.effect_sizes.items()
            }
            stats['effect_size_variances'] = {
                locus: np.var(effects)
                for locus, effects in self.effect_sizes.items()
            }
        
        return stats
    
    def reset_statistics(self):
        """Reset mutation tracking statistics"""
        self.mutation_counts.clear()
        self.effect_sizes.clear()


def create_default_mutation_engine(architecture: GeneticArchitecture) -> MutationEngine:
    """
    Create a mutation engine with default parameters.
    
    Args:
        architecture: Genetic architecture to use
        
    Returns:
        MutationEngine with default configuration
    """
    parameters = MutationParameters(
        base_mutation_rate=1e-4,
        mutational_variance=0.01,
        pleiotropic_correlation=0.1
    )
    
    strategy = PleiotopicMutation(parameters)
    
    return MutationEngine(parameters, architecture, strategy)