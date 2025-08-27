"""
Demographics Analysis

This module provides demographic analysis tools for population data,
including age structure, survival curves, and population growth metrics.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from .population import Population, Individual


@dataclass
class AgeClass:
    """Represents an age class in the population"""
    age: int
    count: int
    mean_fitness: float
    survival_rate: float = 0.0


class Demographics:
    """
    Provides demographic analysis for populations.
    
    This class analyzes population structure, survival patterns,
    and demographic rates for evolutionary simulations.
    """
    
    def __init__(self, population: Population):
        """
        Initialize demographics analysis for a population.
        
        Args:
            population: Population to analyze
        """
        self.population = population
    
    def get_age_structure(self) -> List[AgeClass]:
        """
        Calculate age structure of the population.
        
        Returns:
            List of AgeClass objects representing age distribution
        """
        if not self.population.individuals:
            return []
        
        # Group individuals by age
        age_groups: Dict[int, List[Individual]] = {}
        for individual in self.population.individuals:
            if individual.age not in age_groups:
                age_groups[individual.age] = []
            age_groups[individual.age].append(individual)
        
        # Create age classes
        age_classes = []
        for age, individuals in sorted(age_groups.items()):
            mean_fitness = np.mean([ind.fitness for ind in individuals])
            age_class = AgeClass(
                age=age,
                count=len(individuals),
                mean_fitness=mean_fitness
            )
            age_classes.append(age_class)
        
        return age_classes
    
    def calculate_survival_rates(self, 
                                previous_age_structure: List[AgeClass]) -> List[AgeClass]:
        """
        Calculate survival rates by comparing with previous generation.
        
        Args:
            previous_age_structure: Age structure from previous generation
            
        Returns:
            Current age structure with survival rates calculated
        """
        current_structure = self.get_age_structure()
        
        # Create lookup for previous generation
        prev_lookup = {ac.age - 1: ac.count for ac in previous_age_structure if ac.age > 0}
        
        # Calculate survival rates
        for age_class in current_structure:
            if age_class.age > 0:
                prev_count = prev_lookup.get(age_class.age - 1, 0)
                if prev_count > 0:
                    age_class.survival_rate = age_class.count / prev_count
                else:
                    age_class.survival_rate = 0.0
            else:
                # Newborns don't have survival rate from previous generation
                age_class.survival_rate = 1.0
        
        return current_structure
    
    def get_population_growth_rate(self) -> float:
        """
        Calculate instantaneous population growth rate.
        
        Returns:
            Growth rate (r) where N(t) = N(0) * exp(rt)
        """
        if self.population.generation <= 0:
            return 0.0
        
        # Simple approximation: ln(current_size / initial_size) / generation
        current_size = len(self.population.individuals)
        initial_size = self.population.size
        
        if initial_size <= 0:
            return 0.0
        
        return np.log(current_size / initial_size) / self.population.generation
    
    def calculate_life_expectancy(self) -> float:
        """
        Calculate life expectancy based on current age structure.
        
        Returns:
            Expected lifespan in generations
        """
        age_structure = self.get_age_structure()
        
        if not age_structure:
            return 0.0
        
        # Weighted average age
        total_individuals = sum(ac.count for ac in age_structure)
        weighted_sum = sum(ac.age * ac.count for ac in age_structure)
        
        return weighted_sum / total_individuals if total_individuals > 0 else 0.0
    
    def get_demographic_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive demographic summary.
        
        Returns:
            Dictionary with demographic metrics
        """
        age_structure = self.get_age_structure()
        
        if not age_structure:
            return {
                'total_population': 0,
                'age_classes': 0,
                'life_expectancy': 0.0,
                'growth_rate': 0.0,
                'juvenile_fraction': 0.0,
                'adult_fraction': 0.0
            }
        
        total_pop = sum(ac.count for ac in age_structure)
        max_age = max(ac.age for ac in age_structure)
        
        # Define juvenile vs adult (arbitrary cutoff at age 1)
        juvenile_count = sum(ac.count for ac in age_structure if ac.age == 0)
        adult_count = total_pop - juvenile_count
        
        return {
            'total_population': total_pop,
            'age_classes': len(age_structure),
            'max_age': max_age,
            'life_expectancy': self.calculate_life_expectancy(),
            'growth_rate': self.get_population_growth_rate(),
            'juvenile_fraction': juvenile_count / total_pop if total_pop > 0 else 0.0,
            'adult_fraction': adult_count / total_pop if total_pop > 0 else 0.0,
            'age_structure': age_structure
        }
    
    def track_demographic_trajectory(self, 
                                   generations: int) -> List[Dict[str, Any]]:
        """
        Track demographic changes over multiple generations.
        
        Args:
            generations: Number of generations to track
            
        Returns:
            List of demographic summaries for each generation
        """
        trajectory = []
        
        for gen in range(generations):
            summary = self.get_demographic_summary()
            summary['generation'] = gen
            trajectory.append(summary)
            
            # This would typically be called as part of the evolution loop
            # Here we just record current state
        
        return trajectory