"""
Genotype × Environment (G×E) Interaction Module

This module implements modeling of genotype-by-environment interactions
for phenotypic plasticity analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
import json

from .reaction_norm import ReactionNorm, PlasticityMagnitude


@dataclass
class GxEInteraction:
    """
    Genotype × Environment interaction analysis.
    
    Models how different genotypes respond to environmental variation
    and quantifies interaction effects.
    """
    
    genotypes: Dict[str, ReactionNorm]  # Genotype ID -> ReactionNorm
    environmental_variable: str         # Environmental variable name
    trait_name: str                    # Trait being analyzed
    
    def __post_init__(self):
        """Validate G×E interaction data"""
        if len(self.genotypes) < 2:
            raise ValueError("Need at least 2 genotypes for G×E analysis")
            
        # Check that all reaction norms use same environmental variable and trait
        env_vars = set()
        traits = set() 
        for genotype_id, norm in self.genotypes.items():
            if not isinstance(norm, ReactionNorm):
                raise TypeError(f"Genotype {genotype_id} norm must be ReactionNorm")
            env_vars.add(norm.environmental_variable)
            traits.add(norm.trait_name)
            
        if len(env_vars) > 1:
            raise ValueError(f"All genotypes must use same environmental variable, found: {env_vars}")
        if len(traits) > 1:
            raise ValueError(f"All genotypes must measure same trait, found: {traits}")

    def get_common_environment_range(self) -> Tuple[float, float]:
        """
        Get environmental range common to all genotypes.
        
        Returns:
            Tuple of (min_env, max_env) common to all genotypes
        """
        min_envs = []
        max_envs = []
        
        for norm in self.genotypes.values():
            min_envs.append(np.min(norm.environments))
            max_envs.append(np.max(norm.environments))
            
        return (max(min_envs), min(max_envs))

    def evaluate_at_environments(self, environments: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluate all genotypes at specified environments.
        
        Args:
            environments: Array of environmental values
            
        Returns:
            Dictionary mapping genotype ID to predicted phenotypes
        """
        results = {}
        for genotype_id, norm in self.genotypes.items():
            results[genotype_id] = norm.predict_phenotype(environments)
        return results

    def interaction_variance(self, environments: Optional[np.ndarray] = None) -> float:
        """
        Calculate variance explained by G×E interaction.
        
        Args:
            environments: Environmental values to test (uses common range if None)
            
        Returns:
            Proportion of variance explained by G×E interaction
        """
        if environments is None:
            min_env, max_env = self.get_common_environment_range()
            environments = np.linspace(min_env, max_env, 20)
        
        # Get phenotype predictions for all genotypes
        predictions = self.evaluate_at_environments(environments)
        
        # Convert to matrix: rows = environments, cols = genotypes
        genotype_ids = list(self.genotypes.keys())
        phenotype_matrix = np.array([predictions[gid] for gid in genotype_ids]).T
        
        # Calculate variance components
        total_var = np.var(phenotype_matrix)
        
        # Main effect of genotype (average across environments)
        genotype_means = np.mean(phenotype_matrix, axis=0)
        genotype_var = np.var(genotype_means)
        
        # Main effect of environment (average across genotypes)  
        env_means = np.mean(phenotype_matrix, axis=1)
        env_var = np.var(env_means)
        
        # Interaction variance = total - main effects
        interaction_var = total_var - genotype_var - env_var
        
        return interaction_var / total_var if total_var > 0 else 0.0

    def crossing_environments(self) -> List[float]:
        """
        Find environments where genotype rankings change (crossover points).
        
        Returns:
            List of environmental values where genotypes cross
        """
        min_env, max_env = self.get_common_environment_range()
        environments = np.linspace(min_env, max_env, 100)
        
        predictions = self.evaluate_at_environments(environments)
        genotype_ids = list(self.genotypes.keys())
        
        crossings = []
        
        # Check each adjacent pair of environments
        for i in range(len(environments) - 1):
            env1, env2 = environments[i], environments[i + 1]
            
            # Get phenotype values at both environments
            vals1 = [predictions[gid][i] for gid in genotype_ids]
            vals2 = [predictions[gid][i + 1] for gid in genotype_ids]
            
            # Check if ranking changed
            ranking1 = np.argsort(vals1)
            ranking2 = np.argsort(vals2)
            
            if not np.array_equal(ranking1, ranking2):
                # Approximate crossing point
                crossings.append((env1 + env2) / 2)
        
        return crossings

    def genotype_plasticity_comparison(self) -> Dict[str, Dict]:
        """
        Compare plasticity measures across genotypes.
        
        Returns:
            Dictionary with plasticity metrics for each genotype
        """
        comparison = {}
        
        for genotype_id, norm in self.genotypes.items():
            comparison[genotype_id] = {
                'plasticity_magnitude': norm.plasticity_magnitude(),
                'plasticity_classification': norm.classify_plasticity().value,
                'slope': norm.slope(),
                'curvature': norm.curvature(),
                'environmental_optimum': norm.environmental_optimum()
            }
            
        return comparison

    def most_plastic_genotype(self) -> str:
        """
        Identify genotype with highest plasticity magnitude.
        
        Returns:
            Genotype ID with highest plasticity
        """
        plasticities = {}
        for genotype_id, norm in self.genotypes.items():
            plasticities[genotype_id] = norm.plasticity_magnitude()
            
        return max(plasticities, key=plasticities.get)

    def least_plastic_genotype(self) -> str:
        """
        Identify genotype with lowest plasticity magnitude.
        
        Returns:
            Genotype ID with lowest plasticity
        """
        plasticities = {}
        for genotype_id, norm in self.genotypes.items():
            plasticities[genotype_id] = norm.plasticity_magnitude()
            
        return min(plasticities, key=plasticities.get)

    def specialist_vs_generalist(self) -> Dict[str, str]:
        """
        Classify genotypes as specialists or generalists.
        
        Returns:
            Dictionary mapping genotype ID to 'specialist' or 'generalist'
        """
        classification = {}
        
        # Get common environment range
        min_env, max_env = self.get_common_environment_range()
        environments = np.linspace(min_env, max_env, 50)
        
        predictions = self.evaluate_at_environments(environments)
        
        for genotype_id in self.genotypes.keys():
            phenotypes = predictions[genotype_id]
            
            # Specialists: high performance in narrow range
            # Generalists: moderate performance across wide range
            
            # Calculate coefficient of variation (CV = std/mean)
            cv = np.std(phenotypes) / np.abs(np.mean(phenotypes)) if np.mean(phenotypes) != 0 else 0
            
            # High CV suggests specialization (high variation in performance)
            # Low CV suggests generalization (consistent performance)
            classification[genotype_id] = 'specialist' if cv > 0.2 else 'generalist'
            
        return classification

    def optimal_environments(self) -> Dict[str, float]:
        """
        Find optimal environment for each genotype.
        
        Returns:
            Dictionary mapping genotype ID to optimal environment
        """
        optimums = {}
        
        for genotype_id, norm in self.genotypes.items():
            optimums[genotype_id] = norm.environmental_optimum()
            
        return optimums

    def performance_correlation(self, env1: float, env2: float) -> float:
        """
        Calculate correlation of genotype performance between two environments.
        
        Args:
            env1: First environment
            env2: Second environment
            
        Returns:
            Pearson correlation coefficient
        """
        phenotypes1 = []
        phenotypes2 = []
        
        for norm in self.genotypes.values():
            phenotypes1.append(norm.predict_phenotype(env1))
            phenotypes2.append(norm.predict_phenotype(env2))
            
        if len(phenotypes1) < 2:
            return 1.0
            
        correlation, _ = stats.pearsonr(phenotypes1, phenotypes2)
        return correlation if np.isfinite(correlation) else 0.0

    def environmental_canalization(self, environments: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Measure environmental canalization (buffering) for each genotype.
        
        Args:
            environments: Environmental range to test
            
        Returns:
            Dictionary mapping genotype ID to canalization index
        """
        if environments is None:
            min_env, max_env = self.get_common_environment_range()
            environments = np.linspace(min_env, max_env, 20)
        
        canalization = {}
        
        predictions = self.evaluate_at_environments(environments)
        
        for genotype_id in self.genotypes.keys():
            phenotypes = predictions[genotype_id]
            
            # Canalization = 1 - coefficient of variation
            # Higher values = more canalized (less variable)
            mean_pheno = np.mean(phenotypes)
            cv = np.std(phenotypes) / np.abs(mean_pheno) if mean_pheno != 0 else 0
            canalization[genotype_id] = max(0, 1 - cv)
            
        return canalization

    def to_dict(self) -> Dict:
        """
        Convert G×E interaction to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        genotype_data = {}
        for genotype_id, norm in self.genotypes.items():
            genotype_data[genotype_id] = norm.to_dict()
        
        return {
            'genotypes': genotype_data,
            'environmental_variable': self.environmental_variable,
            'trait_name': self.trait_name,
            'interaction_variance': self.interaction_variance(),
            'crossing_environments': self.crossing_environments(),
            'plasticity_comparison': self.genotype_plasticity_comparison(),
            'most_plastic_genotype': self.most_plastic_genotype(),
            'least_plastic_genotype': self.least_plastic_genotype(),
            'specialist_vs_generalist': self.specialist_vs_generalist(),
            'optimal_environments': self.optimal_environments()
        }

    def to_json(self) -> str:
        """
        Convert G×E interaction to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'GxEInteraction':
        """
        Create GxEInteraction from dictionary.
        
        Args:
            data: Dictionary containing G×E data
            
        Returns:
            New GxEInteraction instance
        """
        genotypes = {}
        for genotype_id, norm_data in data['genotypes'].items():
            genotypes[genotype_id] = ReactionNorm.from_dict(norm_data)
        
        return cls(
            genotypes=genotypes,
            environmental_variable=data['environmental_variable'],
            trait_name=data['trait_name']
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'GxEInteraction':
        """
        Create GxEInteraction from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            New GxEInteraction instance  
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save_json(self, filepath: str):
        """
        Save G×E interaction to JSON file.
        
        Args:
            filepath: Path to output file
        """
        with open(filepath, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load_json(cls, filepath: str) -> 'GxEInteraction':
        """
        Load G×E interaction from JSON file.
        
        Args:
            filepath: Path to input file
            
        Returns:
            New GxEInteraction instance
        """
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())