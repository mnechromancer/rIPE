"""
Tests for Mutation Model (EVOL-003)

This module tests the mutation mechanisms, genetic architecture,
and evolutionary dynamics of mutation rates.
"""

import pytest
import numpy as np
from typing import Dict, List

from ipe.core.physiology.state import PhysiologicalState
from ipe.simulation.population import Individual
from ipe.simulation.genetic_architecture import (
    GeneticLocus, TraitArchitecture, TraitType, GeneticArchitecture,
    create_default_architecture
)
from ipe.simulation.mutation import (
    MutationParameters, MutationStrategy, GaussianMutation,
    PleiotopicMutation, MutationEngine, create_default_mutation_engine
)


class TestGeneticLocus:
    """Test GeneticLocus functionality"""
    
    def test_basic_creation(self):
        """Test creating a genetic locus"""
        locus = GeneticLocus(
            name="test_locus",
            trait_effects={"trait1": 0.5, "trait2": 0.3},
            mutation_rate=1e-4,
            allelic_variance=0.01
        )
        
        assert locus.name == "test_locus"
        assert locus.trait_effects == {"trait1": 0.5, "trait2": 0.3}
        assert locus.mutation_rate == 1e-4
        assert locus.allelic_variance == 0.01
        assert locus.dominance_coefficient == 0.0
        assert locus.epistatic_interactions == {}
    
    def test_creation_errors(self):
        """Test locus creation error conditions"""
        with pytest.raises(ValueError, match="Mutation rate must be non-negative"):
            GeneticLocus("test", {}, mutation_rate=-1.0)
        
        with pytest.raises(ValueError, match="Allelic variance must be non-negative"):
            GeneticLocus("test", {}, allelic_variance=-1.0)
        
        with pytest.raises(ValueError, match="Dominance coefficient must be between -1 and 1"):
            GeneticLocus("test", {}, dominance_coefficient=2.0)


class TestTraitArchitecture:
    """Test TraitArchitecture functionality"""
    
    def test_basic_creation(self):
        """Test creating trait architecture"""
        trait = TraitArchitecture(
            name="test_trait",
            trait_type=TraitType.CONTINUOUS,
            contributing_loci=["locus1", "locus2"],
            heritability=0.6
        )
        
        assert trait.name == "test_trait"
        assert trait.trait_type == TraitType.CONTINUOUS
        assert trait.contributing_loci == ["locus1", "locus2"]
        assert trait.heritability == 0.6
        assert trait.environmental_variance == 0.01
    
    def test_creation_errors(self):
        """Test trait architecture creation errors"""
        with pytest.raises(ValueError, match="Heritability must be between 0 and 1"):
            TraitArchitecture("test", TraitType.CONTINUOUS, [], heritability=1.5)
        
        with pytest.raises(ValueError, match="Environmental variance must be non-negative"):
            TraitArchitecture("test", TraitType.CONTINUOUS, [], environmental_variance=-1.0)


class TestGeneticArchitecture:
    """Test GeneticArchitecture functionality"""
    
    @pytest.fixture
    def sample_architecture(self):
        """Create sample genetic architecture"""
        arch = GeneticArchitecture()
        
        # Add loci
        loci = [
            GeneticLocus("locus1", {"trait1": 0.5, "trait2": 0.2}),
            GeneticLocus("locus2", {"trait1": 0.3, "trait2": 0.7}),
            GeneticLocus("locus3", {"trait2": 0.4})
        ]
        
        for locus in loci:
            arch.add_locus(locus)
        
        # Add traits
        traits = [
            TraitArchitecture("trait1", TraitType.CONTINUOUS, ["locus1", "locus2"]),
            TraitArchitecture("trait2", TraitType.CONTINUOUS, ["locus1", "locus2", "locus3"])
        ]
        
        for trait in traits:
            arch.add_trait(trait)
        
        return arch
    
    def test_architecture_creation(self):
        """Test creating empty architecture"""
        arch = GeneticArchitecture()
        assert len(arch.loci) == 0
        assert len(arch.traits) == 0
        assert arch.pleiotropy_matrix is None
    
    def test_add_locus_and_trait(self, sample_architecture):
        """Test adding loci and traits"""
        assert len(sample_architecture.loci) == 3
        assert len(sample_architecture.traits) == 2
        assert sample_architecture.pleiotropy_matrix is not None
        assert sample_architecture.pleiotropy_matrix.shape == (3, 2)  # 3 loci, 2 traits
    
    def test_pleiotropy_matrix(self, sample_architecture):
        """Test pleiotropy matrix construction"""
        matrix = sample_architecture.pleiotropy_matrix
        
        # Check specific values
        assert matrix[0, 0] == 0.5  # locus1 -> trait1
        assert matrix[0, 1] == 0.2  # locus1 -> trait2
        assert matrix[1, 0] == 0.3  # locus2 -> trait1
        assert matrix[1, 1] == 0.7  # locus2 -> trait2
        assert matrix[2, 0] == 0.0  # locus3 -> trait1 (no effect)
        assert matrix[2, 1] == 0.4  # locus3 -> trait2
    
    def test_calculate_breeding_values(self, sample_architecture):
        """Test breeding value calculation"""
        genotype = {"locus1": 1.0, "locus2": 0.5, "locus3": -0.5}
        
        breeding_values = sample_architecture.calculate_breeding_values(genotype)
        
        # trait1: 1.0*0.5 + 0.5*0.3 = 0.5 + 0.15 = 0.65
        assert abs(breeding_values["trait1"] - 0.65) < 1e-10
        
        # trait2: 1.0*0.2 + 0.5*0.7 + (-0.5)*0.4 = 0.2 + 0.35 - 0.2 = 0.35
        assert abs(breeding_values["trait2"] - 0.35) < 1e-10
    
    def test_calculate_phenotype(self, sample_architecture):
        """Test phenotype calculation with environmental effects"""
        genotype = {"locus1": 1.0, "locus2": 0.0, "locus3": 0.0}
        
        # Set random seed for reproducible environmental effects
        np.random.seed(42)
        phenotypes = sample_architecture.calculate_phenotype(genotype)
        
        # Should have phenotype for each trait
        assert "trait1" in phenotypes
        assert "trait2" in phenotypes
        
        # Phenotype should be breeding value + environmental noise
        # We can't test exact values due to randomness, but should be approximately correct
        assert abs(phenotypes["trait1"] - 0.5) < 1.0  # Should be around 0.5 ± environmental variance
        assert abs(phenotypes["trait2"] - 0.2) < 1.0  # Should be around 0.2 ± environmental variance
    
    def test_genetic_correlations(self, sample_architecture):
        """Test genetic correlation calculation"""
        correlations = sample_architecture.get_genetic_correlations()
        
        assert correlations.shape == (2, 2)  # 2 traits
        
        # Diagonal should be 1 (trait correlated with itself)
        assert abs(correlations[0, 0] - 1.0) < 1e-10
        assert abs(correlations[1, 1] - 1.0) < 1e-10
        
        # Off-diagonal should be symmetric
        assert abs(correlations[0, 1] - correlations[1, 0]) < 1e-10
    
    def test_serialization(self, sample_architecture):
        """Test architecture serialization"""
        serialized = sample_architecture.serialize()
        
        assert "loci" in serialized
        assert "traits" in serialized
        assert len(serialized["loci"]) == 3
        assert len(serialized["traits"]) == 2
        
        # Check specific locus data
        locus1_data = serialized["loci"]["locus1"]
        assert locus1_data["trait_effects"] == {"trait1": 0.5, "trait2": 0.2}


class TestDefaultArchitecture:
    """Test default genetic architecture creation"""
    
    def test_create_default_architecture(self):
        """Test creating default architecture"""
        arch = create_default_architecture()
        
        assert len(arch.loci) > 0
        assert len(arch.traits) > 0
        assert arch.pleiotropy_matrix is not None
        
        # Check that basic physiological traits are present
        trait_names = list(arch.traits.keys())
        expected_traits = ["heart_mass", "cardiac_output", "lung_volume", "hematocrit"]
        
        for expected in expected_traits:
            assert expected in trait_names


class TestMutationParameters:
    """Test MutationParameters dataclass"""
    
    def test_basic_creation(self):
        """Test creating mutation parameters"""
        params = MutationParameters(
            base_mutation_rate=1e-3,
            mutational_variance=0.02,
            pleiotropic_correlation=0.3
        )
        
        assert params.base_mutation_rate == 1e-3
        assert params.mutational_variance == 0.02
        assert params.pleiotropic_correlation == 0.3
    
    def test_creation_errors(self):
        """Test parameter validation errors"""
        with pytest.raises(ValueError, match="Base mutation rate must be non-negative"):
            MutationParameters(base_mutation_rate=-1.0)
        
        with pytest.raises(ValueError, match="Mutational variance must be non-negative"):
            MutationParameters(mutational_variance=-1.0)
        
        with pytest.raises(ValueError, match="Pleiotropic correlation must be between 0 and 1"):
            MutationParameters(pleiotropic_correlation=1.5)


class TestGaussianMutation:
    """Test Gaussian mutation strategy"""
    
    @pytest.fixture
    def sample_individual(self):
        """Create sample individual for mutation"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        return Individual(
            id=1,
            physiological_state=state,
            fitness=0.8,
            genetic_values={"locus1": 1.0, "locus2": 0.5}
        )
    
    @pytest.fixture
    def sample_architecture(self):
        """Create sample architecture"""
        arch = GeneticArchitecture()
        loci = [
            GeneticLocus("locus1", {"trait1": 0.5}, mutation_rate=1.0, allelic_variance=0.1),  # High rate for testing
            GeneticLocus("locus2", {"trait1": 0.3}, mutation_rate=0.0, allelic_variance=0.1)   # No mutation
        ]
        for locus in loci:
            arch.add_locus(locus)
        return arch
    
    def test_mutation(self, sample_individual, sample_architecture):
        """Test Gaussian mutation"""
        params = MutationParameters()
        mutator = GaussianMutation(params)
        
        np.random.seed(42)  # For reproducible results
        mutated = mutator.mutate(sample_individual, sample_architecture)
        
        # Individual should be different object
        assert mutated is not sample_individual
        assert mutated.id == sample_individual.id
        
        # Locus1 should be mutated (mutation_rate=1.0), locus2 should not (rate=0.0)
        assert mutated.genetic_values["locus1"] != 1.0  # Should have changed
        assert mutated.genetic_values["locus2"] == 0.5  # Should be unchanged
    
    def test_no_genetic_values(self, sample_architecture):
        """Test mutation on individual with no genetic values"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        individual = Individual(id=1, physiological_state=state)
        
        params = MutationParameters()
        mutator = GaussianMutation(params)
        
        mutated = mutator.mutate(individual, sample_architecture)
        assert mutated.genetic_values is not None


class TestPleiotopicMutation:
    """Test pleiotropic mutation strategy"""
    
    @pytest.fixture
    def sample_individual(self):
        """Create sample individual"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        return Individual(
            id=1,
            physiological_state=state,
            genetic_values={"locus1": 1.0, "locus2": 0.5}
        )
    
    @pytest.fixture  
    def sample_architecture(self):
        """Create architecture with pleiotropy"""
        arch = GeneticArchitecture()
        loci = [
            GeneticLocus("locus1", {"trait1": 0.5, "trait2": 0.3}, mutation_rate=1.0),
            GeneticLocus("locus2", {"trait1": 0.2, "trait2": 0.8}, mutation_rate=1.0)
        ]
        traits = [
            TraitArchitecture("trait1", TraitType.CONTINUOUS, ["locus1", "locus2"]),
            TraitArchitecture("trait2", TraitType.CONTINUOUS, ["locus1", "locus2"])
        ]
        
        for locus in loci:
            arch.add_locus(locus)
        for trait in traits:
            arch.add_trait(trait)
            
        return arch
    
    def test_pleiotropic_mutation(self, sample_individual, sample_architecture):
        """Test pleiotropic mutation effects"""
        params = MutationParameters()
        mutator = PleiotopicMutation(params)
        
        np.random.seed(42)
        mutated = mutator.mutate(sample_individual, sample_architecture)
        
        # Should have mutations at both loci
        assert mutated.genetic_values["locus1"] != 1.0
        assert mutated.genetic_values["locus2"] != 0.5


class TestMutationEngine:
    """Test MutationEngine functionality"""
    
    @pytest.fixture
    def sample_population(self):
        """Create sample population for testing"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        individuals = []
        
        for i in range(5):
            individual = Individual(
                id=i,
                physiological_state=state,
                genetic_values={"locus1": np.random.normal(0, 1), "locus2": np.random.normal(0, 1)}
            )
            individuals.append(individual)
        
        return individuals
    
    def test_engine_creation(self):
        """Test creating mutation engine"""
        arch = create_default_architecture()
        params = MutationParameters()
        engine = MutationEngine(params, arch)
        
        assert engine.parameters == params
        assert engine.architecture == arch
        assert len(engine.current_mutation_rates) > 0
    
    def test_mutate_individual(self, sample_population):
        """Test mutating single individual"""
        arch = create_default_architecture()
        engine = create_default_mutation_engine(arch)
        
        original = sample_population[0]
        mutated = engine.mutate_individual(original)
        
        assert mutated is not original
        assert mutated.id == original.id
    
    def test_mutate_population(self, sample_population):
        """Test mutating entire population"""
        arch = create_default_architecture()
        engine = create_default_mutation_engine(arch)
        
        mutated_pop = engine.mutate_population(sample_population)
        
        assert len(mutated_pop) == len(sample_population)
        # Each individual should be different object
        for orig, mut in zip(sample_population, mutated_pop):
            assert mut is not orig
    
    def test_evolve_mutation_rates(self, sample_population):
        """Test evolution of mutation rates"""
        arch = create_default_architecture()
        params = MutationParameters(mutation_rate_heritability=0.5)
        engine = MutationEngine(params, arch)
        
        original_rates = engine.current_mutation_rates.copy()
        
        # Run mutation rate evolution
        np.random.seed(42)
        engine.evolve_mutation_rates(sample_population)
        
        # Rates should potentially change (depending on random effects)
        # At least verify the method doesn't crash and rates remain non-negative
        for rate in engine.current_mutation_rates.values():
            assert rate >= 0
    
    def test_maintain_standing_variation(self, sample_population):
        """Test maintaining standing variation"""
        arch = create_default_architecture()
        engine = create_default_mutation_engine(arch)
        
        # Create population with low variance
        for ind in sample_population:
            ind.genetic_values = {"locus1": 0.001, "locus2": 0.001}  # Very low variance
        
        np.random.seed(42)
        maintained_pop = engine.maintain_standing_variation(sample_population, target_variance=0.01)
        
        assert len(maintained_pop) == len(sample_population)
        # Variance should potentially increase (method should run without error)
    
    def test_get_statistics(self):
        """Test getting mutation statistics"""
        arch = create_default_architecture()
        engine = create_default_mutation_engine(arch)
        
        stats = engine.get_mutation_statistics()
        
        assert "current_mutation_rates" in stats
        assert "mean_mutation_rate" in stats
        assert "mutation_rate_variance" in stats
        assert "total_mutations" in stats
    
    def test_reset_statistics(self):
        """Test resetting statistics"""
        arch = create_default_architecture()
        engine = create_default_mutation_engine(arch)
        
        # Add some dummy statistics
        engine.mutation_counts["locus1"] = 5
        engine.effect_sizes["locus1"] = [0.1, 0.2, 0.3]
        
        engine.reset_statistics()
        
        assert len(engine.mutation_counts) == 0
        assert len(engine.effect_sizes) == 0


class TestDefaultMutationEngine:
    """Test default mutation engine creation"""
    
    def test_create_default_engine(self):
        """Test creating default mutation engine"""
        arch = create_default_architecture()
        engine = create_default_mutation_engine(arch)
        
        assert engine.parameters is not None
        assert engine.architecture == arch
        assert engine.strategy is not None
        assert isinstance(engine.strategy, PleiotopicMutation)