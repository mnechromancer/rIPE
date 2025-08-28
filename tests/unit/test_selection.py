"""
Tests for Selection Mechanism (EVOL-002)

This module tests the various selection strategies and selection analysis
functionality for evolutionary simulations.
"""

import pytest
import numpy as np

from ipe.core.physiology.state import PhysiologicalState
from ipe.simulation.population import Individual
from ipe.simulation.selection import (
    TruncationSelection,
    ProportionalSelection,
    TournamentSelection,
    FrequencyDependentSelection,
    MultiTraitSelection,
    SelectionAnalyzer,
    SelectionDifferential,
)


class TestSelectionDifferential:
    """Test SelectionDifferential dataclass"""

    def test_creation(self):
        """Test creating SelectionDifferential"""
        diff = SelectionDifferential(
            trait_name="fitness",
            before_mean=0.5,
            after_mean=0.7,
            selection_differential=0.2,
            selection_intensity=1.5,
        )

        assert diff.trait_name == "fitness"
        assert diff.before_mean == 0.5
        assert diff.after_mean == 0.7
        assert diff.selection_differential == 0.2
        assert diff.selection_intensity == 1.5


class TestTruncationSelection:
    """Test truncation selection strategy"""

    @pytest.fixture
    def sample_individuals(self):
        """Create sample individuals with varying fitness"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        individuals = []
        fitness_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0]

        for i, fitness in enumerate(fitness_values):
            individual = Individual(id=i, physiological_state=state, fitness=fitness)
            individuals.append(individual)

        return individuals

    def test_creation(self):
        """Test creating truncation selection"""
        selector = TruncationSelection(survival_fraction=0.3)
        assert selector.survival_fraction == 0.3

    def test_creation_errors(self):
        """Test creation error conditions"""
        with pytest.raises(
            ValueError, match="survival_fraction must be between 0 and 1"
        ):
            TruncationSelection(survival_fraction=0.0)

        with pytest.raises(
            ValueError, match="survival_fraction must be between 0 and 1"
        ):
            TruncationSelection(survival_fraction=1.5)

    def test_selection(self, sample_individuals):
        """Test truncation selection"""
        selector = TruncationSelection(survival_fraction=0.5)
        survivors = selector.select(sample_individuals)

        assert len(survivors) == 5  # 50% of 10

        # Check that survivors have highest fitness values
        survivor_fitness = [ind.fitness for ind in survivors]
        assert min(survivor_fitness) >= 0.6  # Top 5 should have fitness >= 0.6
        assert max(survivor_fitness) == 1.0

    def test_selection_with_num_survivors(self, sample_individuals):
        """Test truncation selection with explicit num_survivors"""
        selector = TruncationSelection()
        survivors = selector.select(sample_individuals, num_survivors=3)

        assert len(survivors) == 3
        survivor_fitness = [ind.fitness for ind in survivors]
        expected_top_3 = [1.0, 0.9, 0.8]  # Top 3 fitness values
        assert sorted(survivor_fitness, reverse=True) == expected_top_3

    def test_empty_population(self):
        """Test truncation selection with empty population"""
        selector = TruncationSelection()
        survivors = selector.select([])
        assert survivors == []

    def test_single_individual(self, sample_individuals):
        """Test with single individual"""
        selector = TruncationSelection()
        survivors = selector.select([sample_individuals[0]])
        assert len(survivors) == 1
        assert survivors[0] == sample_individuals[0]


class TestProportionalSelection:
    """Test proportional selection strategy"""

    @pytest.fixture
    def sample_individuals(self):
        """Create sample individuals with varying fitness"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        individuals = []
        # Use clearly distinct fitness values
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for i, fitness in enumerate(fitness_values):
            individual = Individual(id=i, physiological_state=state, fitness=fitness)
            individuals.append(individual)

        return individuals

    def test_creation(self):
        """Test creating proportional selection"""
        selector = ProportionalSelection(replacement=False)
        assert selector.replacement is False

    def test_selection_with_positive_fitness(self, sample_individuals):
        """Test proportional selection with positive fitness"""
        selector = ProportionalSelection()
        np.random.seed(42)  # For reproducible results

        survivors = selector.select(sample_individuals, num_survivors=3)

        assert len(survivors) == 3
        # Higher fitness individuals should be more likely to be selected
        # but we can't guarantee specific outcomes due to randomness

    def test_selection_with_zero_fitness(self):
        """Test proportional selection when all fitness is zero"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        individuals = [
            Individual(id=i, physiological_state=state, fitness=0.0) for i in range(5)
        ]

        selector = ProportionalSelection()
        np.random.seed(42)
        survivors = selector.select(individuals, num_survivors=3)

        assert len(survivors) == 3  # Should handle zero fitness gracefully

    def test_selection_with_negative_fitness(self):
        """Test proportional selection with negative fitness"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        fitness_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
        individuals = []

        for i, fitness in enumerate(fitness_values):
            individual = Individual(id=i, physiological_state=state, fitness=fitness)
            individuals.append(individual)

        selector = ProportionalSelection()
        np.random.seed(42)
        survivors = selector.select(individuals, num_survivors=3)

        assert len(survivors) == 3  # Should handle negative fitness by shifting

    def test_empty_population(self):
        """Test proportional selection with empty population"""
        selector = ProportionalSelection()
        survivors = selector.select([])
        assert survivors == []


class TestTournamentSelection:
    """Test tournament selection strategy"""

    @pytest.fixture
    def sample_individuals(self):
        """Create sample individuals with varying fitness"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        individuals = []
        fitness_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        for i, fitness in enumerate(fitness_values):
            individual = Individual(id=i, physiological_state=state, fitness=fitness)
            individuals.append(individual)

        return individuals

    def test_creation(self):
        """Test creating tournament selection"""
        selector = TournamentSelection(tournament_size=5)
        assert selector.tournament_size == 5

    def test_creation_errors(self):
        """Test creation error conditions"""
        with pytest.raises(ValueError, match="tournament_size must be at least 1"):
            TournamentSelection(tournament_size=0)

    def test_selection(self, sample_individuals):
        """Test tournament selection"""
        selector = TournamentSelection(tournament_size=3)
        np.random.seed(42)

        survivors = selector.select(sample_individuals, num_survivors=3)

        assert len(survivors) == 3
        # All survivors should be among the original individuals
        survivor_ids = {ind.id for ind in survivors}
        original_ids = {ind.id for ind in sample_individuals}
        assert survivor_ids.issubset(original_ids)

    def test_tournament_size_larger_than_population(self, sample_individuals):
        """Test when tournament size > population size"""
        selector = TournamentSelection(tournament_size=10)  # Larger than population
        np.random.seed(42)

        survivors = selector.select(sample_individuals, num_survivors=2)

        assert len(survivors) == 2

    def test_empty_population(self):
        """Test tournament selection with empty population"""
        selector = TournamentSelection()
        survivors = selector.select([])
        assert survivors == []


class TestFrequencyDependentSelection:
    """Test frequency-dependent selection strategy"""

    @pytest.fixture
    def sample_individuals(self):
        """Create individuals with discrete traits"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        individuals = []

        # Create individuals with different traits (stored in genetic_values)
        traits = ["A", "A", "B", "B", "B", "C"]
        for i, trait in enumerate(traits):
            individual = Individual(
                id=i,
                physiological_state=state,
                fitness=0.5,
                genetic_values={"trait": trait},
            )
            individuals.append(individual)

        return individuals

    def test_selection(self, sample_individuals):
        """Test frequency-dependent selection"""

        def trait_extractor(ind):
            return ind.genetic_values["trait"]

        def fitness_function(trait, frequencies):
            # Rare-advantage: fitness inversely proportional to frequency
            return 1.0 / frequencies[trait]

        selector = FrequencyDependentSelection(trait_extractor, fitness_function)
        np.random.seed(42)

        survivors = selector.select(sample_individuals, num_survivors=3)

        assert len(survivors) == 3
        # Check that fitness was recalculated
        for ind in sample_individuals:
            assert ind.fitness != 0.5  # Should have changed from initial value

    def test_empty_population(self):
        """Test frequency-dependent selection with empty population"""

        def trait_extractor(ind):
            return ind.genetic_values.get("trait", "default")

        def fitness_function(trait, frequencies):
            return 1.0

        selector = FrequencyDependentSelection(trait_extractor, fitness_function)
        survivors = selector.select([])
        assert survivors == []


class TestMultiTraitSelection:
    """Test multi-trait selection strategy"""

    @pytest.fixture
    def sample_individuals(self):
        """Create individuals with multiple traits"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        individuals = []

        trait_combinations = [
            {"strength": 0.8, "speed": 0.3},
            {"strength": 0.5, "speed": 0.7},
            {"strength": 0.9, "speed": 0.2},
            {"strength": 0.4, "speed": 0.9},
            {"strength": 0.6, "speed": 0.6},
        ]

        for i, traits in enumerate(trait_combinations):
            individual = Individual(
                id=i, physiological_state=state, fitness=0.5, genetic_values=traits
            )
            individuals.append(individual)

        return individuals

    def test_creation(self):
        """Test creating multi-trait selection"""
        extractors = {
            "strength": lambda ind: ind.genetic_values["strength"],
            "speed": lambda ind: ind.genetic_values["speed"],
        }
        weights = {"strength": 0.7, "speed": 0.3}

        selector = MultiTraitSelection(extractors, weights)
        assert selector.trait_extractors == extractors
        assert selector.trait_weights == weights

    def test_selection_equal_weights(self, sample_individuals):
        """Test multi-trait selection with equal weights"""
        extractors = {
            "strength": lambda ind: ind.genetic_values["strength"],
            "speed": lambda ind: ind.genetic_values["speed"],
        }

        selector = MultiTraitSelection(extractors)
        survivors = selector.select(sample_individuals, num_survivors=3)

        assert len(survivors) == 3

        # Check that composite fitness was calculated
        for ind in sample_individuals:
            expected_fitness = (
                ind.genetic_values["strength"] + ind.genetic_values["speed"]
            ) / 2
            assert abs(ind.fitness - expected_fitness) < 1e-10

    def test_selection_weighted(self, sample_individuals):
        """Test multi-trait selection with custom weights"""
        extractors = {
            "strength": lambda ind: ind.genetic_values["strength"],
            "speed": lambda ind: ind.genetic_values["speed"],
        }
        weights = {"strength": 0.8, "speed": 0.2}

        selector = MultiTraitSelection(extractors, weights)
        survivors = selector.select(sample_individuals, num_survivors=2)

        assert len(survivors) == 2

        # Check composite fitness calculation
        for ind in sample_individuals:
            expected_fitness = (
                ind.genetic_values["strength"] * 0.8 + ind.genetic_values["speed"] * 0.2
            )
            assert abs(ind.fitness - expected_fitness) < 1e-10

    def test_empty_population(self):
        """Test multi-trait selection with empty population"""
        extractors = {"trait": lambda ind: ind.fitness}
        selector = MultiTraitSelection(extractors)
        survivors = selector.select([])
        assert survivors == []


class TestSelectionAnalyzer:
    """Test selection analysis functionality"""

    @pytest.fixture
    def before_after_populations(self):
        """Create before and after populations for analysis"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)

        # Before selection: 10 individuals with fitness 0.1 to 1.0
        before = []
        fitness_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i, fitness in enumerate(fitness_values):
            individual = Individual(id=i, physiological_state=state, fitness=fitness)
            before.append(individual)

        # After selection: top 5 individuals
        after = before[5:]  # fitness 0.6 to 1.0

        return before, after

    def test_analyze_selection(self, before_after_populations):
        """Test selection analysis"""
        before, after = before_after_populations

        stats = SelectionAnalyzer.analyze_selection(before, after)

        assert stats["survival_rate"] == 0.5  # 5 out of 10 survived
        assert stats["before_mean_fitness"] == 0.55  # mean of 0.1-1.0
        assert abs(stats["after_mean_fitness"] - 0.8) < 1e-10  # mean of 0.6-1.0
        assert stats["fitness_change"] == pytest.approx(0.25, rel=1e-3)  # 0.8 - 0.55

        # Selection intensity should be positive (directional selection)
        assert stats["selection_intensity"] > 0

    def test_analyze_empty_populations(self):
        """Test analysis with empty populations"""
        stats = SelectionAnalyzer.analyze_selection([], [])

        assert stats["selection_intensity"] == 0.0
        assert stats["survival_rate"] == 0.0
        assert stats["fitness_change"] == 0.0
        assert stats["variance_change"] == 0.0

    def test_analyze_no_selection(self):
        """Test analysis when no selection occurred"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        individuals = [
            Individual(id=i, physiological_state=state, fitness=0.5) for i in range(5)
        ]

        stats = SelectionAnalyzer.analyze_selection(individuals, individuals)

        assert stats["survival_rate"] == 1.0  # All survived
        assert stats["fitness_change"] == 0.0  # No change in mean fitness
        assert stats["selection_intensity"] == 0.0  # No selection


class TestSelectionStrategyIntegration:
    """Test integration between selection strategies and base class"""

    @pytest.fixture
    def sample_individuals(self):
        """Create sample individuals for testing"""
        state = PhysiologicalState(po2=15.0, temperature=25.0, altitude=1000.0)
        individuals = []
        fitness_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        for i, fitness in enumerate(fitness_values):
            individual = Individual(id=i, physiological_state=state, fitness=fitness)
            individuals.append(individual)

        return individuals

    def test_selection_differential_calculation(self, sample_individuals):
        """Test selection differential calculation"""
        selector = TruncationSelection(survival_fraction=0.6)

        # Before selection
        before = sample_individuals.copy()

        # After selection
        after = selector.select(before)

        # Calculate selection differential
        diff = selector.calculate_selection_differential(
            before, after, lambda ind: ind.fitness, "fitness"
        )

        assert diff.trait_name == "fitness"
        assert diff.before_mean == 0.5  # Mean of [0.1, 0.3, 0.5, 0.7, 0.9]
        assert diff.after_mean > diff.before_mean  # Selection should increase mean
        assert diff.selection_differential > 0  # Positive selection differential
        assert diff.selection_intensity > 0  # Positive selection intensity
