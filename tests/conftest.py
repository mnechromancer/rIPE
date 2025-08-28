"""
pytest configuration and shared fixtures for the IPE test suite.

This module provides common test fixtures and configuration for all test modules,
supporting the TEST-001 requirement for comprehensive unit testing.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_data_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_phenotype_data():
    """Provide sample phenotype data for testing."""
    return {
        "organism_id": "test_001",
        "mass": 2.5,  # kg
        "temperature": 15.0,  # °C
        "o2_consumption": 0.8,  # ml/min/g
        "co2_production": 0.6,  # ml/min/g
        "metabolic_rate": 1.2,  # W/kg
        "traits": {
            "body_size": 0.15,  # m
            "organ_ratios": {
                "heart": 0.005,
                "liver": 0.03,
                "kidney": 0.008,
                "brain": 0.02,
            },
        },
    }


@pytest.fixture
def sample_environmental_data():
    """Provide sample environmental conditions for testing."""
    return {
        "temperature": np.array([5, 10, 15, 20, 25]),  # °C
        "oxygen": np.array([21, 18, 15, 12, 9]),  # % O2
        "altitude": np.array([0, 1000, 2000, 3000, 4000]),  # m
        "pressure": np.array([101.3, 89.9, 79.5, 69.7, 61.7]),  # kPa
    }


@pytest.fixture
def sample_genomic_data():
    """Provide sample genomic data for testing."""
    return {
        "gene_expression": {
            "HIF1A": 2.5,  # Hypoxia-inducible factor
            "EPO": 3.2,  # Erythropoietin
            "VEGF": 1.8,  # Vascular endothelial growth factor
            "EPAS1": 2.1,  # Endothelial PAS domain protein 1
        },
        "variants": [
            {"gene": "EPAS1", "position": 12345, "effect": "missense"},
            {"gene": "EGLN1", "position": 67890, "effect": "synonymous"},
        ],
        "allele_frequencies": {"EPAS1_variant": 0.85, "EGLN1_variant": 0.12},
    }


@pytest.fixture
def performance_benchmarks():
    """Define performance benchmarks for testing."""
    return {
        "simulation_runtime": {"max_seconds": 30.0, "target_seconds": 10.0},
        "memory_usage": {"max_mb": 500, "target_mb": 200},
        "convergence": {"max_iterations": 1000, "tolerance": 1e-6},
    }


@pytest.fixture
def mock_api_responses():
    """Provide mock API responses for testing."""
    return {
        "health_check": {"status": "ok", "timestamp": "2024-01-01T00:00:00Z"},
        "simulation_result": {
            "id": "sim_123",
            "status": "completed",
            "results": {
                "fitness": 0.85,
                "traits": {"body_size": 0.16, "organ_mass": 0.025},
                "generations": 500,
            },
        },
        "error_response": {
            "error": "Invalid parameters",
            "code": 400,
            "details": "Temperature must be between -20 and 50°C",
        },
    }


# Configure pytest settings
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "benchmark: mark test as performance benchmark")
    config.addinivalue_line("markers", "validation: mark test as scientific validation")


# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark validation tests
        if "validation" in str(item.fspath):
            item.add_marker(pytest.mark.validation)

        # Mark slow tests based on name patterns
        if any(
            keyword in item.name.lower()
            for keyword in ["benchmark", "performance", "stress"]
        ):
            item.add_marker(pytest.mark.slow)
