"""
SQLAlchemy models for IPE (Integrated Phenotypic Evolution) platform.
Includes models for organisms, simulations, environmental data, and temporal
series data.
Optimized for PostgreSQL with TimescaleDB for time-series performance.
"""

import uuid

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    JSON,
    ForeignKey,
    Index,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

Base = declarative_base()


class TimestampMixin:
    """Mixin for adding created_at and updated_at timestamps."""

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Organism(Base, TimestampMixin):
    """
    Core organism model representing a phenotype in the evolution simulation.
    """

    __tablename__ = "organisms"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    species_id = Column(String(100), nullable=False, index=True)
    generation = Column(Integer, nullable=False, index=True)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("organisms.id"), nullable=True)

    # Physiological state vector
    body_mass = Column(Float, nullable=False)
    metabolic_rate = Column(Float, nullable=False)
    oxygen_consumption = Column(Float, nullable=False)
    temperature_tolerance = Column(ARRAY(Float), nullable=False)  # [min, max]
    plasticity_coefficients = Column(JSON, nullable=False)

    # Genetic information
    genotype = Column(JSON, nullable=False)
    mutation_count = Column(Integer, default=0, nullable=False)
    fitness_score = Column(Float, nullable=True)

    # Simulation metadata
    simulation_id = Column(
        UUID(as_uuid=True), ForeignKey("simulations.id"), nullable=False
    )
    is_alive = Column(Boolean, default=True, nullable=False)

    # Relationships
    parent = relationship("Organism", remote_side=[id], backref="offspring")
    simulation = relationship("Simulation", back_populates="organisms")
    measurements = relationship(
        "PhysiologyMeasurement", back_populates="organism", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_organism_species_generation", "species_id", "generation"),
        Index("idx_organism_simulation", "simulation_id"),
        CheckConstraint("body_mass > 0", name="positive_body_mass"),
        CheckConstraint("metabolic_rate > 0", name="positive_metabolic_rate"),
    )

    @validates("plasticity_coefficients")
    def validate_plasticity(self, key, value):
        """Validate plasticity coefficients structure."""
        required_keys = [
            "thermal_sensitivity",
            "hypoxia_tolerance",
            "metabolic_flexibility",
        ]
        if not all(k in value for k in required_keys):
            raise ValueError(f"Plasticity coefficients must include: {required_keys}")
        return value


class Simulation(Base, TimestampMixin):
    """
    Simulation run configuration and metadata.
    """

    __tablename__ = "simulations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    # Simulation parameters
    initial_population_size = Column(Integer, nullable=False)
    generations = Column(Integer, nullable=False)
    mutation_rate = Column(Float, nullable=False)
    selection_pressure = Column(Float, nullable=False)

    # Environmental scenarios
    environment_config = Column(JSON, nullable=False)

    # Status and progress
    status = Column(String(50), nullable=False, default="initialized")
    current_generation = Column(Integer, default=0, nullable=False)
    completion_percentage = Column(Float, default=0.0, nullable=False)

    # Results summary
    final_population_size = Column(Integer, nullable=True)
    evolutionary_trajectory = Column(JSON, nullable=True)

    # Relationships
    organisms = relationship(
        "Organism", back_populates="simulation", cascade="all, delete-orphan"
    )
    environments = relationship(
        "EnvironmentalCondition",
        back_populates="simulation",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_simulation_status", "status"),
        CheckConstraint("initial_population_size > 0", name="positive_population"),
        CheckConstraint("generations > 0", name="positive_generations"),
        CheckConstraint("mutation_rate >= 0", name="non_negative_mutation_rate"),
    )


class EnvironmentalCondition(Base, TimestampMixin):
    """
    Time-series environmental data for simulations.
    Optimized for TimescaleDB hypertables.
    """

    __tablename__ = "environmental_conditions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    simulation_id = Column(
        UUID(as_uuid=True), ForeignKey("simulations.id"), nullable=False
    )
    timestamp = Column(DateTime(timezone=True), nullable=False)
    generation = Column(Integer, nullable=False)

    # Environmental parameters
    temperature = Column(Float, nullable=False)  # Celsius
    oxygen_partial_pressure = Column(Float, nullable=False)  # kPa
    altitude = Column(Float, nullable=False)  # meters
    humidity = Column(Float, nullable=False)  # percentage
    pressure = Column(Float, nullable=False)  # kPa

    # Additional environmental factors
    resource_availability = Column(Float, nullable=False)  # 0-1 scale
    predation_pressure = Column(Float, nullable=False)  # 0-1 scale
    competition_index = Column(Float, nullable=False)  # 0-1 scale

    # Relationships
    simulation = relationship("Simulation", back_populates="environments")

    __table_args__ = (
        Index("idx_env_simulation_time", "simulation_id", "timestamp"),
        Index("idx_env_generation", "generation"),
        CheckConstraint(
            "temperature >= -50 AND temperature <= 70", name="realistic_temperature"
        ),
        CheckConstraint("oxygen_partial_pressure >= 0", name="positive_oxygen"),
        CheckConstraint("altitude >= 0", name="positive_altitude"),
        CheckConstraint("humidity >= 0 AND humidity <= 100", name="valid_humidity"),
    )


class PhysiologyMeasurement(Base, TimestampMixin):
    """
    Time-series physiological measurements from organisms.
    Supports both simulated and experimental data integration.
    """

    __tablename__ = "physiology_measurements"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organism_id = Column(UUID(as_uuid=True), ForeignKey("organisms.id"), nullable=False)
    measurement_time = Column(DateTime(timezone=True), nullable=False)
    measurement_type = Column(String(100), nullable=False)

    # Core physiological parameters
    oxygen_consumption_rate = Column(Float, nullable=True)  # ml O2/min/g
    carbon_dioxide_production = Column(Float, nullable=True)  # ml CO2/min/g
    respiratory_quotient = Column(Float, nullable=True)
    heart_rate = Column(Float, nullable=True)  # beats/min
    body_temperature = Column(Float, nullable=True)  # Celsius

    # Environmental conditions during measurement
    ambient_temperature = Column(Float, nullable=True)
    ambient_oxygen = Column(Float, nullable=True)
    ambient_pressure = Column(Float, nullable=True)

    # Metadata
    data_source = Column(
        String(100), nullable=False
    )  # 'simulation', 'respirometry', 'field'
    quality_score = Column(Float, nullable=True)  # 0-1 data quality indicator
    experimental_conditions = Column(JSON, nullable=True)

    # Relationships
    organism = relationship("Organism", back_populates="measurements")

    __table_args__ = (
        Index("idx_physiology_organism_time", "organism_id", "measurement_time"),
        Index("idx_physiology_type", "measurement_type"),
        Index("idx_physiology_source", "data_source"),
        CheckConstraint(
            "quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)",
            name="valid_quality_score",
        ),
    )


class GeneticMarker(Base, TimestampMixin):
    """
    Genetic markers and their effects on phenotypes.
    """

    __tablename__ = "genetic_markers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    marker_name = Column(String(100), nullable=False, unique=True)
    chromosome = Column(String(50), nullable=True)
    position = Column(Integer, nullable=True)

    # Effect on phenotype
    phenotype_effect = Column(JSON, nullable=False)
    effect_size = Column(Float, nullable=False)
    dominance = Column(Float, default=0.0, nullable=False)  # -1 to 1

    # Population genetics
    allele_frequency = Column(Float, nullable=True)
    selection_coefficient = Column(Float, default=0.0, nullable=False)

    # Metadata
    discovery_method = Column(String(100), nullable=True)
    validation_status = Column(String(50), default="predicted", nullable=False)

    __table_args__ = (
        Index("idx_marker_name", "marker_name"),
        CheckConstraint("effect_size >= 0", name="positive_effect_size"),
        CheckConstraint("dominance >= -1 AND dominance <= 1", name="valid_dominance"),
        CheckConstraint(
            "allele_frequency IS NULL OR "
            "(allele_frequency >= 0 AND allele_frequency <= 1)",
            name="valid_allele_frequency",
        ),
    )


class ExperimentalData(Base, TimestampMixin):
    """
    Integration table for real experimental data from lab measurements.
    """

    __tablename__ = "experimental_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id = Column(String(100), nullable=False)
    species = Column(String(100), nullable=False)
    individual_id = Column(String(100), nullable=True)

    # Sample metadata
    sample_date = Column(DateTime(timezone=True), nullable=False)
    collection_site = Column(String(200), nullable=True)
    elevation = Column(Float, nullable=True)
    coordinates = Column(JSON, nullable=True)  # {lat, lng}

    # Measurement data
    measurement_data = Column(JSON, nullable=False)
    data_type = Column(
        String(100), nullable=False
    )  # 'respirometry', 'rnaseq', 'morphology'

    # Quality and processing
    processing_pipeline = Column(String(200), nullable=True)
    quality_flags = Column(ARRAY(String), nullable=True)
    notes = Column(Text, nullable=True)

    # Integration status
    integration_status = Column(String(50), default="raw", nullable=False)
    linked_organism_id = Column(
        UUID(as_uuid=True), ForeignKey("organisms.id"), nullable=True
    )

    # Relationships
    linked_organism = relationship("Organism")

    __table_args__ = (
        Index("idx_experiment_id", "experiment_id"),
        Index("idx_experiment_species", "species"),
        Index("idx_experiment_type", "data_type"),
        Index("idx_experiment_date", "sample_date"),
    )


class EvolutionaryEvent(Base, TimestampMixin):
    """
    Significant evolutionary events and transitions in simulations.
    """

    __tablename__ = "evolutionary_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    simulation_id = Column(
        UUID(as_uuid=True), ForeignKey("simulations.id"), nullable=False
    )
    generation = Column(Integer, nullable=False)
    event_time = Column(DateTime(timezone=True), nullable=False)

    # Event details
    event_type = Column(
        String(100), nullable=False
    )  # 'speciation', 'extinction', 'adaptation'
    description = Column(Text, nullable=False)
    affected_lineages = Column(ARRAY(UUID), nullable=True)

    # Quantitative measures
    magnitude = Column(Float, nullable=True)  # Effect size
    fitness_change = Column(Float, nullable=True)
    population_impact = Column(Float, nullable=True)  # Percentage affected

    # Environmental context
    environmental_trigger = Column(JSON, nullable=True)

    # Relationships
    simulation = relationship("Simulation")

    __table_args__ = (
        Index("idx_event_simulation_generation", "simulation_id", "generation"),
        Index("idx_event_type", "event_type"),
    )


# Create TimescaleDB hypertables for time-series data
# These would be executed after table creation via SQL or migration scripts
HYPERTABLE_CONFIGS = [
    {
        "table": "environmental_conditions",
        "time_column": "timestamp",
        "chunk_time_interval": "1 day",
    },
    {
        "table": "physiology_measurements",
        "time_column": "measurement_time",
        "chunk_time_interval": "1 day",
    },
    {
        "table": "evolutionary_events",
        "time_column": "event_time",
        "chunk_time_interval": "1 hour",
    },
]