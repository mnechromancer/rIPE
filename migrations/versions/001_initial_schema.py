"""Initial database schema for IPE platform

Revision ID: 001_initial_schema
Revises:
Create Date: 2024-08-28 06:30:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable TimescaleDB extension
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))

    # Create simulations table
    op.create_table(
        "simulations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("initial_population_size", sa.Integer(), nullable=False),
        sa.Column("generations", sa.Integer(), nullable=False),
        sa.Column("mutation_rate", sa.Float(), nullable=False),
        sa.Column("selection_pressure", sa.Float(), nullable=False),
        sa.Column("environment_config", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("current_generation", sa.Integer(), nullable=False),
        sa.Column("completion_percentage", sa.Float(), nullable=False),
        sa.Column("final_population_size", sa.Integer(), nullable=True),
        sa.Column("evolutionary_trajectory", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("initial_population_size > 0", name="positive_population"),
        sa.CheckConstraint("generations > 0", name="positive_generations"),
        sa.CheckConstraint("mutation_rate >= 0", name="non_negative_mutation_rate"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_simulation_status", "simulations", ["status"], unique=False)

    # Create organisms table
    op.create_table(
        "organisms",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("species_id", sa.String(length=100), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("parent_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("body_mass", sa.Float(), nullable=False),
        sa.Column("metabolic_rate", sa.Float(), nullable=False),
        sa.Column("oxygen_consumption", sa.Float(), nullable=False),
        sa.Column(
            "temperature_tolerance", postgresql.ARRAY(sa.Float()), nullable=False
        ),
        sa.Column("plasticity_coefficients", sa.JSON(), nullable=False),
        sa.Column("genotype", sa.JSON(), nullable=False),
        sa.Column("mutation_count", sa.Integer(), nullable=False),
        sa.Column("fitness_score", sa.Float(), nullable=True),
        sa.Column("simulation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("is_alive", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("body_mass > 0", name="positive_body_mass"),
        sa.CheckConstraint("metabolic_rate > 0", name="positive_metabolic_rate"),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["organisms.id"],
        ),
        sa.ForeignKeyConstraint(
            ["simulation_id"],
            ["simulations.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_organism_simulation", "organisms", ["simulation_id"], unique=False
    )
    op.create_index(
        "idx_organism_species_generation",
        "organisms",
        ["species_id", "generation"],
        unique=False,
    )
    op.create_index(
        op.f("ix_organisms_generation"), "organisms", ["generation"], unique=False
    )
    op.create_index(
        op.f("ix_organisms_species_id"), "organisms", ["species_id"], unique=False
    )

    # Create environmental_conditions table (will become hypertable)
    op.create_table(
        "environmental_conditions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("simulation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("temperature", sa.Float(), nullable=False),
        sa.Column("oxygen_partial_pressure", sa.Float(), nullable=False),
        sa.Column("altitude", sa.Float(), nullable=False),
        sa.Column("humidity", sa.Float(), nullable=False),
        sa.Column("pressure", sa.Float(), nullable=False),
        sa.Column("resource_availability", sa.Float(), nullable=False),
        sa.Column("predation_pressure", sa.Float(), nullable=False),
        sa.Column("competition_index", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("altitude >= 0", name="positive_altitude"),
        sa.CheckConstraint("humidity >= 0 AND humidity <= 100", name="valid_humidity"),
        sa.CheckConstraint("oxygen_partial_pressure >= 0", name="positive_oxygen"),
        sa.CheckConstraint(
            "temperature >= -50 AND temperature <= 70", name="realistic_temperature"
        ),
        sa.ForeignKeyConstraint(
            ["simulation_id"],
            ["simulations.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_env_generation", "environmental_conditions", ["generation"], unique=False
    )
    op.create_index(
        "idx_env_simulation_time",
        "environmental_conditions",
        ["simulation_id", "timestamp"],
        unique=False,
    )

    # Create physiology_measurements table (will become hypertable)
    op.create_table(
        "physiology_measurements",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("organism_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("measurement_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("measurement_type", sa.String(length=100), nullable=False),
        sa.Column("oxygen_consumption_rate", sa.Float(), nullable=True),
        sa.Column("carbon_dioxide_production", sa.Float(), nullable=True),
        sa.Column("respiratory_quotient", sa.Float(), nullable=True),
        sa.Column("heart_rate", sa.Float(), nullable=True),
        sa.Column("body_temperature", sa.Float(), nullable=True),
        sa.Column("ambient_temperature", sa.Float(), nullable=True),
        sa.Column("ambient_oxygen", sa.Float(), nullable=True),
        sa.Column("ambient_pressure", sa.Float(), nullable=True),
        sa.Column("data_source", sa.String(length=100), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("experimental_conditions", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)",
            name="valid_quality_score",
        ),
        sa.ForeignKeyConstraint(
            ["organism_id"],
            ["organisms.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_physiology_organism_time",
        "physiology_measurements",
        ["organism_id", "measurement_time"],
        unique=False,
    )
    op.create_index(
        "idx_physiology_source",
        "physiology_measurements",
        ["data_source"],
        unique=False,
    )
    op.create_index(
        "idx_physiology_type",
        "physiology_measurements",
        ["measurement_type"],
        unique=False,
    )

    # Create evolutionary_events table (will become hypertable)
    op.create_table(
        "evolutionary_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("simulation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "affected_lineages", postgresql.ARRAY(postgresql.UUID()), nullable=True
        ),
        sa.Column("magnitude", sa.Float(), nullable=True),
        sa.Column("fitness_change", sa.Float(), nullable=True),
        sa.Column("population_impact", sa.Float(), nullable=True),
        sa.Column("environmental_trigger", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["simulation_id"],
            ["simulations.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_event_simulation_generation",
        "evolutionary_events",
        ["simulation_id", "generation"],
        unique=False,
    )
    op.create_index(
        "idx_event_type", "evolutionary_events", ["event_type"], unique=False
    )

    # Create genetic_markers table
    op.create_table(
        "genetic_markers",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("marker_name", sa.String(length=100), nullable=False),
        sa.Column("chromosome", sa.String(length=50), nullable=True),
        sa.Column("position", sa.Integer(), nullable=True),
        sa.Column("phenotype_effect", sa.JSON(), nullable=False),
        sa.Column("effect_size", sa.Float(), nullable=False),
        sa.Column("dominance", sa.Float(), nullable=False),
        sa.Column("allele_frequency", sa.Float(), nullable=True),
        sa.Column("selection_coefficient", sa.Float(), nullable=False),
        sa.Column("discovery_method", sa.String(length=100), nullable=True),
        sa.Column("validation_status", sa.String(length=50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "allele_frequency IS NULL OR "
            "(allele_frequency >= 0 AND allele_frequency <= 1)",
            name="valid_allele_frequency",
        ),
        sa.CheckConstraint(
            "dominance >= -1 AND dominance <= 1", name="valid_dominance"
        ),
        sa.CheckConstraint("effect_size >= 0", name="positive_effect_size"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("marker_name"),
    )
    op.create_index("idx_marker_name", "genetic_markers", ["marker_name"], unique=False)

    # Create experimental_data table
    op.create_table(
        "experimental_data",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("experiment_id", sa.String(length=100), nullable=False),
        sa.Column("species", sa.String(length=100), nullable=False),
        sa.Column("individual_id", sa.String(length=100), nullable=True),
        sa.Column("sample_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("collection_site", sa.String(length=200), nullable=True),
        sa.Column("elevation", sa.Float(), nullable=True),
        sa.Column("coordinates", sa.JSON(), nullable=True),
        sa.Column("measurement_data", sa.JSON(), nullable=False),
        sa.Column("data_type", sa.String(length=100), nullable=False),
        sa.Column("processing_pipeline", sa.String(length=200), nullable=True),
        sa.Column("quality_flags", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("integration_status", sa.String(length=50), nullable=False),
        sa.Column("linked_organism_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["linked_organism_id"],
            ["organisms.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_experiment_date", "experimental_data", ["sample_date"], unique=False
    )
    op.create_index(
        "idx_experiment_id", "experimental_data", ["experiment_id"], unique=False
    )
    op.create_index(
        "idx_experiment_species", "experimental_data", ["species"], unique=False
    )
    op.create_index(
        "idx_experiment_type", "experimental_data", ["data_type"], unique=False
    )

    # Convert time-series tables to hypertables
    op.execute(
        sa.text(
            "SELECT create_hypertable('environmental_conditions', 'timestamp', "
            "chunk_time_interval => INTERVAL '1 day');"
        )
    )
    op.execute(
        sa.text(
            "SELECT create_hypertable('physiology_measurements', 'measurement_time', "
            "chunk_time_interval => INTERVAL '1 day');"
        )
    )
    op.execute(
        sa.text(
            "SELECT create_hypertable('evolutionary_events', 'event_time', "
            "chunk_time_interval => INTERVAL '1 day');"
        )
    )


def downgrade() -> None:
    # Drop all tables in reverse order
    op.drop_table("experimental_data")
    op.drop_table("genetic_markers")
    op.drop_table("evolutionary_events")
    op.drop_table("physiology_measurements")
    op.drop_table("environmental_conditions")
    op.drop_table("organisms")
    op.drop_table("simulations")

    # Drop TimescaleDB extension (optional - may want to keep for other uses)
    # op.execute("DROP EXTENSION IF EXISTS timescaledb CASCADE;")
