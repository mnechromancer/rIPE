#!/usr/bin/env python3
"""
Alembic environment configuration for IPE (Integrated Phenotypic Evolution) platform.
Handles both regular PostgreSQL tables and TimescaleDB hypertables.
"""

import logging
import os
from logging.config import fileConfig

import sqlalchemy as sa
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import your models here
from ipe.data.models import Base, HYPERTABLE_CONFIGS

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the sqlalchemy.url from environment if available
if os.environ.get("DATABASE_URL"):
    config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])

# add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def include_object(object, name, type_, reflected, compare_to):
    """
    Include/exclude objects from autogenerate.
    """
    # Skip alembic version table
    if type_ == "table" and name == "alembic_version":
        return False

    # Include all other objects
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )

        with context.begin_transaction():
            # Enable TimescaleDB extension if not already enabled
            connection.execute(
                sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            )

            # Run the standard migrations
            context.run_migrations()

            # Create hypertables for time-series data
            _create_hypertables(connection)


def _create_hypertables(connection):
    """Create TimescaleDB hypertables for time-series data."""
    logger = logging.getLogger("alembic.env")

    for config in HYPERTABLE_CONFIGS:
        table_name = config["table"]
        time_column = config["time_column"]
        chunk_interval = config["chunk_time_interval"]

        try:
            # Check if hypertable already exists
            result = connection.execute(
                sa.text(
                    "SELECT * FROM timescaledb_information.hypertables "
                    "WHERE hypertable_name = %s"
                ),
                (table_name,),
            )

            if result.rowcount == 0:
                # Create hypertable
                logger.info(f"Creating hypertable for {table_name}")
                connection.execute(
                    sa.text(
                        f"SELECT create_hypertable('{table_name}', '{time_column}', "
                        f"chunk_time_interval => INTERVAL '{chunk_interval}');"
                    )
                )

                # Add compression policy (compress data older than 7 days)
                connection.execute(
                    sa.text(
                        f"ALTER TABLE {table_name} SET (timescaledb.compress = true);"
                    )
                )
                connection.execute(
                    sa.text(
                        f"SELECT add_compression_policy('{table_name}', "
                        f"INTERVAL '7 days');"
                    )
                )

                logger.info(f"Successfully created hypertable for {table_name}")
            else:
                logger.info(f"Hypertable {table_name} already exists")

        except Exception as e:
            logger.warning(f"Failed to create hypertable for {table_name}: {e}")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
