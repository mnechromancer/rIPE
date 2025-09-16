-- IPE (Integrated Phenotypic Evolution) Database Initialization Script
-- This script sets up the initial database schema for the IPE platform

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS ipe;

-- Set default search path
SET search_path TO ipe, public;

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create basic tables for IPE data structures

-- Simulations table
CREATE TABLE IF NOT EXISTS simulations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    parameters JSONB,
    status VARCHAR(50) DEFAULT 'created',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Organisms table
CREATE TABLE IF NOT EXISTS organisms (
    id SERIAL PRIMARY KEY,
    simulation_id INTEGER REFERENCES simulations(id),
    generation INTEGER NOT NULL,
    genome JSONB,
    phenotype JSONB,
    fitness REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Environmental conditions table (time-series data)
CREATE TABLE IF NOT EXISTS environmental_conditions (
    time TIMESTAMPTZ NOT NULL,
    simulation_id INTEGER REFERENCES simulations(id),
    temperature REAL,
    oxygen_level REAL,
    pressure REAL,
    conditions JSONB
);

-- Convert environmental_conditions to hypertable for time-series optimization
SELECT create_hypertable('environmental_conditions', 'time', if_not_exists => TRUE);

-- Physiology measurements table (time-series data)
CREATE TABLE IF NOT EXISTS physiology_measurements (
    time TIMESTAMPTZ NOT NULL,
    organism_id INTEGER REFERENCES organisms(id),
    measurement_type VARCHAR(100),
    value REAL,
    unit VARCHAR(50),
    metadata JSONB
);

-- Convert physiology_measurements to hypertable
SELECT create_hypertable('physiology_measurements', 'time', if_not_exists => TRUE);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_simulations_status ON simulations(status);
CREATE INDEX IF NOT EXISTS idx_organisms_simulation_generation ON organisms(simulation_id, generation);
CREATE INDEX IF NOT EXISTS idx_environmental_conditions_simulation ON environmental_conditions(simulation_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_physiology_measurements_organism ON physiology_measurements(organism_id, time DESC);

-- Create a basic test simulation entry
INSERT INTO simulations (name, description, parameters, status) 
VALUES (
    'Test Simulation', 
    'Initial test simulation for IPE platform',
    '{"population_size": 1000, "generations": 100}',
    'completed'
) ON CONFLICT DO NOTHING;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'IPE database initialization completed successfully';
END $$;