/**
 * TypeScript interfaces for simulation data structures
 */

export interface EnvironmentParams {
  altitude: number;
  temperature: number;
  oxygen_level: number;
  [key: string]: any; // Allow additional environment parameters
}

export interface SimulationParameters {
  duration: number;
  population_size: number;
  mutation_rate: number;
  environment_params: EnvironmentParams;
}

export interface Simulation {
  id: string;
  name: string;
  status: 'created' | 'running' | 'completed' | 'failed';
  created_at: string;
  parameters: SimulationParameters;
  progress?: number;
  results?: any[];
}

export interface StateSpacePoint {
  id: string;
  coordinates: number[];
  fitness: number;
  generation: number;
  metadata?: {
    [key: string]: any;
  };
}

export interface FitnessStats {
  min: number;
  max: number;
  avg: number;
}

export interface GenerationRange {
  min: number;
  max: number;
}

export interface StateSummary {
  total_points: number;
  fitness_stats?: FitnessStats;
  generation_range?: GenerationRange;
}

export interface SimulationViewerData {
  simulation: Simulation;
  states: StateSpacePoint[];
  summary: StateSummary;
}
