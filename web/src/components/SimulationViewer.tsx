import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Simulation,
  StateSpacePoint,
  StateSummary,
  SimulationViewerData,
} from '../types/simulation';

interface SimulationViewerProps {
  simulationId: string;
  onClose?: () => void;
}

const SimulationViewer: React.FC<SimulationViewerProps> = ({
  simulationId,
  onClose,
}) => {
  const [data, setData] = useState<SimulationViewerData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSimulationData();
  }, [simulationId]);

  const fetchSimulationData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch simulation details
      const simResponse = await axios.get<Simulation>(
        `/api/v1/simulations/${simulationId}`
      );

      // Fetch state space data
      const statesResponse = await axios.get<StateSpacePoint[]>(
        `/api/v1/states/${simulationId}`
      );

      // Fetch state summary
      const summaryResponse = await axios.get<StateSummary>(
        `/api/v1/states/${simulationId}/summary`
      );

      setData({
        simulation: simResponse.data,
        states: statesResponse.data,
        summary: summaryResponse.data,
      });
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(
          err.response?.data?.detail || 'Failed to fetch simulation data'
        );
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status: string) => {
    const badges = {
      created: 'bg-blue-100 text-blue-800',
      running: 'bg-yellow-100 text-yellow-800',
      completed: 'bg-green-100 text-green-800',
      failed: 'bg-red-100 text-red-800',
    };
    return badges[status as keyof typeof badges] || 'bg-gray-100 text-gray-800';
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const calculateFitnessEvolution = () => {
    if (!data?.states || data.states.length === 0) return null;

    const sortedStates = [...data.states].sort(
      (a, b) => a.generation - b.generation
    );
    const firstGen = sortedStates[0];
    const lastGen = sortedStates[sortedStates.length - 1];

    const change = lastGen.fitness - firstGen.fitness;
    
    // Handle division by zero when initial fitness is 0
    let percentChange = 0;
    if (firstGen.fitness !== 0) {
      percentChange = (change / firstGen.fitness) * 100;
    } else if (lastGen.fitness !== 0) {
      // If starting from 0, show as infinite improvement
      percentChange = 100;
    }

    return {
      initial: firstGen.fitness,
      final: lastGen.fitness,
      change: change,
      percentChange: percentChange,
    };
  };

  const getPopulationSizeOverTime = () => {
    if (!data?.states || data.states.length === 0) return [];

    // Group states by generation and count
    const generationCounts = data.states.reduce((acc, state) => {
      acc[state.generation] = (acc[state.generation] || 0) + 1;
      return acc;
    }, {} as Record<number, number>);

    return Object.entries(generationCounts)
      .map(([gen, count]) => ({
        generation: parseInt(gen),
        population: count,
      }))
      .sort((a, b) => a.generation - b.generation);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-gray-600 text-lg">Loading simulation data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6">
          <div className="text-center">
            <div className="text-red-500 text-5xl mb-4">⚠️</div>
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Error</h2>
            <p className="text-gray-600 mb-4">{error}</p>
            <button
              onClick={fetchSimulationData}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
            >
              Retry
            </button>
            {onClose && (
              <button
                onClick={onClose}
                className="ml-2 bg-gray-600 text-white px-4 py-2 rounded-md hover:bg-gray-700 transition-colors"
              >
                Close
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  const fitnessEvolution = calculateFitnessEvolution();
  const populationOverTime = getPopulationSizeOverTime();

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex justify-between items-start">
            <div>
              <h1 className="text-3xl font-bold text-gray-800 mb-2">
                {data.simulation.name}
              </h1>
              <div className="flex items-center gap-3">
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusBadge(
                    data.simulation.status
                  )}`}
                >
                  {data.simulation.status.toUpperCase()}
                </span>
                <span className="text-gray-500 text-sm">
                  ID: {data.simulation.id}
                </span>
              </div>
            </div>
            {onClose && (
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Summary Statistics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          {/* Total Generations */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-sm font-medium text-gray-500 mb-1">
              Total Generations
            </div>
            <div className="text-3xl font-bold text-blue-600">
              {data.summary.generation_range?.max || 0}
            </div>
            <div className="text-sm text-gray-500 mt-1">
              Configured: {data.simulation.parameters.duration}
            </div>
          </div>

          {/* Population Size */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-sm font-medium text-gray-500 mb-1">
              Population Size
            </div>
            <div className="text-3xl font-bold text-green-600">
              {data.simulation.parameters.population_size}
            </div>
            <div className="text-sm text-gray-500 mt-1">
              Total states: {data.summary.total_points}
            </div>
          </div>

          {/* Fitness Evolution */}
          {fitnessEvolution && (
            <div className="bg-white rounded-lg shadow p-6">
              <div className="text-sm font-medium text-gray-500 mb-1">
                Fitness Change
              </div>
              <div className="text-3xl font-bold text-purple-600">
                {fitnessEvolution.change >= 0 ? '+' : ''}
                {fitnessEvolution.change.toFixed(3)}
              </div>
              <div className="text-sm text-gray-500 mt-1">
                {fitnessEvolution.percentChange >= 0 ? '↑' : '↓'}{' '}
                {Math.abs(fitnessEvolution.percentChange).toFixed(1)}%
              </div>
            </div>
          )}

          {/* Mutation Rate */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="text-sm font-medium text-gray-500 mb-1">
              Mutation Rate
            </div>
            <div className="text-3xl font-bold text-orange-600">
              {data.simulation.parameters.mutation_rate.toFixed(4)}
            </div>
            <div className="text-sm text-gray-500 mt-1">per generation</div>
          </div>
        </div>

        {/* Fitness Statistics */}
        {data.summary.fitness_stats && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">
              📊 Fitness Statistics
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <div className="text-sm font-medium text-gray-500 mb-1">
                  Minimum Fitness
                </div>
                <div className="text-2xl font-bold text-red-600">
                  {data.summary.fitness_stats.min.toFixed(3)}
                </div>
              </div>
              <div>
                <div className="text-sm font-medium text-gray-500 mb-1">
                  Average Fitness
                </div>
                <div className="text-2xl font-bold text-yellow-600">
                  {data.summary.fitness_stats.avg.toFixed(3)}
                </div>
              </div>
              <div>
                <div className="text-sm font-medium text-gray-500 mb-1">
                  Maximum Fitness
                </div>
                <div className="text-2xl font-bold text-green-600">
                  {data.summary.fitness_stats.max.toFixed(3)}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Simulation Metadata */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Time & Duration */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">
              ⏱️ Time & Duration
            </h2>
            <div className="space-y-3">
              <div>
                <div className="text-sm font-medium text-gray-500">
                  Started At
                </div>
                <div className="text-gray-800">
                  {formatDate(data.simulation.created_at)}
                </div>
              </div>
              <div>
                <div className="text-sm font-medium text-gray-500">
                  Configured Duration
                </div>
                <div className="text-gray-800">
                  {data.simulation.parameters.duration} generations
                </div>
              </div>
              <div>
                <div className="text-sm font-medium text-gray-500">
                  Actual Generations
                </div>
                <div className="text-gray-800">
                  {data.summary.generation_range?.min || 0} -{' '}
                  {data.summary.generation_range?.max || 0}
                </div>
              </div>
            </div>
          </div>

          {/* Environment Parameters */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">
              🌍 Environment Parameters
            </h2>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm font-medium text-gray-500">
                  Altitude
                </span>
                <span className="text-gray-800 font-semibold">
                  {data.simulation.parameters.environment_params.altitude}m
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium text-gray-500">
                  Temperature
                </span>
                <span className="text-gray-800 font-semibold">
                  {data.simulation.parameters.environment_params.temperature}°C
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium text-gray-500">
                  Oxygen Level
                </span>
                <span className="text-gray-800 font-semibold">
                  {data.simulation.parameters.environment_params.oxygen_level}
                </span>
              </div>
              {/* Display any additional environment parameters */}
              {Object.entries(data.simulation.parameters.environment_params)
                .filter(
                  ([key]) =>
                    !['altitude', 'temperature', 'oxygen_level'].includes(key)
                )
                .map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-sm font-medium text-gray-500">
                      {key.charAt(0).toUpperCase() + key.slice(1)}
                    </span>
                    <span className="text-gray-800 font-semibold">
                      {String(value)}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>

        {/* Population Over Time */}
        {populationOverTime.length > 0 && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h2 className="text-xl font-bold text-gray-800 mb-4">
              📈 Population Over Time
            </h2>
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-sm font-medium text-gray-500">
                      Generation
                    </th>
                    <th className="px-4 py-2 text-left text-sm font-medium text-gray-500">
                      Population Count
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {populationOverTime.slice(0, 10).map((item) => (
                    <tr key={item.generation}>
                      <td className="px-4 py-2 text-sm text-gray-800">
                        {item.generation}
                      </td>
                      <td className="px-4 py-2 text-sm text-gray-800">
                        {item.population}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {populationOverTime.length > 10 && (
                <div className="text-center text-sm text-gray-500 mt-4">
                  Showing first 10 of {populationOverTime.length} generations
                </div>
              )}
            </div>
          </div>
        )}

        {/* Simulation Configuration */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">
            ⚙️ Simulation Configuration
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <span className="text-sm font-medium text-gray-500">
                Simulation Type
              </span>
              <div className="text-gray-800 font-semibold">
                Phylogenetic Evolution
              </div>
            </div>
            <div>
              <span className="text-sm font-medium text-gray-500">Status</span>
              <div className="text-gray-800 font-semibold capitalize">
                {data.simulation.status}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimulationViewer;
