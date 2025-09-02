import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, ScatterPlot, Scatter } from 'recharts';
import { Simulation } from '../App';

interface SimulationVisualizationProps {
  simulation: Simulation | null;
}

// Sample data structure for evolutionary metrics
interface EvolutionData {
  generation: number;
  populationSize: number;
  averageFitness: number;
  mutationRate: number;
  selectionPressure: number;
  adaptationScore: number;
}

// Sample data for demonstration
const generateSampleData = (duration: number): EvolutionData[] => {
  const data: EvolutionData[] = [];
  
  for (let gen = 0; gen <= duration; gen++) {
    data.push({
      generation: gen,
      populationSize: 150 + Math.random() * 20 - 10, // Some variation around 150
      averageFitness: 0.5 + (gen / duration) * 0.4 + Math.random() * 0.1, // Gradual improvement
      mutationRate: 0.001 + Math.random() * 0.0005,
      selectionPressure: 0.3 + Math.sin(gen / 5) * 0.1, // Cyclical pressure
      adaptationScore: Math.min(0.95, gen / duration + Math.random() * 0.1)
    });
  }
  
  return data;
};

const SimulationVisualization: React.FC<SimulationVisualizationProps> = ({ simulation }) => {
  if (!simulation) {
    return (
      <div className="card">
        <div className="text-center py-12 text-gray-500">
          <p className="text-4xl mb-4">📈</p>
          <p className="text-lg">No simulation selected</p>
          <p className="text-sm">Select a simulation to view its progress and results</p>
        </div>
      </div>
    );
  }

  const evolutionData = generateSampleData(simulation.parameters.duration);
  
  return (
    <div className="space-y-6">
      {/* Simulation Info Header */}
      <div className="card">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-xl font-bold text-gray-800">{simulation.name}</h2>
            <p className="text-gray-600">
              {simulation.parameters.environment_params.altitude}m altitude, 
              {simulation.parameters.environment_params.temperature}°C, 
              Population: {simulation.parameters.population_size}
            </p>
          </div>
          <div className="text-right">
            <div className="text-2xl mb-1">
              {simulation.status === 'running' && '⚡'}
              {simulation.status === 'completed' && '✅'}
              {simulation.status === 'created' && '📋'}
              {simulation.status === 'failed' && '❌'}
            </div>
            <div className="text-sm text-gray-600 capitalize">{simulation.status}</div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
          <div className="text-center">
            <div className="text-lg font-bold text-blue-600">
              {simulation.parameters.duration}
            </div>
            <div className="text-xs text-gray-600">Generations</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-green-600">
              {(simulation.parameters.mutation_rate * 100).toFixed(3)}%
            </div>
            <div className="text-xs text-gray-600">Mutation Rate</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-purple-600">
              {simulation.parameters.environment_params.oxygen_level}
            </div>
            <div className="text-xs text-gray-600">Oxygen Level</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-orange-600">
              {simulation.progress ? Math.round(simulation.progress) : 0}%
            </div>
            <div className="text-xs text-gray-600">Progress</div>
          </div>
        </div>
      </div>

      {/* Population Dynamics */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          👥 Population Dynamics
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={evolutionData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="generation" />
            <YAxis />
            <Tooltip 
              formatter={(value, name) => [
                typeof value === 'number' ? value.toFixed(1) : value, 
                name
              ]}
            />
            <Area 
              type="monotone" 
              dataKey="populationSize" 
              stroke="#3B82F6" 
              fill="#3B82F6" 
              fillOpacity={0.3}
              name="Population Size"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Fitness Evolution */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          🧬 Fitness Evolution
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={evolutionData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="generation" />
            <YAxis domain={[0, 1]} />
            <Tooltip 
              formatter={(value, name) => [
                typeof value === 'number' ? value.toFixed(3) : value, 
                name
              ]}
            />
            <Line 
              type="monotone" 
              dataKey="averageFitness" 
              stroke="#10B981" 
              strokeWidth={2}
              name="Average Fitness"
            />
            <Line 
              type="monotone" 
              dataKey="adaptationScore" 
              stroke="#8B5CF6" 
              strokeWidth={2}
              name="Adaptation Score"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Environmental Pressures */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          🌡️ Environmental Pressures
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={evolutionData.filter((_, i) => i % 5 === 0)}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="generation" />
            <YAxis />
            <Tooltip 
              formatter={(value, name) => [
                typeof value === 'number' ? value.toFixed(3) : value, 
                name
              ]}
            />
            <Bar 
              dataKey="selectionPressure" 
              fill="#F59E0B" 
              name="Selection Pressure"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Mutation Analysis */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          🔬 Mutation Analysis
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={evolutionData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="generation" />
            <YAxis />
            <Tooltip 
              formatter={(value, name) => [
                typeof value === 'number' ? (value * 1000).toFixed(3) : value, 
                name
              ]}
            />
            <Line 
              type="monotone" 
              dataKey="mutationRate" 
              stroke="#EF4444" 
              strokeWidth={2}
              name="Mutation Rate (x1000)"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Summary Stats */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          📊 Evolution Summary
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-blue-600">
              {evolutionData[evolutionData.length - 1]?.populationSize.toFixed(0) || 'N/A'}
            </div>
            <div className="text-sm text-gray-600">Final Population</div>
          </div>
          <div className="bg-green-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-green-600">
              {((evolutionData[evolutionData.length - 1]?.averageFitness || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Final Fitness</div>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-purple-600">
              {((evolutionData[evolutionData.length - 1]?.adaptationScore || 0) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Adaptation</div>
          </div>
          <div className="bg-orange-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-orange-600">
              {(evolutionData.reduce((sum, d) => sum + d.selectionPressure, 0) / evolutionData.length).toFixed(2)}
            </div>
            <div className="text-sm text-gray-600">Avg Selection</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SimulationVisualization;