import React from 'react';
import { Simulation } from '../App';

interface SimulationDashboardProps {
  simulations: Simulation[];
  activeSimulation: Simulation | null;
  onSelectSimulation: (simulation: Simulation) => void;
  onRunSimulation: (simulationId: string) => void;
}

const SimulationDashboard: React.FC<SimulationDashboardProps> = ({
  simulations,
  activeSimulation,
  onSelectSimulation,
  onRunSimulation
}) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'created': return 'bg-blue-100 text-blue-800';
      case 'running': return 'bg-yellow-100 text-yellow-800';
      case 'completed': return 'bg-green-100 text-green-800';
      case 'failed': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'created': return '📋';
      case 'running': return '⚡';
      case 'completed': return '✅';
      case 'failed': return '❌';
      default: return '❓';
    }
  };

  return (
    <div className="card">
      <h2 className="text-xl font-bold text-gray-800 mb-4">
        📊 Simulation Dashboard
      </h2>

      {simulations.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <p className="text-3xl mb-2">🧬</p>
          <p>No simulations yet</p>
          <p className="text-sm">Create your first simulation to get started</p>
        </div>
      ) : (
        <div className="space-y-3">
          {simulations.map((simulation) => (
            <div
              key={simulation.id}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                activeSimulation?.id === simulation.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => onSelectSimulation(simulation)}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-800">
                  {simulation.name}
                </h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(simulation.status)}`}>
                  {getStatusIcon(simulation.status)} {simulation.status}
                </span>
              </div>

              <div className="text-sm text-gray-600 mb-2">
                <div className="grid grid-cols-2 gap-2">
                  <span>Population: {simulation.parameters.population_size}</span>
                  <span>Duration: {simulation.parameters.duration}g</span>
                  <span>Altitude: {simulation.parameters.environment_params.altitude}m</span>
                  <span>Temp: {simulation.parameters.environment_params.temperature}°C</span>
                </div>
              </div>

              {simulation.progress !== undefined && (
                <div className="mb-2">
                  <div className="flex justify-between text-xs text-gray-600 mb-1">
                    <span>Progress</span>
                    <span>{Math.round(simulation.progress)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${simulation.progress}%` }}
                    ></div>
                  </div>
                </div>
              )}

              <div className="flex gap-2">
                {simulation.status === 'created' && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onRunSimulation(simulation.id);
                    }}
                    className="btn-success text-xs py-1 px-2"
                  >
                    ▶️ Run
                  </button>
                )}
                
                {simulation.status === 'running' && (
                  <div className="flex items-center text-xs text-yellow-600">
                    <span className="animate-spin rounded-full h-3 w-3 border-b border-yellow-600 mr-1"></span>
                    Running...
                  </div>
                )}
                
                {simulation.status === 'completed' && (
                  <button className="btn-primary text-xs py-1 px-2">
                    📊 View Results
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Quick Stats */}
      {simulations.length > 0 && (
        <div className="mt-6 pt-4 border-t">
          <h3 className="font-semibold text-gray-700 mb-2">Quick Stats</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {simulations.length}
              </div>
              <div className="text-gray-600">Total Simulations</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {simulations.filter(s => s.status === 'completed').length}
              </div>
              <div className="text-gray-600">Completed</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SimulationDashboard;