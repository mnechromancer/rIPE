import { useState, useEffect } from 'react';
import './index.css';
import SimulationViewer from './components/SimulationViewer';
import { Simulation } from './types/simulation';

function App() {
  const [simulations, setSimulations] = useState<Simulation[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedSimulationId, setSelectedSimulationId] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    name: 'Alpine Evolution',
    duration: 25,
    population_size: 150,
    mutation_rate: 0.001,
    altitude: 3500,
    temperature: -8,
    oxygen_level: 0.7
  });

  // Check API connection on mount
  useEffect(() => {
    checkConnection();
  }, []);

  const checkConnection = async () => {
    try {
      const response = await fetch('/api/health');
      setIsConnected(response.ok);
    } catch (error) {
      setIsConnected(false);
    }
  };

  const createSimulation = async () => {
    try {
      const payload = {
        name: formData.name,
        duration: formData.duration,
        population_size: formData.population_size,
        mutation_rate: formData.mutation_rate,
        environment_params: {
          altitude: formData.altitude,
          temperature: formData.temperature,
          oxygen_level: formData.oxygen_level
        }
      };

      const response = await fetch('/api/v1/simulations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        const newSim = await response.json();
        setSimulations([...simulations, newSim]);
        alert(`Simulation "${newSim.name}" created successfully!`);
      }
    } catch (error) {
      alert('Failed to create simulation: ' + error);
    }
  };

  const loadSimulations = async () => {
    try {
      const response = await fetch('/api/v1/simulations');
      if (response.ok) {
        const sims = await response.json();
        setSimulations(sims);
      }
    } catch (error) {
      console.error('Failed to load simulations:', error);
    }
  };

  // If a simulation is selected, show the viewer
  if (selectedSimulationId) {
    return (
      <SimulationViewer
        simulationId={selectedSimulationId}
        onClose={() => setSelectedSimulationId(null)}
      />
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-500 to-purple-600 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            🧬 RIPE - Phylogenetic Simulation Platform
          </h1>
          <p className="text-blue-100">
            Interactive Evolution & Adaptation Research
          </p>
          <div className="mt-4">
            <span className={`px-3 py-1 rounded-full text-sm ${
              isConnected ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
            }`}>
              {isConnected ? '✅ API Connected' : '❌ API Disconnected'}
            </span>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Left Panel - Controls */}
          <div className="space-y-6">
            {/* Simulation Creation Form */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">
                🎛️ Create New Simulation
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Simulation Name
                  </label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Population Size
                    </label>
                    <input
                      type="number"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      value={formData.population_size}
                      onChange={(e) => setFormData({...formData, population_size: parseInt(e.target.value)})}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Generations
                    </label>
                    <input
                      type="number"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      value={formData.duration}
                      onChange={(e) => setFormData({...formData, duration: parseInt(e.target.value)})}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Altitude (m)
                    </label>
                    <input
                      type="number"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      value={formData.altitude}
                      onChange={(e) => setFormData({...formData, altitude: parseInt(e.target.value)})}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Temperature (°C)
                    </label>
                    <input
                      type="number"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      value={formData.temperature}
                      onChange={(e) => setFormData({...formData, temperature: parseInt(e.target.value)})}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Oxygen Level
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md"
                      value={formData.oxygen_level}
                      onChange={(e) => setFormData({...formData, oxygen_level: parseFloat(e.target.value)})}
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Mutation Rate
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                    value={formData.mutation_rate}
                    onChange={(e) => setFormData({...formData, mutation_rate: parseFloat(e.target.value)})}
                  />
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={createSimulation}
                    className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors"
                  >
                    🚀 Create Simulation
                  </button>
                  <button
                    onClick={loadSimulations}
                    className="bg-gray-600 text-white py-2 px-4 rounded-md hover:bg-gray-700 transition-colors"
                  >
                    🔄 Refresh
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Right Panel - Simulations List */}
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">
                📊 Simulations ({simulations.length})
              </h2>
              
              {simulations.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <p className="text-3xl mb-2">🧬</p>
                  <p>No simulations yet</p>
                  <p className="text-sm">Create your first simulation to get started</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {simulations.map((sim) => (
                    <div key={sim.id} className="p-4 border border-gray-200 rounded-lg">
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="font-semibold text-gray-800">{sim.name}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          sim.status === 'created' ? 'bg-blue-100 text-blue-800' :
                          sim.status === 'running' ? 'bg-yellow-100 text-yellow-800' :
                          sim.status === 'completed' ? 'bg-green-100 text-green-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {sim.status}
                        </span>
                      </div>
                      
                      <div className="text-sm text-gray-600 space-y-1 mb-3">
                        <div>Population: {sim.parameters.population_size} | Generations: {sim.parameters.duration}</div>
                        <div>Altitude: {sim.parameters.environment_params.altitude}m | Temp: {sim.parameters.environment_params.temperature}°C</div>
                        <div>Oxygen: {sim.parameters.environment_params.oxygen_level} | Mutation: {sim.parameters.mutation_rate}</div>
                      </div>

                      <button
                        onClick={() => setSelectedSimulationId(sim.id)}
                        className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors text-sm font-medium"
                      >
                        📊 View Details
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-blue-100 text-sm">
          <p>RIPE Platform - Real-time Interactive Phylogenetic Evolution</p>
          <p>Built with React + TypeScript + Vite</p>
        </div>
      </div>
    </div>
  );
}

export default App;