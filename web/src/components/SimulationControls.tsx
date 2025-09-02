import React, { useState } from 'react';

interface SimulationControlsProps {
  onCreateSimulation: (params: any) => void;
  isConnected: boolean;
}

const SimulationControls: React.FC<SimulationControlsProps> = ({
  onCreateSimulation,
  isConnected
}) => {
  const [params, setParams] = useState({
    name: 'Alpine Evolution',
    duration: 25,
    population_size: 150,
    mutation_rate: 0.002,
    environment_params: {
      altitude: 3500,
      temperature: -8,
      oxygen_level: 0.65
    }
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isConnected) return;

    setIsSubmitting(true);
    try {
      await onCreateSimulation(params);
    } finally {
      setIsSubmitting(false);
    }
  };

  const updateParam = (path: string, value: any) => {
    const keys = path.split('.');
    setParams(prev => {
      const newParams = { ...prev };
      let current: any = newParams;
      
      for (let i = 0; i < keys.length - 1; i++) {
        current = current[keys[i]];
      }
      current[keys[keys.length - 1]] = value;
      
      return newParams;
    });
  };

  return (
    <div className="card">
      <h2 className="text-xl font-bold text-gray-800 mb-4">
        🎛️ Simulation Controls
      </h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Basic Parameters */}
        <div>
          <label className="label">Simulation Name</label>
          <input
            type="text"
            className="input"
            value={params.name}
            onChange={(e) => updateParam('name', e.target.value)}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Duration (generations)</label>
            <input
              type="number"
              className="input"
              value={params.duration}
              onChange={(e) => updateParam('duration', parseInt(e.target.value))}
              min="1"
              max="1000"
            />
          </div>
          
          <div>
            <label className="label">Population Size</label>
            <input
              type="number"
              className="input"
              value={params.population_size}
              onChange={(e) => updateParam('population_size', parseInt(e.target.value))}
              min="10"
              max="1000"
            />
          </div>
        </div>

        <div>
          <label className="label">Mutation Rate</label>
          <input
            type="number"
            className="input"
            value={params.mutation_rate}
            onChange={(e) => updateParam('mutation_rate', parseFloat(e.target.value))}
            step="0.001"
            min="0"
            max="1"
          />
        </div>

        {/* Environment Parameters */}
        <div className="border-t pt-4">
          <h3 className="font-semibold text-gray-700 mb-3">🌍 Environment</h3>
          
          <div className="space-y-3">
            <div>
              <label className="label">Altitude (meters)</label>
              <input
                type="number"
                className="input"
                value={params.environment_params.altitude}
                onChange={(e) => updateParam('environment_params.altitude', parseInt(e.target.value))}
                min="0"
                max="9000"
              />
            </div>
            
            <div>
              <label className="label">Temperature (°C)</label>
              <input
                type="number"
                className="input"
                value={params.environment_params.temperature}
                onChange={(e) => updateParam('environment_params.temperature', parseInt(e.target.value))}
                min="-50"
                max="50"
              />
            </div>
            
            <div>
              <label className="label">Oxygen Level (0-1)</label>
              <input
                type="number"
                className="input"
                value={params.environment_params.oxygen_level}
                onChange={(e) => updateParam('environment_params.oxygen_level', parseFloat(e.target.value))}
                step="0.01"
                min="0"
                max="1"
              />
            </div>
          </div>
        </div>

        {/* Quick Presets */}
        <div className="border-t pt-4">
          <h3 className="font-semibold text-gray-700 mb-3">⚡ Quick Presets</h3>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              className="btn-primary text-sm py-1"
              onClick={() => setParams({
                ...params,
                name: 'Alpine Evolution',
                environment_params: { altitude: 3500, temperature: -8, oxygen_level: 0.65 }
              })}
            >
              🏔️ Alpine
            </button>
            
            <button
              type="button"
              className="btn-primary text-sm py-1"
              onClick={() => setParams({
                ...params,
                name: 'Everest Survival',
                environment_params: { altitude: 8000, temperature: -30, oxygen_level: 0.3 }
              })}
            >
              🏔️ Everest
            </button>
            
            <button
              type="button"
              className="btn-primary text-sm py-1"
              onClick={() => setParams({
                ...params,
                name: 'Sea Level Control',
                environment_params: { altitude: 0, temperature: 15, oxygen_level: 1.0 }
              })}
            >
              🌊 Sea Level
            </button>
            
            <button
              type="button"
              className="btn-primary text-sm py-1"
              onClick={() => setParams({
                ...params,
                name: 'Arctic Survival',
                environment_params: { altitude: 0, temperature: -40, oxygen_level: 1.0 }
              })}
            >
              ❄️ Arctic
            </button>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!isConnected || isSubmitting}
          className={`w-full btn ${
            isConnected && !isSubmitting 
              ? 'btn-success' 
              : 'bg-gray-400 text-gray-700 cursor-not-allowed'
          }`}
        >
          {isSubmitting ? (
            <span className="flex items-center justify-center">
              <span className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></span>
              Creating...
            </span>
          ) : (
            '🚀 Create Simulation'
          )}
        </button>

        {!isConnected && (
          <p className="text-red-600 text-sm text-center">
            ⚠️ API not connected. Please start the RIPE server first.
          </p>
        )}
      </form>
    </div>
  );
};

export default SimulationControls;