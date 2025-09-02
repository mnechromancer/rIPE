import React, { useState, useEffect } from 'react';
import SimulationControls from './components/SimulationControls';
import SimulationDashboard from './components/SimulationDashboard';
import SimulationVisualization from './components/SimulationVisualization';
import './index.css';

export interface Simulation {
  id: string;
  name: string;
  status: 'created' | 'running' | 'completed' | 'failed';
  parameters: {
    duration: number;
    population_size: number;
    mutation_rate: number;
    environment_params: {
      altitude: number;
      temperature: number;
      oxygen_level: number;
    };
  };
  progress?: number;
  results?: any[];
}

function App() {
  const [simulations, setSimulations] = useState<Simulation[]>([]);
  const [activeSimulation, setActiveSimulation] = useState<Simulation | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Check API connection on mount
  useEffect(() => {
    checkApiConnection();
  }, []);

  const checkApiConnection = async () => {
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        setIsConnected(true);
        loadSimulations();
      }
    } catch (error) {
      setIsConnected(false);
      console.error('API connection failed:', error);
    }
  };

  const loadSimulations = async () => {
    try {
      const response = await fetch('/api/v1/simulations');
      if (response.ok) {
        const data = await response.json();
        setSimulations(data);
      }
    } catch (error) {
      console.error('Failed to load simulations:', error);
    }
  };

  const createSimulation = async (params: any) => {
    try {
      const response = await fetch('/api/v1/simulations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });
      
      if (response.ok) {
        const newSimulation = await response.json();
        setSimulations(prev => [...prev, newSimulation]);
        setActiveSimulation(newSimulation);
        return newSimulation;
      }
    } catch (error) {
      console.error('Failed to create simulation:', error);
    }
  };

  const runSimulation = async (simulationId: string) => {
    // Since the API doesn't have a start endpoint yet, we'll simulate this
    setSimulations(prev => 
      prev.map(sim => 
        sim.id === simulationId 
          ? { ...sim, status: 'running', progress: 0 }
          : sim
      )
    );

    // Simulate progress updates
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 10;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        setSimulations(prev => 
          prev.map(sim => 
            sim.id === simulationId 
              ? { ...sim, status: 'completed', progress: 100 }
              : sim
          )
        );
      } else {
        setSimulations(prev => 
          prev.map(sim => 
            sim.id === simulationId 
              ? { ...sim, progress }
              : sim
          )
        );
      }
    }, 500);
  };

  return (
    <div className="app min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-indigo-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            🧬 RIPE - Phylogenetic Evolution Platform
          </h1>
          <p className="text-blue-200 text-lg">
            Interactive Simulation and Visualization Tool
          </p>
          <div className="mt-4">
            <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
              isConnected 
                ? 'bg-green-100 text-green-800' 
                : 'bg-red-100 text-red-800'
            }`}>
              <span className={`w-2 h-2 rounded-full mr-2 ${
                isConnected ? 'bg-green-500' : 'bg-red-500'
              }`}></span>
              {isConnected ? 'API Connected' : 'API Disconnected'}
            </span>
          </div>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-1">
            <SimulationControls 
              onCreateSimulation={createSimulation}
              isConnected={isConnected}
            />
          </div>

          {/* Center Panel - Dashboard */}
          <div className="lg:col-span-1">
            <SimulationDashboard 
              simulations={simulations}
              activeSimulation={activeSimulation}
              onSelectSimulation={setActiveSimulation}
              onRunSimulation={runSimulation}
            />
          </div>

          {/* Right Panel - Visualization */}
          <div className="lg:col-span-1">
            <SimulationVisualization 
              simulation={activeSimulation}
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-12 text-blue-200">
          <p>RIPE Platform - Rapid Interactive Phylogeny Engine</p>
          <p className="text-sm mt-2">Real-time evolutionary simulation and analysis</p>
        </footer>
      </div>
    </div>
  );
}

export default App;