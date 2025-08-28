/**
 * VIZ-002: Plasticity Landscape Viewer
 * G×E interaction surface plot with maladaptive region highlighting
 * Animation of genetic assimilation and interactive parameter adjustment
 */

import React, { useState, useEffect } from 'react';

interface PlasticityLandscapeProps {
  gxeData?: Array<{
    genotype: number;
    environment: number;
    fitness: number;
    isAdaptive: boolean;
  }>;
  animationSpeed?: number;
}

export const PlasticityLandscape: React.FC<PlasticityLandscapeProps> = ({
  gxeData = [],
  animationSpeed = 1
}) => {
  const [isAnimating, setIsAnimating] = useState(false);
  const [currentGeneration, setCurrentGeneration] = useState(0);
  const [parameters, setParameters] = useState({
    plasticityStrength: 0.5,
    environmentVariance: 0.3,
    selectionStrength: 0.8
  });

  useEffect(() => {
    if (isAnimating) {
      const interval = setInterval(() => {
        setCurrentGeneration(prev => (prev + 1) % 100);
      }, 1000 / animationSpeed);
      
      return () => clearInterval(interval);
    }
  }, [isAnimating, animationSpeed]);

  const handleParameterChange = (param: string, value: number) => {
    setParameters(prev => ({
      ...prev,
      [param]: value
    }));
  };

  return (
    <div className="plasticity-landscape">
      <h2>Plasticity Landscape Viewer</h2>
      
      <div className="controls">
        <div className="parameter-controls">
          <h3>Interactive Parameters</h3>
          <div className="parameter">
            <label>Plasticity Strength: {parameters.plasticityStrength}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={parameters.plasticityStrength}
              onChange={(e) => handleParameterChange('plasticityStrength', parseFloat(e.target.value))}
            />
          </div>
          <div className="parameter">
            <label>Environment Variance: {parameters.environmentVariance}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={parameters.environmentVariance}
              onChange={(e) => handleParameterChange('environmentVariance', parseFloat(e.target.value))}
            />
          </div>
          <div className="parameter">
            <label>Selection Strength: {parameters.selectionStrength}</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={parameters.selectionStrength}
              onChange={(e) => handleParameterChange('selectionStrength', parseFloat(e.target.value))}
            />
          </div>
        </div>

        <div className="animation-controls">
          <button onClick={() => setIsAnimating(!isAnimating)}>
            {isAnimating ? 'Pause' : 'Play'} Genetic Assimilation
          </button>
          <div>Generation: {currentGeneration}</div>
        </div>
      </div>

      <div className="landscape-view">
        <div className="gxe-surface">
          <h3>G×E Interaction Surface Plot</h3>
          <svg width="400" height="300" style={{ border: '1px solid #ccc' }}>
            {/* Simulated 3D surface - in real implementation would use Three.js */}
            <defs>
              <linearGradient id="fitnessGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#ff0000" />
                <stop offset="50%" stopColor="#ffff00" />
                <stop offset="100%" stopColor="#00ff00" />
              </linearGradient>
            </defs>
            <rect width="400" height="300" fill="url(#fitnessGradient)" opacity="0.7" />
            
            {/* Maladaptive regions highlighted in red */}
            <rect x="50" y="50" width="100" height="80" fill="red" opacity="0.5" />
            <rect x="250" y="150" width="120" height="100" fill="red" opacity="0.5" />
            
            <text x="10" y="20" fill="black">Genotype →</text>
            <text x="10" y="290" fill="black">Environment →</text>
            <text x="350" y="20" fill="black">Fitness</text>
          </svg>
        </div>

        <div className="legend">
          <h4>Legend</h4>
          <div className="legend-item">
            <span className="adaptive-color"></span> Adaptive regions
          </div>
          <div className="legend-item">
            <span className="maladaptive-color"></span> Maladaptive regions (highlighted)
          </div>
        </div>
      </div>

      <div className="status">
        <p>Plasticity Parameters: Strength={parameters.plasticityStrength}, Variance={parameters.environmentVariance}</p>
        <p>Selection Strength: {parameters.selectionStrength}</p>
        <p>Animation Status: {isAnimating ? 'Running' : 'Paused'}</p>
      </div>
    </div>
  );
};

export default PlasticityLandscape;