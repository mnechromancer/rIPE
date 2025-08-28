/**
 * VIZ-003: Organ System Dashboard
 * Real-time physiological parameters with multi-organ visualization
 * Resource flow animations and comparative views (low vs high altitude)
 */

import React, { useState, useEffect } from 'react';

interface OrganData {
  name: string;
  oxygenConsumption: number;
  bloodFlow: number;
  efficiency: number;
  stress: number;
}

interface PhysiologyData {
  timestamp: number;
  altitude: number;
  organs: {
    heart: OrganData;
    lungs: OrganData;
    brain: OrganData;
    muscle: OrganData;
    liver: OrganData;
  };
  totalO2: number;
  co2Production: number;
  bodyTemp: number;
}

interface OrganSystemDashboardProps {
  isRealTime?: boolean;
  compareAltitudes?: boolean;
}

export const OrganSystemDashboard: React.FC<OrganSystemDashboardProps> = ({
  isRealTime = false,
  compareAltitudes = false
}) => {
  const [currentData, setCurrentData] = useState<PhysiologyData>({
    timestamp: Date.now(),
    altitude: 0,
    organs: {
      heart: { name: 'Heart', oxygenConsumption: 8.0, bloodFlow: 5.0, efficiency: 0.85, stress: 0.2 },
      lungs: { name: 'Lungs', oxygenConsumption: 2.0, bloodFlow: 5.0, efficiency: 0.92, stress: 0.1 },
      brain: { name: 'Brain', oxygenConsumption: 20.0, bloodFlow: 0.75, efficiency: 0.88, stress: 0.15 },
      muscle: { name: 'Muscle', oxygenConsumption: 45.0, bloodFlow: 1.2, efficiency: 0.25, stress: 0.3 },
      liver: { name: 'Liver', oxygenConsumption: 25.0, bloodFlow: 1.5, efficiency: 0.75, stress: 0.25 }
    },
    totalO2: 100,
    co2Production: 80,
    bodyTemp: 37.2
  });

  const [highAltitudeData, setHighAltitudeData] = useState<PhysiologyData>({
    timestamp: Date.now(),
    altitude: 3500,
    organs: {
      heart: { name: 'Heart', oxygenConsumption: 12.0, bloodFlow: 6.5, efficiency: 0.78, stress: 0.6 },
      lungs: { name: 'Lungs', oxygenConsumption: 4.0, bloodFlow: 6.5, efficiency: 0.85, stress: 0.5 },
      brain: { name: 'Brain', oxygenConsumption: 22.0, bloodFlow: 0.9, efficiency: 0.82, stress: 0.4 },
      muscle: { name: 'Muscle', oxygenConsumption: 35.0, bloodFlow: 1.0, efficiency: 0.22, stress: 0.7 },
      liver: { name: 'Liver', oxygenConsumption: 27.0, bloodFlow: 1.3, efficiency: 0.68, stress: 0.45 }
    },
    totalO2: 65,
    co2Production: 85,
    bodyTemp: 36.8
  });

  useEffect(() => {
    if (isRealTime) {
      const interval = setInterval(() => {
        setCurrentData(prev => ({
          ...prev,
          timestamp: Date.now(),
          organs: {
            ...prev.organs,
            heart: {
              ...prev.organs.heart,
              oxygenConsumption: prev.organs.heart.oxygenConsumption + (Math.random() - 0.5) * 0.5,
              stress: Math.max(0, Math.min(1, prev.organs.heart.stress + (Math.random() - 0.5) * 0.1))
            },
            lungs: {
              ...prev.organs.lungs,
              efficiency: Math.max(0.7, Math.min(1, prev.organs.lungs.efficiency + (Math.random() - 0.5) * 0.02))
            }
          }
        }));
      }, 1000);

      return () => clearInterval(interval);
    }
  }, [isRealTime]);

  const renderOrganPanel = (organ: OrganData, isHighAltitude = false) => {
    const stressColor = organ.stress < 0.3 ? '#4CAF50' : organ.stress < 0.6 ? '#FF9800' : '#F44336';
    
    return (
      <div key={organ.name} className="organ-panel" style={{ 
        border: `2px solid ${stressColor}`,
        borderRadius: '8px',
        padding: '10px',
        margin: '5px',
        backgroundColor: isHighAltitude ? '#fff3e0' : '#f0f8ff'
      }}>
        <h4>{organ.name}</h4>
        <div className="organ-metrics">
          <div className="metric">
            <span>O₂ Consumption:</span>
            <div className="metric-bar">
              <div 
                className="metric-fill" 
                style={{ 
                  width: `${organ.oxygenConsumption}%`, 
                  backgroundColor: '#2196F3' 
                }}
              ></div>
              <span className="metric-value">{organ.oxygenConsumption.toFixed(1)}%</span>
            </div>
          </div>
          
          <div className="metric">
            <span>Blood Flow:</span>
            <div className="metric-bar">
              <div 
                className="metric-fill" 
                style={{ 
                  width: `${(organ.bloodFlow / 7) * 100}%`, 
                  backgroundColor: '#FF5722' 
                }}
              ></div>
              <span className="metric-value">{organ.bloodFlow.toFixed(1)} L/min</span>
            </div>
          </div>
          
          <div className="metric">
            <span>Efficiency:</span>
            <div className="metric-bar">
              <div 
                className="metric-fill" 
                style={{ 
                  width: `${organ.efficiency * 100}%`, 
                  backgroundColor: '#4CAF50' 
                }}
              ></div>
              <span className="metric-value">{(organ.efficiency * 100).toFixed(0)}%</span>
            </div>
          </div>
          
          <div className="metric">
            <span>Stress Level:</span>
            <div className="metric-bar">
              <div 
                className="metric-fill" 
                style={{ 
                  width: `${organ.stress * 100}%`, 
                  backgroundColor: stressColor 
                }}
              ></div>
              <span className="metric-value">{(organ.stress * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderResourceFlow = (data: PhysiologyData) => {
    return (
      <div className="resource-flow">
        <h4>Resource Flow Animation</h4>
        <svg width="300" height="200" style={{ border: '1px solid #ccc' }}>
          {/* Simplified resource flow visualization */}
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" 
             refX="0" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#2196F3" />
            </marker>
          </defs>
          
          {/* Central circulation system */}
          <circle cx="150" cy="100" r="30" fill="#FF5722" opacity="0.7" />
          <text x="150" y="105" textAnchor="middle" fontSize="12" fill="white">Heart</text>
          
          {/* Organ connections with animated flow */}
          {Object.entries(data.organs).map(([key, organ], index) => {
            const angle = (index * 2 * Math.PI) / 5;
            const x = 150 + Math.cos(angle) * 80;
            const y = 100 + Math.sin(angle) * 80;
            
            return (
              <g key={key}>
                {/* Organ node */}
                <circle cx={x} cy={y} r="15" fill="#4CAF50" opacity="0.8" />
                <text x={x} y={y + 3} textAnchor="middle" fontSize="8" fill="white">
                  {organ.name.substring(0, 3)}
                </text>
                
                {/* Flow arrow */}
                <line 
                  x1="150" y1="100" 
                  x2={x} y2={y} 
                  stroke="#2196F3" 
                  strokeWidth="2" 
                  markerEnd="url(#arrowhead)"
                  opacity={organ.bloodFlow / 7}
                >
                  {isRealTime && (
                    <animate
                      attributeName="stroke-dasharray"
                      values="0,10;10,0"
                      dur="2s"
                      repeatCount="indefinite"
                    />
                  )}
                </line>
                
                {/* O2 consumption indicator */}
                <circle 
                  cx={x} cy={y - 25} r={organ.oxygenConsumption / 10} 
                  fill="#FFC107" opacity="0.6" 
                />
              </g>
            );
          })}
          
          <text x="10" y="20" fontSize="12">O₂: {data.totalO2.toFixed(0)}%</text>
          <text x="10" y="35" fontSize="12">CO₂: {data.co2Production.toFixed(0)}%</text>
          <text x="10" y="50" fontSize="12">Temp: {data.bodyTemp.toFixed(1)}°C</text>
        </svg>
      </div>
    );
  };

  return (
    <div className="organ-system-dashboard">
      <h2>Organ System Dashboard</h2>
      
      <div className="dashboard-controls">
        <label>
          <input 
            type="checkbox" 
            checked={isRealTime} 
            onChange={(e) => setCurrentData({...currentData, timestamp: Date.now()})}
          />
          Real-time Updates
        </label>
        <label>
          <input 
            type="checkbox" 
            checked={compareAltitudes} 
            readOnly
          />
          Compare Altitudes
        </label>
      </div>

      {compareAltitudes ? (
        <div className="altitude-comparison">
          <div className="altitude-view">
            <h3>Sea Level (0m)</h3>
            <div className="organs-grid">
              {Object.values(currentData.organs).map(organ => renderOrganPanel(organ, false))}
            </div>
            {renderResourceFlow(currentData)}
          </div>
          
          <div className="altitude-view">
            <h3>High Altitude (3500m)</h3>
            <div className="organs-grid">
              {Object.values(highAltitudeData.organs).map(organ => renderOrganPanel(organ, true))}
            </div>
            {renderResourceFlow(highAltitudeData)}
          </div>
        </div>
      ) : (
        <div className="single-view">
          <h3>Current Conditions ({currentData.altitude}m altitude)</h3>
          <div className="organs-grid">
            {Object.values(currentData.organs).map(organ => renderOrganPanel(organ))}
          </div>
          {renderResourceFlow(currentData)}
        </div>
      )}

      <div className="summary-stats">
        <h4>System Summary</h4>
        <div className="stats-grid">
          <div className="stat">
            <label>Total O₂ Consumption:</label>
            <span>{Object.values(currentData.organs).reduce((sum, organ) => sum + organ.oxygenConsumption, 0).toFixed(1)}%</span>
          </div>
          <div className="stat">
            <label>Average Efficiency:</label>
            <span>{(Object.values(currentData.organs).reduce((sum, organ) => sum + organ.efficiency, 0) / 5 * 100).toFixed(0)}%</span>
          </div>
          <div className="stat">
            <label>System Stress:</label>
            <span>{(Object.values(currentData.organs).reduce((sum, organ) => sum + organ.stress, 0) / 5 * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        .organ-system-dashboard {
          padding: 20px;
          font-family: Arial, sans-serif;
        }
        
        .dashboard-controls {
          margin: 20px 0;
        }
        
        .dashboard-controls label {
          margin-right: 20px;
        }
        
        .altitude-comparison {
          display: flex;
          gap: 20px;
        }
        
        .altitude-view {
          flex: 1;
          border: 1px solid #ddd;
          padding: 15px;
          border-radius: 8px;
        }
        
        .organs-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 10px;
          margin: 15px 0;
        }
        
        .metric {
          margin: 5px 0;
        }
        
        .metric-bar {
          position: relative;
          height: 20px;
          background-color: #f0f0f0;
          border-radius: 10px;
          margin: 2px 0;
        }
        
        .metric-fill {
          height: 100%;
          border-radius: 10px;
          transition: width 0.3s ease;
        }
        
        .metric-value {
          position: absolute;
          right: 5px;
          top: 50%;
          transform: translateY(-50%);
          font-size: 12px;
          font-weight: bold;
        }
        
        .stats-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 15px;
          margin-top: 10px;
        }
        
        .stat {
          text-align: center;
          padding: 10px;
          border: 1px solid #ddd;
          border-radius: 4px;
        }
        
        .stat label {
          display: block;
          font-weight: bold;
          margin-bottom: 5px;
        }
      `}</style>
    </div>
  );
};

export default OrganSystemDashboard;