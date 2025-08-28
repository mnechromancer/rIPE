/**
 * VIZ-004: Phylogenetic Network Builder
 * Interactive tree/network view with strategy-based branching
 * Time slider for evolution and export to Newick format
 */

import React, { useState, useEffect } from 'react';

interface PhylogenyNode {
  id: string;
  name: string;
  parent?: string;
  children: string[];
  branchLength: number;
  strategy: string;
  fitness: number;
  generation: number;
  x: number;
  y: number;
}

interface PhylogenyNetworkProps {
  initialNodes?: PhylogenyNode[];
  showStrategies?: boolean;
  enableTimeSlider?: boolean;
}

export const PhylogenyNetwork: React.FC<PhylogenyNetworkProps> = ({
  initialNodes = [],
  showStrategies = true,
  enableTimeSlider = true
}) => {
  const [nodes, setNodes] = useState<PhylogenyNode[]>([
    {
      id: 'root',
      name: 'Ancestral Population',
      children: ['n1', 'n2'],
      branchLength: 0,
      strategy: 'Generalist',
      fitness: 1.0,
      generation: 0,
      x: 50,
      y: 200
    },
    {
      id: 'n1',
      name: 'High-altitude Specialist',
      parent: 'root',
      children: ['n3', 'n4'],
      branchLength: 15,
      strategy: 'Hypoxia Tolerance',
      fitness: 1.3,
      generation: 15,
      x: 200,
      y: 100
    },
    {
      id: 'n2',
      name: 'Thermal Specialist',
      parent: 'root',
      children: ['n5'],
      branchLength: 20,
      strategy: 'Temperature Regulation',
      fitness: 1.1,
      generation: 20,
      x: 200,
      y: 300
    },
    {
      id: 'n3',
      name: 'Extreme Altitude',
      parent: 'n1',
      children: [],
      branchLength: 25,
      strategy: 'Extreme Hypoxia',
      fitness: 1.5,
      generation: 40,
      x: 350,
      y: 50
    },
    {
      id: 'n4',
      name: 'Moderate Altitude',
      parent: 'n1',
      children: [],
      branchLength: 18,
      strategy: 'Moderate Hypoxia',
      fitness: 1.2,
      generation: 33,
      x: 350,
      y: 150
    },
    {
      id: 'n5',
      name: 'Cold Specialist',
      parent: 'n2',
      children: [],
      branchLength: 12,
      strategy: 'Cold Adaptation',
      fitness: 1.4,
      generation: 32,
      x: 350,
      y: 300
    }
  ]);

  const [currentGeneration, setCurrentGeneration] = useState(50);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'tree' | 'network'>('tree');
  const [isAnimating, setIsAnimating] = useState(false);

  const maxGeneration = Math.max(...nodes.map(n => n.generation));

  useEffect(() => {
    if (isAnimating && enableTimeSlider) {
      const interval = setInterval(() => {
        setCurrentGeneration(prev => {
          const next = prev + 1;
          return next > maxGeneration ? 0 : next;
        });
      }, 300);

      return () => clearInterval(interval);
    }
  }, [isAnimating, maxGeneration, enableTimeSlider]);

  const getStrategyColor = (strategy: string) => {
    const colors: Record<string, string> = {
      'Generalist': '#757575',
      'Hypoxia Tolerance': '#2196F3',
      'Temperature Regulation': '#FF5722',
      'Extreme Hypoxia': '#0D47A1',
      'Moderate Hypoxia': '#1976D2',
      'Cold Adaptation': '#00BCD4'
    };
    return colors[strategy] || '#9E9E9E';
  };

  const getVisibleNodes = () => {
    return nodes.filter(node => node.generation <= currentGeneration);
  };

  const exportToNewick = () => {
    const buildNewick = (nodeId: string): string => {
      const node = nodes.find(n => n.id === nodeId);
      if (!node) return '';

      if (node.children.length === 0) {
        return `${node.name}:${node.branchLength}`;
      }

      const childrenNewick = node.children
        .map(childId => buildNewick(childId))
        .join(',');
      
      return `(${childrenNewick})${node.name}:${node.branchLength}`;
    };

    const newick = buildNewick('root') + ';';
    
    // Create download
    const blob = new Blob([newick], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'phylogeny.newick';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const renderNode = (node: PhylogenyNode) => {
    const isVisible = node.generation <= currentGeneration;
    const isSelected = selectedNode === node.id;
    
    if (!isVisible) return null;

    return (
      <g key={node.id}>
        <circle
          cx={node.x}
          cy={node.y}
          r={node.children.length === 0 ? 8 : 12}
          fill={getStrategyColor(node.strategy)}
          stroke={isSelected ? '#FFD700' : '#000'}
          strokeWidth={isSelected ? 3 : 1}
          opacity={isVisible ? 1 : 0.3}
          style={{ cursor: 'pointer' }}
          onClick={() => setSelectedNode(node.id)}
        />
        
        {/* Node label */}
        <text
          x={node.x}
          y={node.y - 20}
          textAnchor="middle"
          fontSize="10"
          fill="#333"
          style={{ cursor: 'pointer' }}
          onClick={() => setSelectedNode(node.id)}
        >
          {node.name}
        </text>
        
        {showStrategies && (
          <text
            x={node.x}
            y={node.y + 25}
            textAnchor="middle"
            fontSize="8"
            fill={getStrategyColor(node.strategy)}
          >
            {node.strategy}
          </text>
        )}

        {/* Fitness indicator */}
        <text
          x={node.x + 15}
          y={node.y - 5}
          fontSize="8"
          fill="#666"
        >
          {node.fitness.toFixed(1)}
        </text>
      </g>
    );
  };

  const renderBranch = (node: PhylogenyNode) => {
    if (!node.parent) return null;
    
    const parent = nodes.find(n => n.id === node.parent);
    if (!parent) return null;

    const isVisible = node.generation <= currentGeneration;
    if (!isVisible) return null;

    return (
      <line
        key={`branch-${node.id}`}
        x1={parent.x}
        y1={parent.y}
        x2={node.x}
        y2={node.y}
        stroke="#666"
        strokeWidth="2"
        opacity={isVisible ? 1 : 0.3}
      />
    );
  };

  const renderSelectedNodeInfo = () => {
    if (!selectedNode) return null;
    
    const node = nodes.find(n => n.id === selectedNode);
    if (!node) return null;

    return (
      <div className="node-info-panel">
        <h4>{node.name}</h4>
        <div className="node-details">
          <div><strong>Strategy:</strong> {node.strategy}</div>
          <div><strong>Fitness:</strong> {node.fitness.toFixed(2)}</div>
          <div><strong>Generation:</strong> {node.generation}</div>
          <div><strong>Branch Length:</strong> {node.branchLength}</div>
          {node.children.length > 0 && (
            <div><strong>Children:</strong> {node.children.length}</div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="phylogeny-network">
      <h2>Phylogenetic Network Builder</h2>
      
      <div className="controls">
        <div className="view-controls">
          <label>
            <input
              type="radio"
              value="tree"
              checked={viewMode === 'tree'}
              onChange={(e) => setViewMode(e.target.value as 'tree' | 'network')}
            />
            Tree View
          </label>
          <label>
            <input
              type="radio"
              value="network"
              checked={viewMode === 'network'}
              onChange={(e) => setViewMode(e.target.value as 'tree' | 'network')}
            />
            Network View
          </label>
        </div>

        <div className="display-controls">
          <label>
            <input
              type="checkbox"
              checked={showStrategies}
              onChange={(e) => setShowStrategies(e.target.checked)}
            />
            Show Strategies
          </label>
        </div>

        <div className="export-controls">
          <button onClick={exportToNewick}>
            Export to Newick Format
          </button>
        </div>
      </div>

      {enableTimeSlider && (
        <div className="time-controls">
          <h4>Evolution Timeline</h4>
          <div className="time-slider-container">
            <button onClick={() => setIsAnimating(!isAnimating)}>
              {isAnimating ? 'Pause' : 'Play'}
            </button>
            <input
              type="range"
              min="0"
              max={maxGeneration}
              value={currentGeneration}
              onChange={(e) => setCurrentGeneration(parseInt(e.target.value))}
              style={{ width: '300px', margin: '0 10px' }}
            />
            <span>Generation: {currentGeneration}</span>
          </div>
        </div>
      )}

      <div className="phylogeny-display">
        <div className="tree-visualization">
          <svg width="500" height="400" style={{ border: '1px solid #ddd' }}>
            {/* Render branches first */}
            {getVisibleNodes().map(node => renderBranch(node))}
            
            {/* Render nodes on top */}
            {getVisibleNodes().map(node => renderNode(node))}
            
            {/* Time indicator */}
            <text x="10" y="20" fontSize="14" fill="#333">
              Generation: {currentGeneration}
            </text>
            
            {/* Scale bar */}
            <line x1="10" y1="380" x2="60" y2="380" stroke="#000" strokeWidth="2" />
            <text x="35" y="395" textAnchor="middle" fontSize="10">
              10 generations
            </text>
          </svg>
        </div>

        {renderSelectedNodeInfo()}
      </div>

      <div className="strategy-legend">
        <h4>Strategy Legend</h4>
        <div className="legend-items">
          {Array.from(new Set(nodes.map(n => n.strategy))).map(strategy => (
            <div key={strategy} className="legend-item">
              <span 
                className="strategy-color" 
                style={{ backgroundColor: getStrategyColor(strategy) }}
              ></span>
              {strategy}
            </div>
          ))}
        </div>
      </div>

      <div className="network-stats">
        <h4>Network Statistics</h4>
        <div className="stats">
          <div>Total Nodes: {getVisibleNodes().length}</div>
          <div>Leaf Nodes: {getVisibleNodes().filter(n => n.children.length === 0).length}</div>
          <div>Max Fitness: {Math.max(...getVisibleNodes().map(n => n.fitness)).toFixed(2)}</div>
          <div>Generations Elapsed: {currentGeneration}</div>
        </div>
      </div>
      
      <style jsx>{`
        .phylogeny-network {
          padding: 20px;
          font-family: Arial, sans-serif;
        }
        
        .controls {
          display: flex;
          gap: 20px;
          margin: 20px 0;
          flex-wrap: wrap;
        }
        
        .controls > div {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }
        
        .controls label {
          display: flex;
          align-items: center;
          gap: 5px;
        }
        
        .time-controls {
          margin: 20px 0;
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 8px;
          background-color: #f9f9f9;
        }
        
        .time-slider-container {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-top: 10px;
        }
        
        .phylogeny-display {
          display: flex;
          gap: 20px;
          margin: 20px 0;
        }
        
        .node-info-panel {
          min-width: 200px;
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 8px;
          background-color: #f5f5f5;
        }
        
        .node-details > div {
          margin: 5px 0;
        }
        
        .strategy-legend {
          margin: 20px 0;
        }
        
        .legend-items {
          display: flex;
          flex-wrap: wrap;
          gap: 15px;
        }
        
        .legend-item {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .strategy-color {
          width: 16px;
          height: 16px;
          border-radius: 50%;
          border: 1px solid #333;
        }
        
        .network-stats {
          margin: 20px 0;
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 8px;
          background-color: #f0f8ff;
        }
        
        .stats {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 10px;
          margin-top: 10px;
        }
      `}</style>
    </div>
  );
};

export default PhylogenyNetwork;