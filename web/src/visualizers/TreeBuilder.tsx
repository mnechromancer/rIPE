/**
 * VIZ-004: Tree Builder Component
 * Interactive phylogenetic tree construction and visualization tools
 */

import React, { useState, useCallback } from 'react';

interface TreeNode {
  id: string;
  name: string;
  parent?: string;
  children: string[];
  branchLength: number;
  support: number;
  x: number;
  y: number;
  depth: number;
}

interface TreeBuilderProps {
  onTreeUpdate?: (nodes: TreeNode[]) => void;
  maxDepth?: number;
}

export const TreeBuilder: React.FC<TreeBuilderProps> = ({
  onTreeUpdate,
  maxDepth = 5
}) => {
  const [nodes, setNodes] = useState<TreeNode[]>([
    {
      id: 'root',
      name: 'Root',
      children: [],
      branchLength: 0,
      support: 1.0,
      x: 50,
      y: 200,
      depth: 0
    }
  ]);

  const [selectedNode, setSelectedNode] = useState<string>('root');
  const [nodeName, setNodeName] = useState('');
  const [branchLength, setBranchLength] = useState(1.0);
  const [support, setSupport] = useState(0.95);
  const [treeLayout, setTreeLayout] = useState<'rectangular' | 'circular'>('rectangular');
  const [showSupport, setShowSupport] = useState(true);
  const [showBranchLengths, setShowBranchLengths] = useState(true);

  const calculateLayout = useCallback(() => {
    const updatedNodes = [...nodes];
    
    if (treeLayout === 'rectangular') {
      // Calculate rectangular (phylogram) layout
      const leafNodes = updatedNodes.filter(n => n.children.length === 0);
      const leafSpacing = 350 / Math.max(leafNodes.length, 1);
      
      // Set y positions for leaves
      leafNodes.forEach((leaf, index) => {
        const nodeIndex = updatedNodes.findIndex(n => n.id === leaf.id);
        if (nodeIndex >= 0) {
          updatedNodes[nodeIndex].y = 50 + index * leafSpacing;
        }
      });
      
      // Calculate internal node positions
      const calculateNodePosition = (nodeId: string): { x: number, y: number } => {
        const node = updatedNodes.find(n => n.id === nodeId);
        if (!node) return { x: 0, y: 0 };
        
        if (node.children.length === 0) {
          // Leaf node - position already set
          return { x: node.x, y: node.y };
        }
        
        // Internal node - position based on children
        const childPositions = node.children.map(childId => calculateNodePosition(childId));
        const avgY = childPositions.reduce((sum, pos) => sum + pos.y, 0) / childPositions.length;
        const maxX = Math.max(...childPositions.map(pos => pos.x));
        
        const nodeIndex = updatedNodes.findIndex(n => n.id === nodeId);
        if (nodeIndex >= 0) {
          updatedNodes[nodeIndex].x = maxX + 80;
          updatedNodes[nodeIndex].y = avgY;
        }
        
        return { x: maxX + 80, y: avgY };
      };
      
      calculateNodePosition('root');
      
    } else {
      // Calculate circular layout
      const leafNodes = updatedNodes.filter(n => n.children.length === 0);
      const angleStep = (2 * Math.PI) / Math.max(leafNodes.length, 1);
      const radius = 150;
      const centerX = 250;
      const centerY = 200;
      
      leafNodes.forEach((leaf, index) => {
        const angle = index * angleStep;
        const nodeIndex = updatedNodes.findIndex(n => n.id === leaf.id);
        if (nodeIndex >= 0) {
          updatedNodes[nodeIndex].x = centerX + Math.cos(angle) * radius;
          updatedNodes[nodeIndex].y = centerY + Math.sin(angle) * radius;
        }
      });
      
      // Position internal nodes
      const positionInternalNode = (nodeId: string) => {
        const node = updatedNodes.find(n => n.id === nodeId);
        if (!node || node.children.length === 0) return;
        
        const childPositions = node.children.map(childId => {
          const child = updatedNodes.find(n => n.id === childId);
          return child ? { x: child.x, y: child.y } : { x: 0, y: 0 };
        });
        
        const avgX = childPositions.reduce((sum, pos) => sum + pos.x, 0) / childPositions.length;
        const avgY = childPositions.reduce((sum, pos) => sum + pos.y, 0) / childPositions.length;
        
        // Move toward center
        const distanceFromCenter = Math.sqrt((avgX - centerX) ** 2 + (avgY - centerY) ** 2);
        const newDistance = distanceFromCenter * 0.7;
        const angle = Math.atan2(avgY - centerY, avgX - centerX);
        
        const nodeIndex = updatedNodes.findIndex(n => n.id === nodeId);
        if (nodeIndex >= 0) {
          updatedNodes[nodeIndex].x = centerX + Math.cos(angle) * newDistance;
          updatedNodes[nodeIndex].y = centerY + Math.sin(angle) * newDistance;
        }
      };
      
      // Position from leaves toward root
      const processedNodes = new Set<string>();
      const queue = [...leafNodes.map(n => n.id)];
      
      while (queue.length > 0) {
        const nodeId = queue.shift()!;
        const node = updatedNodes.find(n => n.id === nodeId);
        if (!node) continue;
        
        processedNodes.add(nodeId);
        
        // Find parent and check if all siblings are processed
        const parent = updatedNodes.find(n => n.children.includes(nodeId));
        if (parent && !processedNodes.has(parent.id)) {
          const allChildrenProcessed = parent.children.every(childId => processedNodes.has(childId));
          if (allChildrenProcessed) {
            positionInternalNode(parent.id);
            queue.push(parent.id);
          }
        }
      }
    }
    
    setNodes(updatedNodes);
    onTreeUpdate?.(updatedNodes);
  }, [nodes, treeLayout, onTreeUpdate]);

  const addChild = () => {
    if (!nodeName.trim() || !selectedNode) return;
    
    const newId = `node_${Date.now()}`;
    const parent = nodes.find(n => n.id === selectedNode);
    if (!parent) return;
    
    const newNode: TreeNode = {
      id: newId,
      name: nodeName.trim(),
      parent: selectedNode,
      children: [],
      branchLength,
      support,
      x: parent.x + 100,
      y: parent.y + (parent.children.length * 50),
      depth: parent.depth + 1
    };
    
    const updatedNodes = nodes.map(node => 
      node.id === selectedNode 
        ? { ...node, children: [...node.children, newId] }
        : node
    );
    
    setNodes([...updatedNodes, newNode]);
    setNodeName('');
    setBranchLength(1.0);
    setSupport(0.95);
    
    // Recalculate layout after adding node
    setTimeout(calculateLayout, 100);
  };

  const removeNode = (nodeId: string) => {
    if (nodeId === 'root') return;
    
    const nodeToRemove = nodes.find(n => n.id === nodeId);
    if (!nodeToRemove || !nodeToRemove.parent) return;
    
    // Remove from parent's children
    const updatedNodes = nodes
      .filter(n => n.id !== nodeId && !isDescendant(nodeId, n.id))
      .map(node => 
        node.id === nodeToRemove.parent
          ? { ...node, children: node.children.filter(childId => childId !== nodeId) }
          : node
      );
    
    setNodes(updatedNodes);
    if (selectedNode === nodeId) {
      setSelectedNode('root');
    }
    
    setTimeout(calculateLayout, 100);
  };

  const isDescendant = (ancestorId: string, nodeId: string): boolean => {
    const node = nodes.find(n => n.id === nodeId);
    if (!node || !node.parent) return false;
    if (node.parent === ancestorId) return true;
    return isDescendant(ancestorId, node.parent);
  };

  const exportNewick = () => {
    const buildNewick = (nodeId: string): string => {
      const node = nodes.find(n => n.id === nodeId);
      if (!node) return '';

      if (node.children.length === 0) {
        return `${node.name}:${node.branchLength.toFixed(3)}`;
      }

      const childrenNewick = node.children
        .map(childId => buildNewick(childId))
        .join(',');
      
      const supportStr = showSupport && node.id !== 'root' ? node.support.toFixed(2) : '';
      const nameStr = node.id === 'root' ? '' : node.name;
      const branchStr = node.id === 'root' ? '' : `:${node.branchLength.toFixed(3)}`;
      
      return `(${childrenNewick})${supportStr}${nameStr}${branchStr}`;
    };

    const newick = buildNewick('root') + ';';
    
    const blob = new Blob([newick], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tree.newick';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const renderNode = (node: TreeNode) => {
    const isSelected = selectedNode === node.id;
    const isLeaf = node.children.length === 0;
    
    return (
      <g key={node.id}>
        <circle
          cx={node.x}
          cy={node.y}
          r={isLeaf ? 6 : 8}
          fill={isLeaf ? '#4CAF50' : '#2196F3'}
          stroke={isSelected ? '#FFD700' : '#000'}
          strokeWidth={isSelected ? 3 : 1}
          style={{ cursor: 'pointer' }}
          onClick={() => setSelectedNode(node.id)}
        />
        
        <text
          x={node.x + 12}
          y={node.y - 8}
          fontSize="10"
          fill="#333"
          style={{ cursor: 'pointer' }}
          onClick={() => setSelectedNode(node.id)}
        >
          {node.name}
        </text>
        
        {showSupport && node.id !== 'root' && (
          <text
            x={node.x - 10}
            y={node.y - 10}
            fontSize="8"
            fill="#666"
          >
            {node.support.toFixed(2)}
          </text>
        )}
        
        {showBranchLengths && node.parent && (
          <text
            x={(node.x + (nodes.find(n => n.id === node.parent)?.x || 0)) / 2}
            y={node.y - 15}
            fontSize="8"
            fill="#666"
            textAnchor="middle"
          >
            {node.branchLength.toFixed(2)}
          </text>
        )}
        
        {/* Remove button for non-root nodes */}
        {node.id !== 'root' && isSelected && (
          <circle
            cx={node.x + 20}
            cy={node.y - 20}
            r="8"
            fill="#F44336"
            style={{ cursor: 'pointer' }}
            onClick={(e) => {
              e.stopPropagation();
              removeNode(node.id);
            }}
          />
        )}
        {node.id !== 'root' && isSelected && (
          <text
            x={node.x + 20}
            y={node.y - 16}
            fontSize="10"
            fill="white"
            textAnchor="middle"
            style={{ cursor: 'pointer' }}
            onClick={(e) => {
              e.stopPropagation();
              removeNode(node.id);
            }}
          >
            ×
          </text>
        )}
      </g>
    );
  };

  const renderBranch = (node: TreeNode) => {
    if (!node.parent) return null;
    
    const parent = nodes.find(n => n.id === node.parent);
    if (!parent) return null;

    return (
      <line
        key={`branch-${node.id}`}
        x1={parent.x}
        y1={parent.y}
        x2={node.x}
        y2={node.y}
        stroke="#666"
        strokeWidth="2"
      />
    );
  };

  return (
    <div className="tree-builder">
      <h3>Interactive Tree Builder</h3>
      
      <div className="builder-controls">
        <div className="add-node-controls">
          <h4>Add Node</h4>
          <div className="control-group">
            <label>
              Node Name:
              <input
                type="text"
                value={nodeName}
                onChange={(e) => setNodeName(e.target.value)}
                placeholder="Enter node name"
              />
            </label>
          </div>
          
          <div className="control-group">
            <label>
              Branch Length:
              <input
                type="number"
                value={branchLength}
                onChange={(e) => setBranchLength(parseFloat(e.target.value))}
                step="0.1"
                min="0"
              />
            </label>
          </div>
          
          <div className="control-group">
            <label>
              Support Value:
              <input
                type="number"
                value={support}
                onChange={(e) => setSupport(parseFloat(e.target.value))}
                step="0.01"
                min="0"
                max="1"
              />
            </label>
          </div>
          
          <button onClick={addChild} disabled={!nodeName.trim() || !selectedNode}>
            Add Child to Selected Node
          </button>
        </div>
        
        <div className="display-controls">
          <h4>Display Options</h4>
          
          <div className="control-group">
            <label>
              Layout:
              <select 
                value={treeLayout} 
                onChange={(e) => setTreeLayout(e.target.value as 'rectangular' | 'circular')}
              >
                <option value="rectangular">Rectangular</option>
                <option value="circular">Circular</option>
              </select>
            </label>
          </div>
          
          <div className="control-group">
            <label>
              <input
                type="checkbox"
                checked={showSupport}
                onChange={(e) => setShowSupport(e.target.checked)}
              />
              Show Support Values
            </label>
          </div>
          
          <div className="control-group">
            <label>
              <input
                type="checkbox"
                checked={showBranchLengths}
                onChange={(e) => setShowBranchLengths(e.target.checked)}
              />
              Show Branch Lengths
            </label>
          </div>
          
          <button onClick={calculateLayout}>
            Recalculate Layout
          </button>
          
          <button onClick={exportNewick}>
            Export Newick
          </button>
        </div>
      </div>

      <div className="tree-canvas">
        <svg width="600" height="400" style={{ border: '1px solid #ddd' }}>
          {/* Render branches first */}
          {nodes.map(node => renderBranch(node))}
          
          {/* Render nodes on top */}
          {nodes.map(node => renderNode(node))}
          
          {/* Instructions */}
          <text x="10" y="20" fontSize="12" fill="#666">
            Click nodes to select, then add children. Click red × to remove nodes.
          </text>
          
          {selectedNode && (
            <text x="10" y="35" fontSize="12" fill="#333">
              Selected: {nodes.find(n => n.id === selectedNode)?.name || 'None'}
            </text>
          )}
        </svg>
      </div>

      <div className="tree-stats">
        <h4>Tree Statistics</h4>
        <div className="stats">
          <div>Total Nodes: {nodes.length}</div>
          <div>Leaf Nodes: {nodes.filter(n => n.children.length === 0).length}</div>
          <div>Internal Nodes: {nodes.filter(n => n.children.length > 0).length}</div>
          <div>Max Depth: {Math.max(...nodes.map(n => n.depth))}</div>
        </div>
      </div>
      
      <style jsx>{`
        .tree-builder {
          padding: 20px;
          font-family: Arial, sans-serif;
        }
        
        .builder-controls {
          display: flex;
          gap: 30px;
          margin: 20px 0;
          flex-wrap: wrap;
        }
        
        .add-node-controls, .display-controls {
          border: 1px solid #ddd;
          padding: 15px;
          border-radius: 8px;
          background-color: #f9f9f9;
        }
        
        .control-group {
          margin: 10px 0;
        }
        
        .control-group label {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }
        
        .control-group input, .control-group select {
          padding: 5px;
          border: 1px solid #ccc;
          border-radius: 4px;
        }
        
        .tree-canvas {
          margin: 20px 0;
        }
        
        .tree-stats {
          margin: 20px 0;
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 8px;
          background-color: #f0f8ff;
        }
        
        .stats {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 10px;
          margin-top: 10px;
        }
        
        button {
          padding: 8px 16px;
          background-color: #2196F3;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          margin: 5px 0;
        }
        
        button:hover {
          background-color: #1976D2;
        }
        
        button:disabled {
          background-color: #ccc;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default TreeBuilder;