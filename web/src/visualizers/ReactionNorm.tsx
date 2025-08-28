/**
 * VIZ-002: Reaction Norm Visualizer
 * Displays reaction norms with G×E interactions and plasticity patterns
 */

import React, { useMemo } from 'react';

interface ReactionNormProps {
  genotypes?: Array<{
    id: string;
    name: string;
    color: string;
  }>;
  environments?: number[];
  responses?: Array<Array<number>>;
  showAssimilation?: boolean;
}

export const ReactionNorm: React.FC<ReactionNormProps> = ({
  genotypes = [
    { id: 'g1', name: 'Genotype A', color: '#ff0000' },
    { id: 'g2', name: 'Genotype B', color: '#00ff00' },
    { id: 'g3', name: 'Genotype C', color: '#0000ff' }
  ],
  environments = [-2, -1, 0, 1, 2],
  responses = [
    [0.2, 0.4, 0.6, 0.8, 1.0],  // Genotype A
    [0.8, 0.7, 0.6, 0.5, 0.4],  // Genotype B
    [0.5, 0.5, 0.5, 0.5, 0.5]   // Genotype C (no plasticity)
  ],
  showAssimilation = false
}) => {
  const svgWidth = 500;
  const svgHeight = 300;
  const margin = { top: 20, right: 20, bottom: 40, left: 40 };
  const plotWidth = svgWidth - margin.left - margin.right;
  const plotHeight = svgHeight - margin.top - margin.bottom;

  const paths = useMemo(() => {
    return genotypes.map((genotype, gIndex) => {
      const points = environments.map((env, eIndex) => {
        const x = margin.left + (eIndex / (environments.length - 1)) * plotWidth;
        const response = responses[gIndex] ? responses[gIndex][eIndex] : 0.5;
        const y = margin.top + (1 - response) * plotHeight;
        return `${x},${y}`;
      }).join(' ');
      
      return {
        genotype,
        points,
        pathData: `M ${points.replace(/ /g, ' L ')}`
      };
    });
  }, [genotypes, environments, responses]);

  const xScale = (envIndex: number) => {
    return margin.left + (envIndex / (environments.length - 1)) * plotWidth;
  };

  const yScale = (response: number) => {
    return margin.top + (1 - response) * plotHeight;
  };

  return (
    <div className="reaction-norm">
      <h3>Reaction Norm Visualization</h3>
      
      <div className="plot-container">
        <svg width={svgWidth} height={svgHeight}>
          {/* Background grid */}
          <defs>
            <pattern id="grid" width="40" height="30" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 30" fill="none" stroke="#e0e0e0" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width={svgWidth} height={svgHeight} fill="url(#grid)" />
          
          {/* Axes */}
          <line 
            x1={margin.left} 
            y1={margin.top} 
            x2={margin.left} 
            y2={svgHeight - margin.bottom} 
            stroke="black" 
            strokeWidth="2"
          />
          <line 
            x1={margin.left} 
            y1={svgHeight - margin.bottom} 
            x2={svgWidth - margin.right} 
            y2={svgHeight - margin.bottom} 
            stroke="black" 
            strokeWidth="2"
          />
          
          {/* Axis labels */}
          <text 
            x={svgWidth / 2} 
            y={svgHeight - 5} 
            textAnchor="middle" 
            fontSize="12"
          >
            Environment
          </text>
          <text 
            x={15} 
            y={svgHeight / 2} 
            textAnchor="middle" 
            fontSize="12" 
            transform={`rotate(-90, 15, ${svgHeight / 2})`}
          >
            Phenotype Response
          </text>
          
          {/* Environment tick marks and labels */}
          {environments.map((env, index) => (
            <g key={index}>
              <line 
                x1={xScale(index)} 
                y1={svgHeight - margin.bottom} 
                x2={xScale(index)} 
                y2={svgHeight - margin.bottom + 5} 
                stroke="black"
              />
              <text 
                x={xScale(index)} 
                y={svgHeight - margin.bottom + 18} 
                textAnchor="middle" 
                fontSize="10"
              >
                {env}
              </text>
            </g>
          ))}
          
          {/* Response tick marks and labels */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((response, index) => (
            <g key={index}>
              <line 
                x1={margin.left - 5} 
                y1={yScale(response)} 
                x2={margin.left} 
                y2={yScale(response)} 
                stroke="black"
              />
              <text 
                x={margin.left - 10} 
                y={yScale(response) + 3} 
                textAnchor="end" 
                fontSize="10"
              >
                {response.toFixed(2)}
              </text>
            </g>
          ))}
          
          {/* Reaction norm lines */}
          {paths.map(({ genotype, pathData }, index) => (
            <g key={genotype.id}>
              <path
                d={pathData}
                fill="none"
                stroke={genotype.color}
                strokeWidth="3"
                opacity={showAssimilation ? 0.7 : 1.0}
              />
              {/* Data points */}
              {environments.map((env, eIndex) => {
                const response = responses[index] ? responses[index][eIndex] : 0.5;
                return (
                  <circle
                    key={eIndex}
                    cx={xScale(eIndex)}
                    cy={yScale(response)}
                    r="4"
                    fill={genotype.color}
                    stroke="white"
                    strokeWidth="1"
                  />
                );
              })}
            </g>
          ))}
          
          {/* Genetic Assimilation visualization */}
          {showAssimilation && (
            <g opacity="0.5">
              <text x={svgWidth / 2} y={margin.top / 2} textAnchor="middle" fontSize="11" fill="red">
                Genetic Assimilation in Progress...
              </text>
            </g>
          )}
        </svg>
      </div>
      
      <div className="legend">
        <h4>Genotypes</h4>
        {genotypes.map(genotype => (
          <div key={genotype.id} className="legend-item">
            <span 
              className="color-box" 
              style={{ 
                backgroundColor: genotype.color,
                width: '15px',
                height: '15px',
                display: 'inline-block',
                marginRight: '8px'
              }}
            ></span>
            {genotype.name}
          </div>
        ))}
      </div>
      
      <div className="interpretation">
        <h4>Interpretation</h4>
        <p>
          • Flat lines indicate no plasticity (constitutive traits)
        </p>
        <p>
          • Sloped lines show plastic responses to environment
        </p>
        <p>
          • Crossing lines indicate genotype × environment interactions
        </p>
        {showAssimilation && (
          <p style={{ color: 'red' }}>
            • Genetic assimilation: plastic responses becoming constitutive
          </p>
        )}
      </div>
    </div>
  );
};

export default ReactionNorm;