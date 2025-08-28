/**
 * VIZ-003: Physiology Monitor
 * Real-time physiological parameter monitoring with alerts and trends
 */

import React, { useState, useEffect } from 'react';

interface PhysiologyReading {
  timestamp: number;
  heartRate: number;
  oxygenSaturation: number;
  respiratoryRate: number;
  bodyTemperature: number;
  bloodPressureSystolic: number;
  bloodPressureDiastolic: number;
  metabolicRate: number;
  co2Levels: number;
}

interface PhysiologyMonitorProps {
  realTimeMode?: boolean;
  alertThresholds?: {
    heartRateMin: number;
    heartRateMax: number;
    oxygenSatMin: number;
    tempMin: number;
    tempMax: number;
  };
}

export const PhysiologyMonitor: React.FC<PhysiologyMonitorProps> = ({
  realTimeMode = false,
  alertThresholds = {
    heartRateMin: 60,
    heartRateMax: 100,
    oxygenSatMin: 95,
    tempMin: 36.1,
    tempMax: 37.5
  }
}) => {
  const [currentReading, setCurrentReading] = useState<PhysiologyReading>({
    timestamp: Date.now(),
    heartRate: 72,
    oxygenSaturation: 98,
    respiratoryRate: 16,
    bodyTemperature: 37.0,
    bloodPressureSystolic: 120,
    bloodPressureDiastolic: 80,
    metabolicRate: 1800,
    co2Levels: 40
  });

  const [readings, setReadings] = useState<PhysiologyReading[]>([currentReading]);
  const [alerts, setAlerts] = useState<string[]>([]);

  useEffect(() => {
    if (realTimeMode) {
      const interval = setInterval(() => {
        const newReading: PhysiologyReading = {
          timestamp: Date.now(),
          heartRate: Math.max(50, Math.min(120, currentReading.heartRate + (Math.random() - 0.5) * 4)),
          oxygenSaturation: Math.max(85, Math.min(100, currentReading.oxygenSaturation + (Math.random() - 0.5) * 2)),
          respiratoryRate: Math.max(10, Math.min(25, currentReading.respiratoryRate + (Math.random() - 0.5) * 2)),
          bodyTemperature: Math.max(35.5, Math.min(38.5, currentReading.bodyTemperature + (Math.random() - 0.5) * 0.2)),
          bloodPressureSystolic: Math.max(90, Math.min(150, currentReading.bloodPressureSystolic + (Math.random() - 0.5) * 8)),
          bloodPressureDiastolic: Math.max(60, Math.min(100, currentReading.bloodPressureDiastolic + (Math.random() - 0.5) * 4)),
          metabolicRate: Math.max(1200, Math.min(2500, currentReading.metabolicRate + (Math.random() - 0.5) * 100)),
          co2Levels: Math.max(30, Math.min(50, currentReading.co2Levels + (Math.random() - 0.5) * 3))
        };

        setCurrentReading(newReading);
        setReadings(prev => [...prev.slice(-29), newReading]); // Keep last 30 readings

        // Check for alerts
        const newAlerts: string[] = [];
        if (newReading.heartRate < alertThresholds.heartRateMin || newReading.heartRate > alertThresholds.heartRateMax) {
          newAlerts.push(`Heart rate abnormal: ${newReading.heartRate.toFixed(0)} bpm`);
        }
        if (newReading.oxygenSaturation < alertThresholds.oxygenSatMin) {
          newAlerts.push(`Low oxygen saturation: ${newReading.oxygenSaturation.toFixed(1)}%`);
        }
        if (newReading.bodyTemperature < alertThresholds.tempMin || newReading.bodyTemperature > alertThresholds.tempMax) {
          newAlerts.push(`Body temperature abnormal: ${newReading.bodyTemperature.toFixed(1)}°C`);
        }
        setAlerts(newAlerts);
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [realTimeMode, currentReading, alertThresholds]);

  const getStatusColor = (value: number, min: number, max: number, reverse = false) => {
    if (reverse) {
      return value < min ? '#F44336' : value > max ? '#F44336' : '#4CAF50';
    }
    return value >= min && value <= max ? '#4CAF50' : '#F44336';
  };

  const renderTrendChart = (data: number[], label: string, unit: string, color: string) => {
    const maxVal = Math.max(...data);
    const minVal = Math.min(...data);
    const range = maxVal - minVal || 1;
    
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * 280;
      const y = 60 - ((value - minVal) / range) * 50;
      return `${x},${y}`;
    }).join(' ');

    return (
      <div className="trend-chart">
        <h4>{label}</h4>
        <svg width="300" height="80" style={{ border: '1px solid #ddd' }}>
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth="2"
          />
          {/* Grid lines */}
          <line x1="0" y1="10" x2="280" y2="10" stroke="#eee" strokeWidth="1" />
          <line x1="0" y1="30" x2="280" y2="30" stroke="#eee" strokeWidth="1" />
          <line x1="0" y1="50" x2="280" y2="50" stroke="#eee" strokeWidth="1" />
          
          {/* Current value */}
          <text x="285" y="15" fontSize="12" fill={color}>
            {data[data.length - 1]?.toFixed(1)}{unit}
          </text>
        </svg>
      </div>
    );
  };

  return (
    <div className="physiology-monitor">
      <h2>Physiology Monitor</h2>
      
      <div className="monitor-controls">
        <button 
          onClick={() => setReadings([currentReading])}
          style={{ marginRight: '10px' }}
        >
          Reset Data
        </button>
        <span className={realTimeMode ? 'status-active' : 'status-inactive'}>
          {realTimeMode ? '● LIVE' : '○ PAUSED'}
        </span>
      </div>

      {alerts.length > 0 && (
        <div className="alerts-panel">
          <h3>⚠️ Alerts</h3>
          {alerts.map((alert, index) => (
            <div key={index} className="alert-item">
              {alert}
            </div>
          ))}
        </div>
      )}

      <div className="vital-signs-grid">
        <div className="vital-sign">
          <h4>Heart Rate</h4>
          <div 
            className="vital-value"
            style={{ 
              color: getStatusColor(
                currentReading.heartRate, 
                alertThresholds.heartRateMin, 
                alertThresholds.heartRateMax
              )
            }}
          >
            {currentReading.heartRate.toFixed(0)} bpm
          </div>
          <div className="vital-status">
            {currentReading.heartRate >= alertThresholds.heartRateMin && 
             currentReading.heartRate <= alertThresholds.heartRateMax ? 'Normal' : 'Abnormal'}
          </div>
        </div>

        <div className="vital-sign">
          <h4>O₂ Saturation</h4>
          <div 
            className="vital-value"
            style={{ 
              color: getStatusColor(currentReading.oxygenSaturation, alertThresholds.oxygenSatMin, 100)
            }}
          >
            {currentReading.oxygenSaturation.toFixed(1)}%
          </div>
          <div className="vital-status">
            {currentReading.oxygenSaturation >= alertThresholds.oxygenSatMin ? 'Normal' : 'Low'}
          </div>
        </div>

        <div className="vital-sign">
          <h4>Respiratory Rate</h4>
          <div className="vital-value" style={{ color: '#2196F3' }}>
            {currentReading.respiratoryRate.toFixed(0)} /min
          </div>
          <div className="vital-status">
            {currentReading.respiratoryRate >= 12 && currentReading.respiratoryRate <= 20 ? 'Normal' : 'Monitor'}
          </div>
        </div>

        <div className="vital-sign">
          <h4>Body Temperature</h4>
          <div 
            className="vital-value"
            style={{ 
              color: getStatusColor(
                currentReading.bodyTemperature, 
                alertThresholds.tempMin, 
                alertThresholds.tempMax
              )
            }}
          >
            {currentReading.bodyTemperature.toFixed(1)}°C
          </div>
          <div className="vital-status">
            {currentReading.bodyTemperature >= alertThresholds.tempMin && 
             currentReading.bodyTemperature <= alertThresholds.tempMax ? 'Normal' : 'Abnormal'}
          </div>
        </div>

        <div className="vital-sign">
          <h4>Blood Pressure</h4>
          <div className="vital-value" style={{ color: '#9C27B0' }}>
            {currentReading.bloodPressureSystolic.toFixed(0)}/
            {currentReading.bloodPressureDiastolic.toFixed(0)} mmHg
          </div>
          <div className="vital-status">
            {currentReading.bloodPressureSystolic < 140 && currentReading.bloodPressureDiastolic < 90 ? 'Normal' : 'Elevated'}
          </div>
        </div>

        <div className="vital-sign">
          <h4>Metabolic Rate</h4>
          <div className="vital-value" style={{ color: '#FF9800' }}>
            {currentReading.metabolicRate.toFixed(0)} kcal/day
          </div>
          <div className="vital-status">Active</div>
        </div>
      </div>

      {readings.length > 1 && (
        <div className="trends-section">
          <h3>Trends (Last {readings.length} readings)</h3>
          <div className="trends-grid">
            {renderTrendChart(
              readings.map(r => r.heartRate), 
              'Heart Rate', 
              ' bpm', 
              '#F44336'
            )}
            {renderTrendChart(
              readings.map(r => r.oxygenSaturation), 
              'O₂ Saturation', 
              '%', 
              '#2196F3'
            )}
            {renderTrendChart(
              readings.map(r => r.bodyTemperature), 
              'Temperature', 
              '°C', 
              '#FF9800'
            )}
            {renderTrendChart(
              readings.map(r => r.metabolicRate), 
              'Metabolic Rate', 
              ' kcal', 
              '#4CAF50'
            )}
          </div>
        </div>
      )}
      
      <style jsx>{`
        .physiology-monitor {
          padding: 20px;
          font-family: Arial, sans-serif;
          background-color: #f5f5f5;
        }
        
        .monitor-controls {
          margin: 20px 0;
          display: flex;
          align-items: center;
        }
        
        .status-active {
          color: #4CAF50;
          font-weight: bold;
        }
        
        .status-inactive {
          color: #757575;
        }
        
        .alerts-panel {
          background-color: #ffebee;
          border: 2px solid #f44336;
          border-radius: 8px;
          padding: 15px;
          margin: 15px 0;
        }
        
        .alert-item {
          color: #d32f2f;
          font-weight: bold;
          margin: 5px 0;
        }
        
        .vital-signs-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 15px;
          margin: 20px 0;
        }
        
        .vital-sign {
          background-color: white;
          border-radius: 8px;
          padding: 15px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          text-align: center;
        }
        
        .vital-value {
          font-size: 24px;
          font-weight: bold;
          margin: 10px 0;
        }
        
        .vital-status {
          font-size: 14px;
          color: #757575;
        }
        
        .trends-section {
          background-color: white;
          border-radius: 8px;
          padding: 20px;
          margin-top: 20px;
        }
        
        .trends-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 20px;
          margin-top: 15px;
        }
        
        .trend-chart h4 {
          margin: 0 0 10px 0;
          color: #333;
        }
      `}</style>
    </div>
  );
};

export default PhysiologyMonitor;