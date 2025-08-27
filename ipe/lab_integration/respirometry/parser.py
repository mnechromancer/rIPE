"""
DATA-001: Respirometry Data Import - Generic Parser

This module provides generic parsing functionality for respirometry data
from various manufacturers and formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
from datetime import datetime

from .sable_import import SableSessionData, RespirometryMeasurement


class RespirometryParser(ABC):
    """
    Abstract base class for respirometry data parsers
    
    Defines the interface that all respirometry parsers must implement
    """
    
    @abstractmethod
    def parse_file(self, filepath: Union[str, Path]) -> SableSessionData:
        """Parse a single respirometry file"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions"""
        pass
    
    @abstractmethod
    def validate_format(self, filepath: Union[str, Path]) -> bool:
        """Check if file format is supported by this parser"""
        pass


@dataclass
class MetabolicSummary:
    """Summary statistics for metabolic measurements"""
    mean_vo2: float
    std_vo2: float
    mean_vco2: float
    std_vco2: float
    mean_rer: float
    std_rer: float
    min_vo2: float
    max_vo2: float
    min_vco2: float
    max_vco2: float
    measurement_count: int
    duration_minutes: float
    
    @property
    def vo2_coefficient_variation(self) -> float:
        """Calculate coefficient of variation for VO2"""
        return (self.std_vo2 / self.mean_vo2) * 100 if self.mean_vo2 > 0 else 0
    
    @property
    def vco2_coefficient_variation(self) -> float:
        """Calculate coefficient of variation for VCO2"""
        return (self.std_vco2 / self.mean_vco2) * 100 if self.mean_vco2 > 0 else 0


class GenericRespirometryParser:
    """
    Generic parser that can handle multiple respirometry formats
    
    Automatically detects format and uses appropriate parser
    """
    
    def __init__(self):
        self.parsers = {}
        self._register_parsers()
    
    def _register_parsers(self):
        """Register available parsers"""
        from .sable_import import SableSystemsImporter
        
        # Register Sable Systems parser
        sable_parser = SableFormatParser(SableSystemsImporter())
        for fmt in sable_parser.get_supported_formats():
            self.parsers[fmt] = sable_parser
    
    def parse_file(self, filepath: Union[str, Path]) -> SableSessionData:
        """
        Parse respirometry file using appropriate parser
        
        Args:
            filepath: Path to respirometry file
            
        Returns:
            Parsed session data
            
        Raises:
            ValueError: If file format not supported
        """
        filepath = Path(filepath)
        extension = filepath.suffix.lower()
        
        if extension not in self.parsers:
            raise ValueError(f"Unsupported file format: {extension}")
        
        parser = self.parsers[extension]
        return parser.parse_file(filepath)
    
    def get_supported_formats(self) -> List[str]:
        """Get all supported file formats"""
        return list(self.parsers.keys())
    
    def calculate_summary(self, session_data: SableSessionData) -> MetabolicSummary:
        """
        Calculate summary statistics for a session
        
        Args:
            session_data: Session data to summarize
            
        Returns:
            Summary statistics
        """
        measurements = session_data.measurements
        
        if not measurements:
            raise ValueError("No measurements to summarize")
        
        vo2_values = [m.vo2_ml_min for m in measurements]
        vco2_values = [m.vco2_ml_min for m in measurements]
        rer_values = [m.rer for m in measurements]
        
        return MetabolicSummary(
            mean_vo2=np.mean(vo2_values),
            std_vo2=np.std(vo2_values),
            mean_vco2=np.mean(vco2_values),
            std_vco2=np.std(vco2_values),
            mean_rer=np.mean(rer_values),
            std_rer=np.std(rer_values),
            min_vo2=np.min(vo2_values),
            max_vo2=np.max(vo2_values),
            min_vco2=np.min(vco2_values),
            max_vco2=np.max(vco2_values),
            measurement_count=len(measurements),
            duration_minutes=session_data.duration_minutes
        )
    
    def filter_measurements(self, session_data: SableSessionData,
                          min_vo2: Optional[float] = None,
                          max_vo2: Optional[float] = None,
                          min_rer: Optional[float] = None,
                          max_rer: Optional[float] = None) -> SableSessionData:
        """
        Filter measurements based on quality criteria
        
        Args:
            session_data: Original session data
            min_vo2: Minimum acceptable VO2
            max_vo2: Maximum acceptable VO2
            min_rer: Minimum acceptable RER
            max_rer: Maximum acceptable RER
            
        Returns:
            New session data with filtered measurements
        """
        filtered_measurements = []
        
        for measurement in session_data.measurements:
            # Apply filters
            if min_vo2 is not None and measurement.vo2_ml_min < min_vo2:
                continue
            if max_vo2 is not None and measurement.vo2_ml_min > max_vo2:
                continue
            if min_rer is not None and measurement.rer < min_rer:
                continue
            if max_rer is not None and measurement.rer > max_rer:
                continue
            
            filtered_measurements.append(measurement)
        
        # Create new session data with filtered measurements
        return SableSessionData(
            session_id=session_data.session_id,
            subject_id=session_data.subject_id,
            start_time=filtered_measurements[0].timestamp if filtered_measurements else session_data.start_time,
            end_time=filtered_measurements[-1].timestamp if filtered_measurements else session_data.end_time,
            measurements=filtered_measurements,
            metadata={**session_data.metadata, 'filtered': True},
            baseline_period=session_data.baseline_period
        )
    
    def smooth_measurements(self, session_data: SableSessionData,
                          window_size: int = 5) -> SableSessionData:
        """
        Apply smoothing to measurements using moving average
        
        Args:
            session_data: Original session data
            window_size: Size of smoothing window (odd number recommended)
            
        Returns:
            New session data with smoothed measurements
        """
        measurements = session_data.measurements
        if len(measurements) < window_size:
            return session_data  # Not enough data to smooth
        
        smoothed_measurements = []
        half_window = window_size // 2
        
        for i in range(len(measurements)):
            # Define window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(len(measurements), i + half_window + 1)
            
            # Calculate smoothed values
            window_measurements = measurements[start_idx:end_idx]
            smooth_vo2 = np.mean([m.vo2_ml_min for m in window_measurements])
            smooth_vco2 = np.mean([m.vco2_ml_min for m in window_measurements])
            smooth_rer = smooth_vco2 / smooth_vo2 if smooth_vo2 > 0 else measurements[i].rer
            
            # Create smoothed measurement
            smoothed_measurement = RespirometryMeasurement(
                timestamp=measurements[i].timestamp,
                vo2_ml_min=smooth_vo2,
                vco2_ml_min=smooth_vco2,
                rer=smooth_rer,
                chamber_temp_c=measurements[i].chamber_temp_c,
                chamber_humidity_percent=measurements[i].chamber_humidity_percent,
                ambient_pressure_kpa=measurements[i].ambient_pressure_kpa,
                flow_rate_ml_min=measurements[i].flow_rate_ml_min,
                baseline_corrected=measurements[i].baseline_corrected
            )
            
            smoothed_measurements.append(smoothed_measurement)
        
        # Create new session data with smoothed measurements
        return SableSessionData(
            session_id=session_data.session_id,
            subject_id=session_data.subject_id,
            start_time=session_data.start_time,
            end_time=session_data.end_time,
            measurements=smoothed_measurements,
            metadata={**session_data.metadata, 'smoothed': True, 'smooth_window': window_size},
            baseline_period=session_data.baseline_period
        )


class SableFormatParser(RespirometryParser):
    """Parser for Sable Systems format files"""
    
    def __init__(self, sable_importer):
        self.importer = sable_importer
    
    def parse_file(self, filepath: Union[str, Path]) -> SableSessionData:
        """Parse Sable Systems file"""
        return self.importer.import_session(filepath)
    
    def get_supported_formats(self) -> List[str]:
        """Return supported Sable formats"""
        return ['.exp', '.csv']
    
    def validate_format(self, filepath: Union[str, Path]) -> bool:
        """Check if file is a valid Sable format"""
        filepath = Path(filepath)
        return filepath.suffix.lower() in self.get_supported_formats()


class QualityAssessment:
    """
    Quality assessment tools for respirometry data
    """
    
    @staticmethod
    def assess_data_quality(session_data: SableSessionData) -> Dict[str, Any]:
        """
        Assess overall data quality of a respirometry session
        
        Returns:
            Dictionary with quality metrics and recommendations
        """
        measurements = session_data.measurements
        
        if not measurements:
            return {'quality_score': 0, 'issues': ['No measurements found']}
        
        issues = []
        quality_score = 100  # Start with perfect score
        
        # Check measurement count
        if len(measurements) < 60:  # Less than 1 minute at 1Hz
            issues.append('Very short recording duration')
            quality_score -= 30
        elif len(measurements) < 300:  # Less than 5 minutes
            issues.append('Short recording duration')
            quality_score -= 10
        
        # Check for missing data
        missing_flow = sum(1 for m in measurements if m.flow_rate_ml_min is None)
        if missing_flow > len(measurements) * 0.1:  # More than 10% missing
            issues.append('Significant missing flow rate data')
            quality_score -= 15
        
        # Check VO2 stability
        vo2_values = [m.vo2_ml_min for m in measurements]
        vo2_cv = (np.std(vo2_values) / np.mean(vo2_values)) * 100
        if vo2_cv > 20:
            issues.append('High VO2 variability (CV > 20%)')
            quality_score -= 15
        elif vo2_cv > 10:
            issues.append('Moderate VO2 variability (CV > 10%)')
            quality_score -= 5
        
        # Check RER values
        rer_values = [m.rer for m in measurements]
        invalid_rer = sum(1 for rer in rer_values if rer < 0.5 or rer > 1.5)
        if invalid_rer > len(measurements) * 0.05:  # More than 5% invalid
            issues.append('Significant physiologically unrealistic RER values')
            quality_score -= 20
        
        # Check temperature stability
        temp_values = [m.chamber_temp_c for m in measurements]
        temp_range = max(temp_values) - min(temp_values)
        if temp_range > 5:
            issues.append('Large temperature fluctuations (>5°C)')
            quality_score -= 10
        elif temp_range > 2:
            issues.append('Moderate temperature fluctuations (>2°C)')
            quality_score -= 5
        
        # Overall quality rating
        if quality_score >= 90:
            quality_rating = 'Excellent'
        elif quality_score >= 80:
            quality_rating = 'Good'
        elif quality_score >= 70:
            quality_rating = 'Fair'
        elif quality_score >= 60:
            quality_rating = 'Poor'
        else:
            quality_rating = 'Very Poor'
        
        return {
            'quality_score': max(0, quality_score),
            'quality_rating': quality_rating,
            'issues': issues,
            'metrics': {
                'measurement_count': len(measurements),
                'duration_minutes': session_data.duration_minutes,
                'vo2_cv_percent': vo2_cv,
                'temperature_range_c': temp_range,
                'invalid_rer_percent': (invalid_rer / len(measurements)) * 100,
                'missing_flow_percent': (missing_flow / len(measurements)) * 100
            }
        }
    
    @staticmethod
    def recommend_processing(quality_assessment: Dict[str, Any]) -> List[str]:
        """
        Recommend processing steps based on quality assessment
        
        Args:
            quality_assessment: Output from assess_data_quality
            
        Returns:
            List of recommended processing steps
        """
        recommendations = []
        issues = quality_assessment.get('issues', [])
        metrics = quality_assessment.get('metrics', {})
        
        if 'High VO2 variability' in ' '.join(issues):
            recommendations.append('Apply smoothing filter to reduce noise')
        
        if 'temperature fluctuations' in ' '.join(issues).lower():
            recommendations.append('Enable temperature correction')
        
        if metrics.get('invalid_rer_percent', 0) > 5:
            recommendations.append('Apply RER filtering to remove outliers')
        
        if 'Short recording' in ' '.join(issues):
            recommendations.append('Consider longer recording duration for better statistics')
        
        if quality_assessment.get('quality_score', 0) < 70:
            recommendations.append('Consider data validation before analysis')
        
        return recommendations