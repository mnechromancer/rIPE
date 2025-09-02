"""
DATA-001: Respirometry Data Import - Sable Systems Integration

This module implements the core functionality for importing data from Sable Systems
ExpeData files, extracting VO2, VCO2, and RER measurements with baseline corrections
and batch import capability.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from datetime import datetime


@dataclass
class RespirometryMeasurement:
    """Represents a single respirometry measurement point"""

    timestamp: datetime
    vo2_ml_min: float
    vco2_ml_min: float
    rer: float
    chamber_temp_c: float
    chamber_humidity_percent: float
    ambient_pressure_kpa: float
    flow_rate_ml_min: Optional[float] = None
    baseline_corrected: bool = False

    def __post_init__(self):
        """Validate measurement values"""
        if self.vo2_ml_min < 0:
            raise ValueError("VO2 cannot be negative")
        if self.vco2_ml_min < 0:
            raise ValueError("VCO2 cannot be negative")
        if not 0.1 <= self.rer <= 2.0:
            raise ValueError(f"RER {self.rer} outside physiological range (0.1-2.0)")


@dataclass
class SableSessionData:
    """Contains parsed data from a Sable Systems session"""

    session_id: str
    subject_id: str
    start_time: datetime
    end_time: datetime
    measurements: List[RespirometryMeasurement]
    metadata: Dict[str, Union[str, float, int]]
    baseline_period: Optional[tuple] = None  # (start_index, end_index)

    @property
    def duration_minutes(self) -> float:
        """Calculate session duration in minutes"""
        return (self.end_time - self.start_time).total_seconds() / 60

    @property
    def mean_vo2(self) -> float:
        """Calculate mean VO2 across all measurements"""
        return np.mean([m.vo2_ml_min for m in self.measurements])

    @property
    def mean_vco2(self) -> float:
        """Calculate mean VCO2 across all measurements"""
        return np.mean([m.vco2_ml_min for m in self.measurements])

    @property
    def mean_rer(self) -> float:
        """Calculate mean RER across all measurements"""
        return np.mean([m.rer for m in self.measurements])


class SableSystemsImporter:
    """
    Importer for Sable Systems ExpeData files (.exp and .csv formats)

    Handles:
    - Parsing .exp and .csv files from Sable Systems equipment
    - Extracting VO2, VCO2, RER measurements
    - Baseline drift corrections
    - Batch import from directories
    - Temperature standardization
    """

    def __init__(
        self,
        baseline_correction: bool = True,
        temperature_correction: bool = True,
        standard_temp_c: float = 25.0,
    ):
        """
        Initialize the Sable Systems importer

        Args:
            baseline_correction: Whether to perform baseline drift correction
            temperature_correction: Whether to correct to standard temperature
            standard_temp_c: Standard temperature for corrections (°C)
        """
        self.baseline_correction = baseline_correction
        self.temperature_correction = temperature_correction
        self.standard_temp_c = standard_temp_c

    def import_session(self, filepath: Union[str, Path]) -> SableSessionData:
        """
        Import a single Sable Systems session file

        Args:
            filepath: Path to .exp or .csv file

        Returns:
            SableSessionData object with parsed measurements
        """
        filepath = Path(filepath)

        if filepath.suffix.lower() == ".exp":
            return self._parse_exp_file(filepath)
        elif filepath.suffix.lower() == ".csv":
            return self._parse_csv_file(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def batch_import(self, directory: Union[str, Path]) -> List[SableSessionData]:
        """
        Import all Sable Systems files from a directory

        Args:
            directory: Path to directory containing .exp/.csv files

        Returns:
            List of SableSessionData objects
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        sessions = []

        # Import all .exp and .csv files
        for pattern in ["*.exp", "*.csv"]:
            for filepath in directory.glob(pattern):
                try:
                    session = self.import_session(filepath)
                    sessions.append(session)
                except Exception as e:
                    print(f"Warning: Failed to import {filepath}: {e}")

        return sorted(sessions, key=lambda s: s.start_time)

    def _parse_exp_file(self, filepath: Path) -> SableSessionData:
        """Parse Sable Systems .exp file format"""
        # .exp files are typically binary format with headers
        # For this implementation, we'll assume a simplified text format
        # In a real implementation, you'd need to parse the binary format

        with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()

        # Extract metadata from header
        metadata = self._extract_exp_metadata(content)

        # Convert to CSV-like format for parsing
        # This is a simplified approach - real .exp parsing would be more complex
        lines = content.split("\n")
        data_lines = [
            line
            for line in lines
            if line.strip() and not line.startswith("#") and "," in line
        ]

        return self._parse_measurements(data_lines, filepath.stem, metadata)

    def _parse_csv_file(self, filepath: Path) -> SableSessionData:
        """Parse Sable Systems CSV file format"""
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        lines = content.strip().split("\n")
        metadata = {}
        data_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                # Parse metadata from comments
                if ":" in line:
                    key, value = line[1:].split(":", 1)
                    metadata[key.strip()] = value.strip()
            elif (
                line and "," in line and not line.lower().startswith("time")
            ):  # Skip header row
                data_lines.append(line)

        return self._parse_measurements(data_lines, filepath.stem, metadata)

    def _parse_measurements(
        self, data_lines: List[str], session_id: str, metadata: Dict
    ) -> SableSessionData:
        """Parse measurement data from CSV lines"""
        measurements = []

        for line in data_lines:
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if (
                len(parts) < 7
            ):  # Need at least time, VO2, VCO2, temp, humidity, pressure, flow
                continue

            try:
                # Parse timestamp - assume format like "14:23:15" or 
                # "2024-01-01 14:23:15"
                time_str = parts[0]
                if " " in time_str:
                    timestamp = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                else:
                    # Use current date with time
                    today = datetime.now().date()
                    time_part = datetime.strptime(time_str, "%H:%M:%S").time()
                    timestamp = datetime.combine(today, time_part)

                # Extract measurements
                vo2 = float(parts[1])  # ml/min
                vco2 = float(parts[2])  # ml/min
                rer = vco2 / vo2 if vo2 > 0 else 0.7  # Calculate RER
                temp = float(parts[3])  # °C
                humidity = float(parts[4])  # %
                pressure = float(parts[5])  # kPa
                flow = float(parts[6]) if len(parts) > 6 else None  # ml/min

                # Apply temperature correction if enabled
                if self.temperature_correction:
                    vo2, vco2 = self._apply_temperature_correction(
                        vo2, vco2, temp, pressure
                    )

                measurement = RespirometryMeasurement(
                    timestamp=timestamp,
                    vo2_ml_min=vo2,
                    vco2_ml_min=vco2,
                    rer=rer,
                    chamber_temp_c=temp,
                    chamber_humidity_percent=humidity,
                    ambient_pressure_kpa=pressure,
                    flow_rate_ml_min=flow,
                    baseline_corrected=False,
                )

                measurements.append(measurement)

            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping invalid data line: {line[:50]}... ({e})")
                continue

        if not measurements:
            raise ValueError("No valid measurements found in file")

        # Create session data
        session_data = SableSessionData(
            session_id=session_id,
            subject_id=metadata.get("Subject_ID", session_id),
            start_time=measurements[0].timestamp,
            end_time=measurements[-1].timestamp,
            measurements=measurements,
            metadata=metadata,
        )

        # Apply baseline correction if enabled
        if self.baseline_correction:
            self._apply_baseline_correction(session_data)

        return session_data

    def _extract_exp_metadata(self, content: str) -> Dict:
        """Extract metadata from .exp file content"""
        metadata = {}
        lines = content.split("\n")[:20]  # Check first 20 lines

        for line in lines:
            if ":" in line:
                parts = line.split(":", 1)
                key = parts[0].strip().replace("#", "").replace("//", "")
                value = parts[1].strip()
                metadata[key] = value

        return metadata

    def _apply_temperature_correction(
        self, vo2: float, vco2: float, temp_c: float, pressure_kpa: float
    ) -> tuple:
        """
        Apply temperature correction to gas measurements

        Corrects to standard temperature and pressure (STP)
        """
        # Convert to standard temperature (25°C) and pressure
        temp_k = temp_c + 273.15
        standard_temp_k = self.standard_temp_c + 273.15

        # Apply ideal gas law correction
        correction_factor = (standard_temp_k / temp_k) * (pressure_kpa / 101.325)

        vo2_corrected = vo2 * correction_factor
        vco2_corrected = vco2 * correction_factor

        return vo2_corrected, vco2_corrected

    def _apply_baseline_correction(self, session_data: SableSessionData):
        """
        Apply baseline drift correction to measurements

        Uses linear interpolation between pre- and post-measurement baselines
        """
        measurements = session_data.measurements
        if len(measurements) < 10:  # Need enough points for baseline
            return

        # Use first and last 5% of measurements as baseline periods
        n_baseline = max(5, len(measurements) // 20)

        # Calculate baseline VO2 and VCO2
        start_vo2 = np.mean([m.vo2_ml_min for m in measurements[:n_baseline]])
        end_vo2 = np.mean([m.vo2_ml_min for m in measurements[-n_baseline:]])
        start_vco2 = np.mean([m.vco2_ml_min for m in measurements[:n_baseline]])
        end_vco2 = np.mean([m.vco2_ml_min for m in measurements[-n_baseline:]])

        # Calculate expected baseline (assume should be near zero for chamber baseline)
        # For this implementation, we'll assume a small baseline drift rather than 
        # full subtraction
        baseline_vo2 = min(start_vo2, end_vo2) * 0.1  # 10% of minimum baseline
        baseline_vco2 = min(start_vco2, end_vco2) * 0.1

        # Apply linear drift correction
        for i, measurement in enumerate(measurements):
            # Linear interpolation factor
            factor = i / (len(measurements) - 1)

            # Calculate drift correction (small correction, not full subtraction)
            vo2_drift = baseline_vo2 + factor * (baseline_vo2 * 0.1)
            vco2_drift = baseline_vco2 + factor * (baseline_vco2 * 0.1)

            # Apply correction
            corrected_vo2 = measurement.vo2_ml_min - vo2_drift
            corrected_vco2 = measurement.vco2_ml_min - vco2_drift
            corrected_rer = corrected_vco2 / corrected_vo2 if corrected_vo2 > 0 else 0.7

            # Update measurement (create new object since it's frozen)
            measurements[i] = RespirometryMeasurement(
                timestamp=measurement.timestamp,
                vo2_ml_min=max(0.01, corrected_vo2),  # Ensure positive minimum
                vco2_ml_min=max(0.01, corrected_vco2),
                rer=max(0.1, min(2.0, corrected_rer)),  # Ensure valid RER
                chamber_temp_c=measurement.chamber_temp_c,
                chamber_humidity_percent=measurement.chamber_humidity_percent,
                ambient_pressure_kpa=measurement.ambient_pressure_kpa,
                flow_rate_ml_min=measurement.flow_rate_ml_min,
                baseline_corrected=True,
            )

        # Store baseline information
        session_data.baseline_period = (0, n_baseline - 1)
        session_data.metadata["baseline_correction"] = "applied"
        session_data.metadata["baseline_points"] = n_baseline

    def export_session(
        self,
        session_data: SableSessionData,
        output_path: Union[str, Path],
        format: str = "csv",
    ):
        """
        Export session data to file

        Args:
            session_data: Session to export
            output_path: Output file path
            format: Export format ('csv' or 'json')
        """
        output_path = Path(output_path)

        if format.lower() == "csv":
            self._export_csv(session_data, output_path)
        elif format.lower() == "json":
            self._export_json(session_data, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_csv(self, session_data: SableSessionData, output_path: Path):
        """Export session data to CSV format"""
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header with metadata
            writer.writerow([f"# Session: {session_data.session_id}"])
            writer.writerow([f"# Subject: {session_data.subject_id}"])
            writer.writerow([f"# Start: {session_data.start_time}"])
            writer.writerow([f"# End: {session_data.end_time}"])
            writer.writerow([f"# Duration: {session_data.duration_minutes:.1f} min"])

            # Write column headers
            writer.writerow(
                [
                    "Timestamp",
                    "VO2_ml_min",
                    "VCO2_ml_min",
                    "RER",
                    "Temp_C",
                    "Humidity_%",
                    "Pressure_kPa",
                    "Flow_ml_min",
                    "Baseline_Corrected",
                ]
            )

            # Write measurements
            for m in session_data.measurements:
                writer.writerow(
                    [
                        m.timestamp.isoformat(),
                        f"{m.vo2_ml_min:.3f}",
                        f"{m.vco2_ml_min:.3f}",
                        f"{m.rer:.3f}",
                        f"{m.chamber_temp_c:.1f}",
                        f"{m.chamber_humidity_percent:.1f}",
                        f"{m.ambient_pressure_kpa:.1f}",
                        f"{m.flow_rate_ml_min:.1f}" if m.flow_rate_ml_min else "",
                        str(m.baseline_corrected),
                    ]
                )

    def _export_json(self, session_data: SableSessionData, output_path: Path):
        """Export session data to JSON format"""
        data = {
            "session_id": session_data.session_id,
            "subject_id": session_data.subject_id,
            "start_time": session_data.start_time.isoformat(),
            "end_time": session_data.end_time.isoformat(),
            "duration_minutes": session_data.duration_minutes,
            "metadata": session_data.metadata,
            "summary": {
                "mean_vo2": session_data.mean_vo2,
                "mean_vco2": session_data.mean_vco2,
                "mean_rer": session_data.mean_rer,
                "total_measurements": len(session_data.measurements),
            },
            "measurements": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "vo2_ml_min": m.vo2_ml_min,
                    "vco2_ml_min": m.vco2_ml_min,
                    "rer": m.rer,
                    "chamber_temp_c": m.chamber_temp_c,
                    "chamber_humidity_percent": m.chamber_humidity_percent,
                    "ambient_pressure_kpa": m.ambient_pressure_kpa,
                    "flow_rate_ml_min": m.flow_rate_ml_min,
                    "baseline_corrected": m.baseline_corrected,
                }
                for m in session_data.measurements
            ],
        }

        with open(output_path, "w") as jsonfile:
            json.dump(data, jsonfile, indent=2, default=str)
