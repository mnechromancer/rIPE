"""
DATA-003: Field Data Connectors - Environmental Data

This module implements functionality for importing environmental data
from weather stations, GPS coordinates, and time series alignment.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz


@dataclass
class GPSCoordinate:
    """Represents a GPS coordinate with elevation"""
    latitude: float
    longitude: float
    elevation_m: Optional[float] = None
    accuracy_m: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate GPS coordinates"""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}")
    
    def distance_to(self, other: 'GPSCoordinate') -> float:
        """
        Calculate distance to another coordinate using Haversine formula
        Returns distance in meters
        """
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Earth radius in meters
        
        lat1, lon1 = radians(self.latitude), radians(self.longitude)
        lat2, lon2 = radians(other.latitude), radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c


@dataclass
class EnvironmentalReading:
    """Represents an environmental measurement"""
    timestamp: datetime
    location: GPSCoordinate
    temperature_c: Optional[float] = None
    humidity_percent: Optional[float] = None
    pressure_hpa: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    precipitation_mm: Optional[float] = None
    solar_radiation_wm2: Optional[float] = None
    uv_index: Optional[float] = None
    soil_temperature_c: Optional[float] = None
    soil_moisture_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """Check if reading has essential measurements"""
        return (self.temperature_c is not None and 
                self.humidity_percent is not None and 
                self.pressure_hpa is not None)


@dataclass
class WeatherStation:
    """Represents a weather station"""
    station_id: str
    name: str
    location: GPSCoordinate
    station_type: str  # "automatic", "manual", "research"
    installation_date: Optional[datetime] = None
    active: bool = True
    instruments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentalDataset:
    """Contains environmental data from multiple stations/locations"""
    dataset_id: str
    stations: Dict[str, WeatherStation]
    readings: List[EnvironmentalReading]
    time_range: Tuple[datetime, datetime]
    timezone: str = 'UTC'
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Create indices for fast lookup"""
        self._station_readings = {}
        for reading in self.readings:
            station_id = reading.metadata.get('station_id', 'unknown')
            if station_id not in self._station_readings:
                self._station_readings[station_id] = []
            self._station_readings[station_id].append(reading)
    
    def get_readings_by_station(self, station_id: str) -> List[EnvironmentalReading]:
        """Get all readings from a specific station"""
        return self._station_readings.get(station_id, [])
    
    def get_readings_by_timerange(self, start: datetime, end: datetime) -> List[EnvironmentalReading]:
        """Get readings within a time range"""
        return [r for r in self.readings if start <= r.timestamp <= end]
    
    def get_readings_by_location(self, center: GPSCoordinate, 
                                radius_m: float) -> List[EnvironmentalReading]:
        """Get readings within radius of a location"""
        return [r for r in self.readings 
                if r.location.distance_to(center) <= radius_m]


class EnvironmentalDataImporter:
    """
    Importer for environmental data from various sources
    
    Supports:
    - Weather station data (CSV, JSON formats)
    - GPS coordinate handling with time alignment
    - Multiple data formats and time zone conversion
    - Time series interpolation and gap filling
    """
    
    def __init__(self, default_timezone: str = 'UTC'):
        """
        Initialize environmental data importer
        
        Args:
            default_timezone: Default timezone for data without timezone info
        """
        self.default_timezone = pytz.timezone(default_timezone)
        self.station_registry = {}
    
    def register_station(self, station: WeatherStation):
        """Register a weather station"""
        self.station_registry[station.station_id] = station
    
    def import_weather_station_csv(self, filepath: Union[str, Path],
                                  station_id: str,
                                  dataset_id: str) -> EnvironmentalDataset:
        """
        Import weather station data from CSV
        
        Expected columns:
        - timestamp (or date/time columns)
        - latitude, longitude (optional, can use station location)
        - temperature, humidity, pressure, etc.
        
        Args:
            filepath: Path to CSV file
            station_id: Weather station identifier
            dataset_id: Dataset identifier
            
        Returns:
            EnvironmentalDataset object
        """
        filepath = Path(filepath)
        df = pd.read_csv(filepath)
        
        # Parse timestamps
        df = self._parse_timestamps(df)
        
        # Get station info
        station = self.station_registry.get(station_id)
        if not station:
            # Create default station from first data point
            if 'latitude' in df.columns and 'longitude' in df.columns:
                first_lat = df['latitude'].iloc[0]
                first_lon = df['longitude'].iloc[0]
                first_elev = df.get('elevation', pd.Series([None])).iloc[0]
            else:
                first_lat, first_lon, first_elev = 0.0, 0.0, None
            
            location = GPSCoordinate(first_lat, first_lon, first_elev)
            station = WeatherStation(
                station_id=station_id,
                name=f"Station {station_id}",
                location=location,
                station_type="unknown"
            )
        
        # Convert DataFrame to readings
        readings = []
        for _, row in df.iterrows():
            # Get location (use station location if not in data)
            if 'latitude' in df.columns and 'longitude' in df.columns:
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                elev = row.get('elevation')
                location = GPSCoordinate(lat, lon, elev)
            else:
                location = station.location
            
            # Create reading
            reading = EnvironmentalReading(
                timestamp=row['timestamp'],
                location=location,
                temperature_c=self._safe_float(row.get('temperature')),
                humidity_percent=self._safe_float(row.get('humidity')),
                pressure_hpa=self._safe_float(row.get('pressure')),
                wind_speed_ms=self._safe_float(row.get('wind_speed')),
                wind_direction_deg=self._safe_float(row.get('wind_direction')),
                precipitation_mm=self._safe_float(row.get('precipitation')),
                solar_radiation_wm2=self._safe_float(row.get('solar_radiation')),
                uv_index=self._safe_float(row.get('uv_index')),
                soil_temperature_c=self._safe_float(row.get('soil_temperature')),
                soil_moisture_percent=self._safe_float(row.get('soil_moisture')),
                metadata={'station_id': station_id, 'source_file': str(filepath)}
            )
            readings.append(reading)
        
        # Calculate time range
        timestamps = [r.timestamp for r in readings]
        time_range = (min(timestamps), max(timestamps))
        
        return EnvironmentalDataset(
            dataset_id=dataset_id,
            stations={station_id: station},
            readings=readings,
            time_range=time_range,
            metadata={
                'source_file': str(filepath),
                'import_date': datetime.now().isoformat(),
                'total_readings': len(readings),
                'station_count': 1
            }
        )
    
    def import_multiple_stations(self, data_dir: Union[str, Path],
                                dataset_id: str,
                                file_pattern: str = "*.csv") -> EnvironmentalDataset:
        """
        Import data from multiple weather stations
        
        Args:
            data_dir: Directory containing station data files
            dataset_id: Dataset identifier
            file_pattern: File pattern to match (e.g., "*.csv")
            
        Returns:
            Combined EnvironmentalDataset
        """
        data_dir = Path(data_dir)
        all_stations = {}
        all_readings = []
        
        for filepath in data_dir.glob(file_pattern):
            # Use filename as station ID
            station_id = filepath.stem
            
            try:
                dataset = self.import_weather_station_csv(filepath, station_id, f"{dataset_id}_{station_id}")
                all_stations.update(dataset.stations)
                all_readings.extend(dataset.readings)
            except Exception as e:
                print(f"Warning: Failed to import {filepath}: {e}")
        
        if not all_readings:
            raise ValueError("No valid data files found")
        
        # Calculate overall time range
        timestamps = [r.timestamp for r in all_readings]
        time_range = (min(timestamps), max(timestamps))
        
        return EnvironmentalDataset(
            dataset_id=dataset_id,
            stations=all_stations,
            readings=all_readings,
            time_range=time_range,
            metadata={
                'source_directory': str(data_dir),
                'import_date': datetime.now().isoformat(),
                'total_readings': len(all_readings),
                'station_count': len(all_stations)
            }
        )
    
    def import_gps_track(self, filepath: Union[str, Path],
                        dataset_id: str) -> List[GPSCoordinate]:
        """
        Import GPS track data (GPX or CSV format)
        
        Args:
            filepath: Path to GPS track file
            dataset_id: Dataset identifier
            
        Returns:
            List of GPSCoordinate objects
        """
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.gpx':
            return self._parse_gpx_file(filepath)
        else:
            # Assume CSV
            df = pd.read_csv(filepath)
            df = self._parse_timestamps(df)
            
            coordinates = []
            for _, row in df.iterrows():
                coord = GPSCoordinate(
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude']),
                    elevation_m=self._safe_float(row.get('elevation')),
                    accuracy_m=self._safe_float(row.get('accuracy')),
                    timestamp=row.get('timestamp')
                )
                coordinates.append(coord)
            
            return coordinates
    
    def align_with_gps_track(self, dataset: EnvironmentalDataset,
                            gps_track: List[GPSCoordinate],
                            max_distance_m: float = 1000,
                            max_time_diff_minutes: float = 30) -> EnvironmentalDataset:
        """
        Align environmental readings with GPS track points
        
        Args:
            dataset: Environmental dataset
            gps_track: List of GPS coordinates with timestamps
            max_distance_m: Maximum distance for alignment (meters)
            max_time_diff_minutes: Maximum time difference for alignment (minutes)
            
        Returns:
            New dataset with aligned readings
        """
        aligned_readings = []
        
        for reading in dataset.readings:
            best_gps = None
            best_distance = float('inf')
            best_time_diff = float('inf')
            
            # Find closest GPS point in time and space
            for gps_point in gps_track:
                if gps_point.timestamp is None:
                    continue
                
                # Calculate time difference
                time_diff = abs((reading.timestamp - gps_point.timestamp).total_seconds() / 60)
                if time_diff > max_time_diff_minutes:
                    continue
                
                # Calculate spatial distance
                distance = reading.location.distance_to(gps_point)
                if distance > max_distance_m:
                    continue
                
                # Check if this is the best match
                combined_score = distance + time_diff * 60  # Weight time difference
                if combined_score < (best_distance + best_time_diff * 60):
                    best_gps = gps_point
                    best_distance = distance
                    best_time_diff = time_diff
            
            if best_gps:
                # Create aligned reading with GPS location
                aligned_reading = EnvironmentalReading(
                    timestamp=reading.timestamp,
                    location=best_gps,
                    temperature_c=reading.temperature_c,
                    humidity_percent=reading.humidity_percent,
                    pressure_hpa=reading.pressure_hpa,
                    wind_speed_ms=reading.wind_speed_ms,
                    wind_direction_deg=reading.wind_direction_deg,
                    precipitation_mm=reading.precipitation_mm,
                    solar_radiation_wm2=reading.solar_radiation_wm2,
                    uv_index=reading.uv_index,
                    soil_temperature_c=reading.soil_temperature_c,
                    soil_moisture_percent=reading.soil_moisture_percent,
                    metadata={
                        **reading.metadata,
                        'gps_aligned': True,
                        'alignment_distance_m': best_distance,
                        'alignment_time_diff_min': best_time_diff
                    }
                )
                aligned_readings.append(aligned_reading)
        
        # Create new dataset with aligned readings
        return EnvironmentalDataset(
            dataset_id=f"{dataset.dataset_id}_aligned",
            stations=dataset.stations,
            readings=aligned_readings,
            time_range=dataset.time_range,
            timezone=dataset.timezone,
            metadata={
                **dataset.metadata,
                'gps_aligned': True,
                'alignment_parameters': {
                    'max_distance_m': max_distance_m,
                    'max_time_diff_minutes': max_time_diff_minutes
                },
                'original_readings': len(dataset.readings),
                'aligned_readings': len(aligned_readings)
            }
        )
    
    def interpolate_missing_data(self, dataset: EnvironmentalDataset,
                                method: str = 'linear',
                                max_gap_hours: float = 6) -> EnvironmentalDataset:
        """
        Interpolate missing environmental data
        
        Args:
            dataset: Environmental dataset
            method: Interpolation method ('linear', 'nearest', 'cubic')
            max_gap_hours: Maximum gap size to interpolate (hours)
            
        Returns:
            Dataset with interpolated values
        """
        # Group readings by station
        station_readings = {}
        for reading in dataset.readings:
            station_id = reading.metadata.get('station_id', 'unknown')
            if station_id not in station_readings:
                station_readings[station_id] = []
            station_readings[station_id].append(reading)
        
        all_interpolated = []
        
        for station_id, readings in station_readings.items():
            # Sort by timestamp
            readings.sort(key=lambda r: r.timestamp)
            
            # Create DataFrame for interpolation
            df_data = []
            for reading in readings:
                df_data.append({
                    'timestamp': reading.timestamp,
                    'temperature_c': reading.temperature_c,
                    'humidity_percent': reading.humidity_percent,
                    'pressure_hpa': reading.pressure_hpa,
                    'wind_speed_ms': reading.wind_speed_ms,
                    'precipitation_mm': reading.precipitation_mm,
                    'solar_radiation_wm2': reading.solar_radiation_wm2
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            
            # Interpolate only small gaps
            df_interp = df.copy()
            for col in df.columns:
                # Find gaps
                mask = df[col].notna()
                if mask.sum() < 2:  # Need at least 2 points for interpolation
                    continue
                
                # Interpolate
                if method == 'linear':
                    df_interp[col] = df[col].interpolate(method='time', limit=int(max_gap_hours * 60))
                elif method == 'nearest':
                    df_interp[col] = df[col].fillna(method='nearest', limit=int(max_gap_hours * 60))
                else:
                    df_interp[col] = df[col].interpolate(method=method, limit=int(max_gap_hours * 60))
            
            # Convert back to readings
            for i, (timestamp, row) in enumerate(df_interp.iterrows()):
                original_reading = readings[i]
                interpolated_reading = EnvironmentalReading(
                    timestamp=timestamp,
                    location=original_reading.location,
                    temperature_c=self._safe_float(row['temperature_c']),
                    humidity_percent=self._safe_float(row['humidity_percent']),
                    pressure_hpa=self._safe_float(row['pressure_hpa']),
                    wind_speed_ms=self._safe_float(row['wind_speed_ms']),
                    precipitation_mm=self._safe_float(row['precipitation_mm']),
                    solar_radiation_wm2=self._safe_float(row['solar_radiation_wm2']),
                    metadata={
                        **original_reading.metadata,
                        'interpolated': True
                    }
                )
                all_interpolated.append(interpolated_reading)
        
        return EnvironmentalDataset(
            dataset_id=f"{dataset.dataset_id}_interpolated",
            stations=dataset.stations,
            readings=all_interpolated,
            time_range=dataset.time_range,
            timezone=dataset.timezone,
            metadata={
                **dataset.metadata,
                'interpolated': True,
                'interpolation_method': method,
                'max_gap_hours': max_gap_hours
            }
        )
    
    def export_dataset(self, dataset: EnvironmentalDataset,
                      output_path: Union[str, Path],
                      format: str = 'csv',
                      include_metadata: bool = True):
        """
        Export environmental dataset to file
        
        Args:
            dataset: Environmental dataset to export
            output_path: Output file path
            format: Export format ('csv', 'json')
            include_metadata: Whether to include metadata
        """
        output_path = Path(output_path)
        
        if format.lower() == 'csv':
            self._export_csv(dataset, output_path, include_metadata)
        elif format.lower() == 'json':
            self._export_json(dataset, output_path, include_metadata)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp columns in DataFrame"""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns and 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        elif 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
        else:
            # Try to find timestamp-like columns
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    try:
                        df['timestamp'] = pd.to_datetime(df[col])
                        break
                    except:
                        continue
            
            if 'timestamp' not in df.columns:
                raise ValueError("No timestamp column found")
        
        return df
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if pd.isna(value) or value is None or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _parse_gpx_file(self, filepath: Path) -> List[GPSCoordinate]:
        """Parse GPX file (simplified implementation)"""
        # In a real implementation, you'd use a GPX parsing library
        # This is a placeholder that would read GPX XML format
        coordinates = []
        
        # For now, assume GPX data is converted to CSV format
        # In practice, use libraries like gpxpy
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Find trackpoints (simplified)
            for trkpt in root.findall('.//{http://www.topografix.com/GPX/1/1}trkpt'):
                lat = float(trkpt.get('lat'))
                lon = float(trkpt.get('lon'))
                
                # Get elevation
                ele_elem = trkpt.find('{http://www.topografix.com/GPX/1/1}ele')
                elevation = float(ele_elem.text) if ele_elem is not None else None
                
                # Get time
                time_elem = trkpt.find('{http://www.topografix.com/GPX/1/1}time')
                timestamp = None
                if time_elem is not None:
                    timestamp = datetime.fromisoformat(time_elem.text.replace('Z', '+00:00'))
                
                coord = GPSCoordinate(lat, lon, elevation, timestamp=timestamp)
                coordinates.append(coord)
        
        except Exception as e:
            print(f"Warning: Failed to parse GPX file {filepath}: {e}")
        
        return coordinates
    
    def _export_csv(self, dataset: EnvironmentalDataset, output_path: Path, include_metadata: bool):
        """Export dataset to CSV"""
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = [
                'timestamp', 'station_id', 'latitude', 'longitude', 'elevation_m',
                'temperature_c', 'humidity_percent', 'pressure_hpa',
                'wind_speed_ms', 'wind_direction_deg', 'precipitation_mm',
                'solar_radiation_wm2', 'uv_index', 'soil_temperature_c', 'soil_moisture_percent'
            ]
            
            if include_metadata:
                writer.writerow([f"# Dataset: {dataset.dataset_id}"])
                writer.writerow([f"# Time range: {dataset.time_range[0]} to {dataset.time_range[1]}"])
                writer.writerow([f"# Stations: {len(dataset.stations)}"])
                writer.writerow([f"# Total readings: {len(dataset.readings)}"])
            
            writer.writerow(header)
            
            # Write data
            for reading in dataset.readings:
                row = [
                    reading.timestamp.isoformat(),
                    reading.metadata.get('station_id', ''),
                    reading.location.latitude,
                    reading.location.longitude,
                    reading.location.elevation_m or '',
                    reading.temperature_c or '',
                    reading.humidity_percent or '',
                    reading.pressure_hpa or '',
                    reading.wind_speed_ms or '',
                    reading.wind_direction_deg or '',
                    reading.precipitation_mm or '',
                    reading.solar_radiation_wm2 or '',
                    reading.uv_index or '',
                    reading.soil_temperature_c or '',
                    reading.soil_moisture_percent or ''
                ]
                writer.writerow(row)
    
    def _export_json(self, dataset: EnvironmentalDataset, output_path: Path, include_metadata: bool):
        """Export dataset to JSON"""
        data = {
            'dataset_id': dataset.dataset_id,
            'time_range': {
                'start': dataset.time_range[0].isoformat(),
                'end': dataset.time_range[1].isoformat()
            },
            'timezone': dataset.timezone,
            'stations': {},
            'readings': []
        }
        
        if include_metadata:
            data['metadata'] = dataset.metadata
            data['created_date'] = dataset.created_date.isoformat()
        
        # Add stations
        for station_id, station in dataset.stations.items():
            data['stations'][station_id] = {
                'name': station.name,
                'location': {
                    'latitude': station.location.latitude,
                    'longitude': station.location.longitude,
                    'elevation_m': station.location.elevation_m
                },
                'type': station.station_type,
                'active': station.active,
                'instruments': station.instruments
            }
        
        # Add readings
        for reading in dataset.readings:
            reading_data = {
                'timestamp': reading.timestamp.isoformat(),
                'station_id': reading.metadata.get('station_id'),
                'location': {
                    'latitude': reading.location.latitude,
                    'longitude': reading.location.longitude,
                    'elevation_m': reading.location.elevation_m
                },
                'measurements': {}
            }
            
            # Add non-null measurements
            measurements = [
                ('temperature_c', reading.temperature_c),
                ('humidity_percent', reading.humidity_percent),
                ('pressure_hpa', reading.pressure_hpa),
                ('wind_speed_ms', reading.wind_speed_ms),
                ('wind_direction_deg', reading.wind_direction_deg),
                ('precipitation_mm', reading.precipitation_mm),
                ('solar_radiation_wm2', reading.solar_radiation_wm2),
                ('uv_index', reading.uv_index),
                ('soil_temperature_c', reading.soil_temperature_c),
                ('soil_moisture_percent', reading.soil_moisture_percent)
            ]
            
            for key, value in measurements:
                if value is not None:
                    reading_data['measurements'][key] = value
            
            data['readings'].append(reading_data)
        
        with open(output_path, 'w') as jsonfile:
            json.dump(data, jsonfile, indent=2, default=str)