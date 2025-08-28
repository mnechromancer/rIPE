"""
Tests for DATA-003: Field Data Connectors

Tests the environmental and morphology data import functionality including
weather station data, GPS coordinate handling, and morphometric data processing.
"""

import pytest
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

from ipe.lab_integration.field.environmental import (
    EnvironmentalDataImporter,
    EnvironmentalDataset,
    EnvironmentalReading,
    GPSCoordinate,
    WeatherStation
)
from ipe.lab_integration.field.morphology import (
    MorphologyDataImporter,
    MorphometricDataset,
    Specimen,
    MorphologicalMeasurement
)


class TestGPSCoordinate:
    """Test suite for GPSCoordinate"""
    
    def test_coordinate_creation(self):
        """Test basic GPS coordinate creation"""
        coord = GPSCoordinate(
            latitude=40.7128,
            longitude=-74.0060,
            elevation_m=10.0,
            accuracy_m=5.0,
            timestamp=datetime.now()
        )
        
        assert coord.latitude == 40.7128
        assert coord.longitude == -74.0060
        assert coord.elevation_m == 10.0
        assert coord.accuracy_m == 5.0
    
    def test_coordinate_validation(self):
        """Test GPS coordinate validation"""
        # Test invalid latitude
        with pytest.raises(ValueError, match="Invalid latitude"):
            GPSCoordinate(latitude=91.0, longitude=0.0)
        
        with pytest.raises(ValueError, match="Invalid latitude"):
            GPSCoordinate(latitude=-91.0, longitude=0.0)
        
        # Test invalid longitude
        with pytest.raises(ValueError, match="Invalid longitude"):
            GPSCoordinate(latitude=0.0, longitude=181.0)
        
        with pytest.raises(ValueError, match="Invalid longitude"):
            GPSCoordinate(latitude=0.0, longitude=-181.0)
    
    def test_distance_calculation(self):
        """Test distance calculation between coordinates"""
        # New York City
        coord1 = GPSCoordinate(40.7128, -74.0060)
        # Los Angeles
        coord2 = GPSCoordinate(34.0522, -118.2437)
        
        distance = coord1.distance_to(coord2)
        
        # Distance should be approximately 3944 km
        assert 3900000 < distance < 4000000  # meters
    
    def test_same_location_distance(self):
        """Test distance calculation for same location"""
        coord1 = GPSCoordinate(40.7128, -74.0060)
        coord2 = GPSCoordinate(40.7128, -74.0060)
        
        distance = coord1.distance_to(coord2)
        assert distance == pytest.approx(0, abs=1)  # Should be very close to 0


class TestEnvironmentalReading:
    """Test suite for EnvironmentalReading"""
    
    def create_test_reading(self):
        """Helper to create test environmental reading"""
        location = GPSCoordinate(40.0, -74.0, 100.0)
        return EnvironmentalReading(
            timestamp=datetime.now(),
            location=location,
            temperature_c=25.0,
            humidity_percent=60.0,
            pressure_hpa=1013.25,
            wind_speed_ms=5.0
        )
    
    def test_reading_creation(self):
        """Test environmental reading creation"""
        reading = self.create_test_reading()
        
        assert reading.temperature_c == 25.0
        assert reading.humidity_percent == 60.0
        assert reading.pressure_hpa == 1013.25
        assert reading.wind_speed_ms == 5.0
    
    def test_completeness_check(self):
        """Test reading completeness check"""
        # Complete reading
        complete_reading = self.create_test_reading()
        assert complete_reading.is_complete()
        
        # Incomplete reading
        location = GPSCoordinate(40.0, -74.0)
        incomplete_reading = EnvironmentalReading(
            timestamp=datetime.now(),
            location=location,
            temperature_c=25.0,
            # Missing humidity and pressure
        )
        assert not incomplete_reading.is_complete()


class TestEnvironmentalDataImporter:
    """Test suite for EnvironmentalDataImporter"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.importer = EnvironmentalDataImporter()
        self.temp_dir = tempfile.mkdtemp()
    
    def create_weather_station_csv(self, filename="weather_data.csv", n_points=24):
        """Create test weather station CSV file"""
        filepath = Path(self.temp_dir) / filename
        
        # Generate 24 hours of hourly data
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        data = []
        for i in range(n_points):
            timestamp = start_time + timedelta(hours=i)
            data.append({
                'timestamp': timestamp.isoformat(),
                'latitude': 40.7128 + 0.001 * np.random.randn(),
                'longitude': -74.0060 + 0.001 * np.random.randn(),
                'elevation': 10 + np.random.randn(),
                'temperature': 20 + 5 * np.sin(i * 2 * np.pi / 24) + np.random.randn(),
                'humidity': 60 + 10 * np.random.randn(),
                'pressure': 1013 + 5 * np.random.randn(),
                'wind_speed': 5 + 3 * np.random.randn(),
                'wind_direction': np.random.uniform(0, 360),
                'precipitation': np.random.exponential(0.1) if np.random.rand() < 0.2 else 0
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def test_single_station_import(self):
        """Test importing data from a single weather station"""
        filepath = self.create_weather_station_csv(n_points=12)
        dataset = self.importer.import_weather_station_csv(filepath, "TEST_STATION", "test_dataset")
        
        assert dataset.dataset_id == "test_dataset"
        assert len(dataset.readings) == 12
        assert "TEST_STATION" in dataset.stations
        assert dataset.stations["TEST_STATION"].station_id == "TEST_STATION"
        
        # Check data integrity
        for reading in dataset.readings:
            assert reading.temperature_c is not None
            assert reading.humidity_percent is not None
            assert reading.pressure_hpa is not None
            assert 39 < reading.location.latitude < 42  # Rough range check
            assert -75 < reading.location.longitude < -73
    
    def test_multiple_stations_import(self):
        """Test importing data from multiple weather stations"""
        # Create multiple station files
        self.create_weather_station_csv("station_A.csv", 10)
        self.create_weather_station_csv("station_B.csv", 8)
        self.create_weather_station_csv("station_C.csv", 12)
        
        dataset = self.importer.import_multiple_stations(self.temp_dir, "multi_station_test")
        
        assert dataset.dataset_id == "multi_station_test"
        assert len(dataset.stations) == 3
        assert len(dataset.readings) == 30  # 10 + 8 + 12
        
        # Check that all stations are present
        station_ids = list(dataset.stations.keys())
        assert "station_A" in station_ids
        assert "station_B" in station_ids
        assert "station_C" in station_ids
    
    def test_gps_track_import(self):
        """Test GPS track import from CSV"""
        # Create GPS track CSV
        gps_file = Path(self.temp_dir) / "gps_track.csv"
        
        track_data = []
        base_time = datetime.now()
        for i in range(10):
            track_data.append({
                'timestamp': (base_time + timedelta(minutes=i*10)).isoformat(),
                'latitude': 40.7128 + i * 0.001,
                'longitude': -74.0060 + i * 0.001,
                'elevation': 10 + i * 2,
                'accuracy': 5.0
            })
        
        df = pd.DataFrame(track_data)
        df.to_csv(gps_file, index=False)
        
        # Import track
        track = self.importer.import_gps_track(gps_file, "test_track")
        
        assert len(track) == 10
        assert all(isinstance(coord, GPSCoordinate) for coord in track)
        
        # Check progression
        assert track[0].latitude < track[-1].latitude
        assert track[0].elevation_m < track[-1].elevation_m
    
    def test_gps_alignment(self):
        """Test aligning environmental data with GPS track"""
        # Create environmental dataset
        weather_file = self.create_weather_station_csv("weather.csv", 6)
        dataset = self.importer.import_weather_station_csv(weather_file, "WEATHER_STATION", "alignment_test")
        
        # Create GPS track that roughly matches timing
        gps_track = []
        for i, reading in enumerate(dataset.readings):
            # GPS points slightly offset in time and space
            gps_time = reading.timestamp + timedelta(minutes=np.random.randint(-15, 16))
            gps_coord = GPSCoordinate(
                latitude=reading.location.latitude + 0.001 * np.random.randn(),
                longitude=reading.location.longitude + 0.001 * np.random.randn(),
                timestamp=gps_time
            )
            gps_track.append(gps_coord)
        
        # Align data
        aligned_dataset = self.importer.align_with_gps_track(
            dataset, gps_track, max_distance_m=500, max_time_diff_minutes=20)
        
        assert len(aligned_dataset.readings) <= len(dataset.readings)  # Some may not align
        assert aligned_dataset.metadata['gps_aligned'] == True
        
        # Check that aligned readings have GPS metadata
        for reading in aligned_dataset.readings:
            assert reading.metadata.get('gps_aligned') == True
            assert 'alignment_distance_m' in reading.metadata
            assert 'alignment_time_diff_min' in reading.metadata
    
    def test_data_interpolation(self):
        """Test missing data interpolation"""
        # Create dataset with some missing values
        weather_file = self.create_weather_station_csv("weather_gaps.csv", 10)
        dataset = self.importer.import_weather_station_csv(weather_file, "GAP_STATION", "gap_test")
        
        # Artificially introduce gaps by setting some values to None
        for i in [2, 4, 6]:
            if i < len(dataset.readings):
                dataset.readings[i].temperature_c = None
                dataset.readings[i].humidity_percent = None
        
        # Interpolate
        interpolated = self.importer.interpolate_missing_data(dataset, method='linear')
        
        assert interpolated.metadata['interpolated'] == True
        assert interpolated.metadata['interpolation_method'] == 'linear'
        
        # Check that gaps were filled (approximately)
        interpolated_temps = [r.temperature_c for r in interpolated.readings if r.temperature_c is not None]
        assert len(interpolated_temps) > len([r.temperature_c for r in dataset.readings if r.temperature_c is not None])
    
    def test_dataset_queries(self):
        """Test dataset query methods"""
        weather_file = self.create_weather_station_csv("query_test.csv", 24)
        dataset = self.importer.import_weather_station_csv(weather_file, "QUERY_STATION", "query_dataset")
        
        # Test station-based query
        station_readings = dataset.get_readings_by_station("QUERY_STATION")
        assert len(station_readings) == 24
        
        # Test time-based query
        start_time = dataset.time_range[0] + timedelta(hours=6)
        end_time = dataset.time_range[0] + timedelta(hours=12)
        time_readings = dataset.get_readings_by_timerange(start_time, end_time)
        assert len(time_readings) <= 7  # Should be 6-7 readings
        
        # Test location-based query
        center = GPSCoordinate(40.7128, -74.0060)
        nearby_readings = dataset.get_readings_by_location(center, radius_m=1000)
        assert len(nearby_readings) > 0  # Should find some readings
    
    def test_export_csv(self):
        """Test CSV export"""
        weather_file = self.create_weather_station_csv("export_test.csv", 5)
        dataset = self.importer.import_weather_station_csv(weather_file, "EXPORT_STATION", "export_dataset")
        
        output_file = Path(self.temp_dir) / "exported_weather.csv"
        self.importer.export_dataset(dataset, output_file, format='csv')
        
        assert output_file.exists()
        
        # Verify export content
        exported_df = pd.read_csv(output_file, comment='#')
        assert len(exported_df) == 5
        assert 'timestamp' in exported_df.columns
        assert 'temperature_c' in exported_df.columns
    
    def test_export_json(self):
        """Test JSON export"""
        weather_file = self.create_weather_station_csv("json_test.csv", 3)
        dataset = self.importer.import_weather_station_csv(weather_file, "JSON_STATION", "json_dataset")
        
        output_file = Path(self.temp_dir) / "exported_weather.json"
        self.importer.export_dataset(dataset, output_file, format='json')
        
        assert output_file.exists()
        
        # Verify export content
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert data['dataset_id'] == "json_dataset"
        assert len(data['readings']) == 3
        assert 'stations' in data
        assert 'JSON_STATION' in data['stations']


class TestMorphologicalMeasurement:
    """Test suite for MorphologicalMeasurement"""
    
    def test_measurement_creation(self):
        """Test basic measurement creation"""
        measurement = MorphologicalMeasurement(
            measurement_id="M001",
            specimen_id="SPEC001",
            trait_name="body_mass",
            value=25.5,
            unit="g",
            measurement_method="digital_scale",
            observer="researcher1",
            date_measured=datetime.now()
        )
        
        assert measurement.measurement_id == "M001"
        assert measurement.specimen_id == "SPEC001"
        assert measurement.trait_name == "body_mass"
        assert measurement.value == 25.5
        assert measurement.unit == "g"
    
    def test_measurement_validation(self):
        """Test measurement validation"""
        # Test negative value
        with pytest.raises(ValueError, match="Measurement value cannot be negative"):
            MorphologicalMeasurement(
                measurement_id="M001",
                specimen_id="SPEC001",
                trait_name="body_mass",
                value=-5.0,
                unit="g",
                measurement_method="digital_scale"
            )
        
        # Test empty trait name
        with pytest.raises(ValueError, match="Trait name is required"):
            MorphologicalMeasurement(
                measurement_id="M001",
                specimen_id="SPEC001",
                trait_name="",
                value=25.5,
                unit="g",
                measurement_method="digital_scale"
            )


class TestSpecimen:
    """Test suite for Specimen"""
    
    def create_test_specimen(self):
        """Helper to create test specimen"""
        return Specimen(
            specimen_id="SPEC001",
            species="Peromyscus maniculatus",
            collection_date=datetime.now(),
            collection_location="Colorado, USA",
            latitude=39.7392,
            longitude=-104.9903,
            elevation_m=1650,
            sex="F",
            age_class="adult",
            collector="Field Researcher"
        )
    
    def test_specimen_creation(self):
        """Test specimen creation"""
        specimen = self.create_test_specimen()
        
        assert specimen.specimen_id == "SPEC001"
        assert specimen.species == "Peromyscus maniculatus"
        assert specimen.sex == "F"
        assert specimen.age_class == "adult"
    
    def test_add_measurement(self):
        """Test adding measurements to specimen"""
        specimen = self.create_test_specimen()
        
        measurement = MorphologicalMeasurement(
            measurement_id="M001",
            specimen_id="",  # Will be set by add_measurement
            trait_name="body_mass",
            value=25.5,
            unit="g",
            measurement_method="digital_scale"
        )
        
        specimen.add_measurement(measurement)
        
        assert len(specimen.measurements) == 1
        assert measurement.specimen_id == "SPEC001"
    
    def test_get_measurement(self):
        """Test retrieving measurements"""
        specimen = self.create_test_specimen()
        
        mass_measurement = MorphologicalMeasurement(
            measurement_id="M001", specimen_id="SPEC001",
            trait_name="body_mass", value=25.5, unit="g", measurement_method="scale"
        )
        length_measurement = MorphologicalMeasurement(
            measurement_id="M002", specimen_id="SPEC001",
            trait_name="total_length", value=180.0, unit="mm", measurement_method="ruler"
        )
        
        specimen.add_measurement(mass_measurement)
        specimen.add_measurement(length_measurement)
        
        # Test get_measurement
        retrieved = specimen.get_measurement("body_mass")
        assert retrieved is not None
        assert retrieved.value == 25.5
        
        # Test get_measurement_value
        length_value = specimen.get_measurement_value("total_length")
        assert length_value == 180.0
        
        # Test non-existent trait
        assert specimen.get_measurement("wingspan") is None
        assert specimen.get_measurement_value("wingspan") is None


class TestMorphologyDataImporter:
    """Test suite for MorphologyDataImporter"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.importer = MorphologyDataImporter()
        self.temp_dir = tempfile.mkdtemp()
    
    def create_specimen_csv(self, filename="specimens.csv", n_specimens=10):
        """Create test specimen CSV file"""
        filepath = Path(self.temp_dir) / filename
        
        species_list = ["Peromyscus maniculatus", "Peromyscus leucopus", "Microtus pennsylvanicus"]
        
        data = []
        for i in range(n_specimens):
            specimen_data = {
                'specimen_id': f"SPEC{i:03d}",
                'species': np.random.choice(species_list),
                'collection_date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).date(),
                'collection_location': f"Site {i%3 + 1}",
                'latitude': 40.0 + np.random.uniform(-1, 1),
                'longitude': -74.0 + np.random.uniform(-1, 1),
                'elevation_m': np.random.uniform(0, 1000),
                'sex': np.random.choice(['M', 'F', 'U']),
                'age_class': np.random.choice(['adult', 'juvenile', 'subadult']),
                'collector': f"Researcher {i%2 + 1}",
                'catalog_number': f"CAT{i:05d}",
                'museum_code': 'TEST',
                # Morphometric measurements
                'body_mass': np.random.uniform(15, 35),
                'total_length': np.random.uniform(150, 220),
                'tail_length': np.random.uniform(60, 110),
                'hind_foot_length': np.random.uniform(18, 25),
                'ear_length': np.random.uniform(12, 18)
            }
            data.append(specimen_data)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def test_specimen_csv_import(self):
        """Test importing specimens from CSV"""
        filepath = self.create_specimen_csv(n_specimens=5)
        dataset = self.importer.import_specimen_csv(filepath, "test_morphometry")
        
        assert dataset.dataset_id == "test_morphometry"
        assert len(dataset.specimens) == 5
        assert len(dataset.trait_definitions) == 5  # 5 morphometric traits
        
        # Check that measurements were created
        for specimen in dataset.specimens.values():
            assert len(specimen.measurements) == 5
            assert specimen.get_measurement_value("body_mass") is not None
            assert specimen.get_measurement_value("total_length") is not None
    
    def test_trait_statistics(self):
        """Test trait statistics calculation"""
        filepath = self.create_specimen_csv(n_specimens=20)
        dataset = self.importer.import_specimen_csv(filepath, "stats_test")
        
        # Test overall statistics
        mass_stats = dataset.get_trait_statistics("body_mass")
        assert mass_stats['count'] == 20
        assert 15 <= mass_stats['mean'] <= 35
        assert mass_stats['std'] > 0
        assert mass_stats['min'] <= mass_stats['mean'] <= mass_stats['max']
        
        # Test species-specific statistics
        species = list(dataset.get_available_species())[0]
        species_stats = dataset.get_trait_statistics("body_mass", species=species)
        assert species_stats['count'] <= 20
        assert species_stats['count'] > 0
    
    def test_unit_standardization(self):
        """Test unit standardization"""
        # Create dataset with mixed units
        filepath = Path(self.temp_dir) / "mixed_units.csv"
        
        data = []
        for i in range(5):
            # Some measurements in different units
            mass_unit = 'g' if i < 3 else 'kg'
            mass_value = np.random.uniform(20, 30) if mass_unit == 'g' else np.random.uniform(0.02, 0.03)
            
            specimen_data = {
                'specimen_id': f"SPEC{i:03d}",
                'species': "Test species",
                'body_mass': mass_value,
                'total_length': np.random.uniform(150, 200)
            }
            data.append(specimen_data)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        # Import and standardize
        dataset = self.importer.import_specimen_csv(filepath, "mixed_units_test")
        
        # Manually set some measurements to different units for testing
        for i, specimen in enumerate(dataset.specimens.values()):
            if i >= 3:  # Last 2 specimens
                mass_measurement = specimen.get_measurement("body_mass")
                if mass_measurement:
                    mass_measurement.unit = "kg"
        
        standardized = self.importer.standardize_units(dataset)
        
        # Check that all units are standardized
        for specimen in standardized.specimens.values():
            mass_measurement = specimen.get_measurement("body_mass")
            if mass_measurement:
                assert mass_measurement.unit == "g"  # Standard unit for mass
    
    def test_quality_control(self):
        """Test quality control checks"""
        filepath = self.create_specimen_csv(n_specimens=15)
        dataset = self.importer.import_specimen_csv(filepath, "qc_test")
        
        # Introduce some quality issues
        specimens_list = list(dataset.specimens.values())
        
        # Remove species from one specimen
        specimens_list[0].species = ""
        
        # Add extreme outlier
        outlier_measurement = MorphologicalMeasurement(
            measurement_id="OUTLIER", specimen_id=specimens_list[1].specimen_id,
            trait_name="body_mass", value=1000.0, unit="g", measurement_method="test"
        )
        specimens_list[1].add_measurement(outlier_measurement)
        
        # Run quality control
        qc_report = self.importer.quality_control(dataset, outlier_threshold=2.0)
        
        assert qc_report['dataset_id'] == "qc_test"
        assert qc_report['total_specimens'] == 15
        assert len(qc_report['issues']) > 0  # Should have issues
        
        # Check for outliers
        if 'body_mass' in qc_report['outliers']:
            outliers = qc_report['outliers']['body_mass']
            assert len(outliers) > 0
            assert any(o['value'] == 1000.0 for o in outliers)
    
    def test_export_formats(self):
        """Test different export formats"""
        filepath = self.create_specimen_csv(n_specimens=3)
        dataset = self.importer.import_specimen_csv(filepath, "export_test")
        
        # Test long CSV export
        long_csv = Path(self.temp_dir) / "export_long.csv"
        self.importer.export_dataset(dataset, long_csv, format='csv')
        assert long_csv.exists()
        
        long_df = pd.read_csv(long_csv, comment='#')
        assert len(long_df) == 15  # 3 specimens × 5 measurements each
        
        # Test wide CSV export
        wide_csv = Path(self.temp_dir) / "export_wide.csv"
        self.importer.export_dataset(dataset, wide_csv, format='wide_csv')
        assert wide_csv.exists()
        
        wide_df = pd.read_csv(wide_csv, comment='#')
        assert len(wide_df) == 3  # One row per specimen
        
        # Test JSON export
        json_file = Path(self.temp_dir) / "export.json"
        self.importer.export_dataset(dataset, json_file, format='json')
        assert json_file.exists()
        
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        assert json_data['dataset_id'] == "export_test"
        assert len(json_data['specimens']) == 3
        assert 'statistics' in json_data


class TestMorphometricDataset:
    """Test suite for MorphometricDataset"""
    
    def create_test_dataset(self, n_specimens=5):
        """Helper to create test dataset"""
        specimens = {}
        trait_definitions = {
            'body_mass': {'standard_unit': 'g', 'category': 'body_size'},
            'total_length': {'standard_unit': 'mm', 'category': 'body_size'}
        }
        
        for i in range(n_specimens):
            specimen = Specimen(
                specimen_id=f"SPEC{i:03d}",
                species="Test species" if i < 3 else "Other species",
                sex="M" if i % 2 == 0 else "F"
            )
            
            # Add measurements
            mass = MorphologicalMeasurement(
                measurement_id=f"M{i}_mass", specimen_id=specimen.specimen_id,
                trait_name="body_mass", value=20.0 + i * 2, unit="g", measurement_method="scale"
            )
            length = MorphologicalMeasurement(
                measurement_id=f"M{i}_length", specimen_id=specimen.specimen_id,
                trait_name="total_length", value=180.0 + i * 5, unit="mm", measurement_method="ruler"
            )
            
            specimen.add_measurement(mass)
            specimen.add_measurement(length)
            specimens[specimen.specimen_id] = specimen
        
        return MorphometricDataset(
            dataset_id="test_dataset",
            specimens=specimens,
            trait_definitions=trait_definitions,
            study_metadata={}
        )
    
    def test_dataset_queries(self):
        """Test dataset query methods"""
        dataset = self.create_test_dataset(5)
        
        # Test species query
        test_species = dataset.get_specimens_by_species("Test species")
        assert len(test_species) == 3
        
        other_species = dataset.get_specimens_by_species("Other species")
        assert len(other_species) == 2
        
        # Test trait query
        mass_measurements = dataset.get_measurements_by_trait("body_mass")
        assert len(mass_measurements) == 5
        
        # Test available lists
        available_traits = dataset.get_available_traits()
        assert "body_mass" in available_traits
        assert "total_length" in available_traits
        
        available_species = dataset.get_available_species()
        assert "Test species" in available_species
        assert "Other species" in available_species


class TestIntegration:
    """Integration tests for field data workflow"""
    
    def test_full_field_data_workflow(self):
        """Test complete field data integration workflow"""
        temp_dir = tempfile.mkdtemp()
        
        # Step 1: Create environmental data
        env_importer = EnvironmentalDataImporter()
        
        # Create weather data
        weather_data = []
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(12):
            weather_data.append({
                'timestamp': (base_time + timedelta(hours=i)).isoformat(),
                'latitude': 40.0 + i * 0.001,
                'longitude': -74.0 + i * 0.001,
                'temperature': 20 + 5 * np.sin(i * np.pi / 12),
                'humidity': 60 + 10 * np.random.randn(),
                'pressure': 1013 + 3 * np.random.randn()
            })
        
        weather_df = pd.DataFrame(weather_data)
        weather_file = Path(temp_dir) / "field_weather.csv"
        weather_df.to_csv(weather_file, index=False)
        
        env_dataset = env_importer.import_weather_station_csv(
            weather_file, "FIELD_STATION", "field_study")
        
        # Step 2: Create morphology data
        morph_importer = MorphologyDataImporter()
        
        # Create specimen data
        specimen_data = []
        for i in range(5):
            specimen_data.append({
                'specimen_id': f"FIELD{i:03d}",
                'species': "Peromyscus maniculatus",
                'collection_date': (base_time.date() + timedelta(days=i)),
                'latitude': 40.0 + i * 0.002,
                'longitude': -74.0 + i * 0.002,
                'sex': 'M' if i % 2 == 0 else 'F',
                'body_mass': 25 + 3 * np.random.randn(),
                'total_length': 185 + 10 * np.random.randn(),
                'tail_length': 85 + 5 * np.random.randn()
            })
        
        specimen_df = pd.DataFrame(specimen_data)
        specimen_file = Path(temp_dir) / "field_specimens.csv"
        specimen_df.to_csv(specimen_file, index=False)
        
        morph_dataset = morph_importer.import_specimen_csv(specimen_file, "field_morphometry")
        
        # Step 3: Quality control
        qc_report = morph_importer.quality_control(morph_dataset)
        
        # Step 4: Export integrated results
        env_export = Path(temp_dir) / "environmental_summary.json"
        env_importer.export_dataset(env_dataset, env_export, format='json')
        
        morph_export = Path(temp_dir) / "morphometry_summary.json"
        morph_importer.export_dataset(morph_dataset, morph_export, format='json')
        
        # Verify workflow
        assert len(env_dataset.readings) == 12
        assert len(morph_dataset.specimens) == 5
        assert qc_report['total_specimens'] == 5
        assert env_export.exists()
        assert morph_export.exists()
        
        # Check data quality
        assert all(r.is_complete() for r in env_dataset.readings)
        assert all(len(s.measurements) == 3 for s in morph_dataset.specimens.values())


@pytest.mark.performance
class TestPerformance:
    """Performance tests for field data import"""
    
    def test_large_environmental_dataset(self):
        """Test performance with large environmental datasets"""
        import time
        
        temp_dir = tempfile.mkdtemp()
        
        # Create large weather dataset (1 week of 5-minute data = ~2000 points)
        n_points = 2000
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        data = []
        for i in range(n_points):
            timestamp = start_time + timedelta(minutes=i*5)
            data.append({
                'timestamp': timestamp.isoformat(),
                'latitude': 40.0 + 0.0001 * np.sin(i * 0.01),
                'longitude': -74.0 + 0.0001 * np.cos(i * 0.01),
                'temperature': 20 + 8 * np.sin(i * 2 * np.pi / 288),  # Daily cycle
                'humidity': 60 + 15 * np.random.randn(),
                'pressure': 1013 + 5 * np.random.randn()
            })
        
        df = pd.DataFrame(data)
        large_file = Path(temp_dir) / "large_weather.csv"
        df.to_csv(large_file, index=False)
        
        # Time the import
        importer = EnvironmentalDataImporter()
        start_time_import = time.time()
        dataset = importer.import_weather_station_csv(large_file, "LARGE_STATION", "performance_test")
        import_time = time.time() - start_time_import
        
        # Verify performance and correctness
        assert len(dataset.readings) == n_points
        assert import_time < 5.0  # Should complete within 5 seconds
        assert dataset.dataset_id == "performance_test"
        
        # Test query performance
        start_time_query = time.time()
        center = GPSCoordinate(40.0, -74.0)
        nearby = dataset.get_readings_by_location(center, 1000)
        query_time = time.time() - start_time_query
        
        assert query_time < 1.0  # Should query within 1 second
        assert len(nearby) > 0