"""
Tests for DATA-001: Respirometry Data Import

Tests the Sable Systems importer and generic respirometry parser
functionality including file parsing, data validation, and quality assessment.
"""

import pytest
import tempfile
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from ipe.lab_integration.respirometry.sable_import import (
    SableSystemsImporter,
    SableSessionData,
    RespirometryMeasurement,
)
from ipe.lab_integration.respirometry.parser import (
    GenericRespirometryParser,
    MetabolicSummary,
    QualityAssessment,
)


class TestRespirometryMeasurement:
    """Test suite for RespirometryMeasurement"""

    def test_measurement_creation(self):
        """Test basic measurement creation"""
        timestamp = datetime.now()
        measurement = RespirometryMeasurement(
            timestamp=timestamp,
            vo2_ml_min=2.5,
            vco2_ml_min=2.0,
            rer=0.8,
            chamber_temp_c=25.0,
            chamber_humidity_percent=50.0,
            ambient_pressure_kpa=101.325,
        )

        assert measurement.timestamp == timestamp
        assert measurement.vo2_ml_min == 2.5
        assert measurement.vco2_ml_min == 2.0
        assert measurement.rer == 0.8
        assert not measurement.baseline_corrected

    def test_measurement_validation(self):
        """Test measurement validation"""
        timestamp = datetime.now()

        # Test negative VO2
        with pytest.raises(ValueError, match="VO2 cannot be negative"):
            RespirometryMeasurement(
                timestamp=timestamp,
                vo2_ml_min=-1.0,
                vco2_ml_min=2.0,
                rer=0.8,
                chamber_temp_c=25.0,
                chamber_humidity_percent=50.0,
                ambient_pressure_kpa=101.325,
            )

        # Test invalid RER
        with pytest.raises(ValueError, match="RER .* outside physiological range"):
            RespirometryMeasurement(
                timestamp=timestamp,
                vo2_ml_min=2.5,
                vco2_ml_min=2.0,
                rer=3.0,  # Invalid RER
                chamber_temp_c=25.0,
                chamber_humidity_percent=50.0,
                ambient_pressure_kpa=101.325,
            )


class TestSableSessionData:
    """Test suite for SableSessionData"""

    def create_test_session(self, n_measurements=10):
        """Helper to create test session data"""
        start_time = datetime.now()
        measurements = []

        for i in range(n_measurements):
            measurement = RespirometryMeasurement(
                timestamp=start_time + timedelta(seconds=i),
                vo2_ml_min=2.5 + 0.1 * np.sin(i * 0.1),  # Small variation
                vco2_ml_min=2.0 + 0.1 * np.sin(i * 0.1),
                rer=0.8,
                chamber_temp_c=25.0,
                chamber_humidity_percent=50.0,
                ambient_pressure_kpa=101.325,
            )
            measurements.append(measurement)

        return SableSessionData(
            session_id="test_session",
            subject_id="test_subject",
            start_time=start_time,
            end_time=start_time + timedelta(seconds=n_measurements - 1),
            measurements=measurements,
            metadata={"test": "data"},
        )

    def test_session_properties(self):
        """Test session property calculations"""
        session = self.create_test_session(60)  # 1 minute

        assert session.duration_minutes == pytest.approx(
            59 / 60, abs=0.01
        )  # 59 seconds
        assert session.mean_vo2 > 0
        assert session.mean_vco2 > 0
        assert session.mean_rer == pytest.approx(0.8, abs=0.1)


class TestSableSystemsImporter:
    """Test suite for SableSystemsImporter"""

    def setup_method(self):
        """Setup test fixtures"""
        self.importer = SableSystemsImporter()
        self.temp_dir = tempfile.mkdtemp()

    def create_test_csv_file(self, filename="test.csv", n_points=10):
        """Create a test CSV file"""
        filepath = Path(self.temp_dir) / filename

        with open(filepath, "w") as f:
            # Write header comments
            f.write("# Subject_ID: TEST001\n")
            f.write("# Date: 2024-01-01\n")
            f.write("# Experiment: Respirometry Test\n")

            # Write data
            base_time = datetime.now()
            for i in range(n_points):
                timestamp = (base_time + timedelta(seconds=i)).strftime("%H:%M:%S")
                vo2 = 2.5 + 0.1 * np.sin(i * 0.1)
                vco2 = 2.0 + 0.1 * np.sin(i * 0.1)
                temp = 25.0 + 0.5 * np.random.randn()
                humidity = 50.0 + 2.0 * np.random.randn()
                pressure = 101.325
                flow = 500.0

                f.write(
                    f"{timestamp},{vo2:.3f},{vco2:.3f},{temp:.1f},{humidity:.1f},{pressure:.1f},{flow:.1f}\n"
                )

        return filepath

    def test_csv_import(self):
        """Test CSV file import"""
        filepath = self.create_test_csv_file()
        session = self.importer.import_session(filepath)

        assert session.session_id == "test"
        assert session.subject_id == "TEST001"
        assert len(session.measurements) == 10
        assert all(m.vo2_ml_min > 0 for m in session.measurements)
        assert all(0.1 <= m.rer <= 2.0 for m in session.measurements)

    def test_batch_import(self):
        """Test batch import from directory"""
        # Create multiple test files
        self.create_test_csv_file("test1.csv", 5)
        self.create_test_csv_file("test2.csv", 8)
        self.create_test_csv_file("ignore.txt", 3)  # Should be ignored

        sessions = self.importer.batch_import(self.temp_dir)

        assert len(sessions) == 2  # Only CSV files should be imported
        assert all(isinstance(s, SableSessionData) for s in sessions)

    def test_temperature_correction(self):
        """Test temperature correction"""
        importer = SableSystemsImporter(temperature_correction=True)
        vo2_orig, vco2_orig = 2.5, 2.0
        temp_c, pressure_kpa = 30.0, 101.325

        vo2_corr, vco2_corr = importer._apply_temperature_correction(
            vo2_orig, vco2_orig, temp_c, pressure_kpa
        )

        # At higher temperature, corrected values should be lower
        assert vo2_corr < vo2_orig
        assert vco2_corr < vco2_orig

    def test_baseline_correction(self):
        """Test baseline drift correction"""
        importer = SableSystemsImporter(baseline_correction=True)
        filepath = self.create_test_csv_file("drift_test.csv", 100)

        session = importer.import_session(filepath)

        # Check that baseline correction was applied
        assert any(m.baseline_corrected for m in session.measurements)
        assert "baseline_correction" in session.metadata
        assert session.baseline_period is not None

    def test_export_csv(self):
        """Test CSV export functionality"""
        filepath = self.create_test_csv_file()
        session = self.importer.import_session(filepath)

        output_path = Path(self.temp_dir) / "exported.csv"
        self.importer.export_session(session, output_path, format="csv")

        assert output_path.exists()

        # Verify export content
        with open(output_path, "r") as f:
            content = f.read()
            assert "Session:" in content
            assert "VO2_ml_min" in content

    def test_export_json(self):
        """Test JSON export functionality"""
        filepath = self.create_test_csv_file()
        session = self.importer.import_session(filepath)

        output_path = Path(self.temp_dir) / "exported.json"
        self.importer.export_session(session, output_path, format="json")

        assert output_path.exists()

        import json

        with open(output_path, "r") as f:
            data = json.load(f)
            assert "session_id" in data
            assert "measurements" in data
            assert "summary" in data

    def test_unsupported_format(self):
        """Test handling of unsupported file formats"""
        unsupported_file = Path(self.temp_dir) / "test.txt"
        unsupported_file.write_text("dummy content")

        with pytest.raises(ValueError, match="Unsupported file format"):
            self.importer.import_session(unsupported_file)


class TestGenericRespirometryParser:
    """Test suite for GenericRespirometryParser"""

    def setup_method(self):
        """Setup test fixtures"""
        self.parser = GenericRespirometryParser()

    def create_test_session(self):
        """Create test session data"""
        measurements = []
        base_time = datetime.now()

        for i in range(100):
            measurement = RespirometryMeasurement(
                timestamp=base_time + timedelta(seconds=i),
                vo2_ml_min=2.5 + 0.2 * np.sin(i * 0.1),
                vco2_ml_min=2.0 + 0.15 * np.sin(i * 0.1),
                rer=0.8,
                chamber_temp_c=25.0,
                chamber_humidity_percent=50.0,
                ambient_pressure_kpa=101.325,
            )
            measurements.append(measurement)

        return SableSessionData(
            session_id="test_session",
            subject_id="test_subject",
            start_time=base_time,
            end_time=base_time + timedelta(seconds=99),
            measurements=measurements,
            metadata={},
        )

    def test_supported_formats(self):
        """Test getting supported formats"""
        formats = self.parser.get_supported_formats()
        assert ".csv" in formats
        assert ".exp" in formats

    def test_calculate_summary(self):
        """Test summary calculation"""
        session = self.create_test_session()
        summary = self.parser.calculate_summary(session)

        assert isinstance(summary, MetabolicSummary)
        assert summary.mean_vo2 > 0
        assert summary.mean_vco2 > 0
        assert summary.measurement_count == 100
        assert summary.vo2_coefficient_variation >= 0

    def test_filter_measurements(self):
        """Test measurement filtering"""
        session = self.create_test_session()

        # Filter with VO2 range
        filtered = self.parser.filter_measurements(session, min_vo2=2.0, max_vo2=3.0)

        assert len(filtered.measurements) <= len(session.measurements)
        assert all(2.0 <= m.vo2_ml_min <= 3.0 for m in filtered.measurements)
        assert filtered.metadata["filtered"] is True

    def test_smooth_measurements(self):
        """Test measurement smoothing"""
        session = self.create_test_session()
        smoothed = self.parser.smooth_measurements(session, window_size=5)

        assert len(smoothed.measurements) == len(session.measurements)
        assert smoothed.metadata["smoothed"] is True
        assert smoothed.metadata["smooth_window"] == 5

        # Smoothed data should have lower variance
        orig_vo2_std = np.std([m.vo2_ml_min for m in session.measurements])
        smooth_vo2_std = np.std([m.vo2_ml_min for m in smoothed.measurements])
        assert smooth_vo2_std <= orig_vo2_std


class TestQualityAssessment:
    """Test suite for QualityAssessment"""

    def create_session_with_quality(self, quality_type="good"):
        """Create sessions with different quality characteristics"""
        base_time = datetime.now()
        measurements = []

        if quality_type == "good":
            # High quality data
            for i in range(300):  # 5 minutes
                measurement = RespirometryMeasurement(
                    timestamp=base_time + timedelta(seconds=i),
                    vo2_ml_min=2.5 + 0.05 * np.random.randn(),  # Low noise
                    vco2_ml_min=2.0 + 0.04 * np.random.randn(),
                    rer=0.8,
                    chamber_temp_c=25.0 + 0.1 * np.random.randn(),  # Stable temp
                    chamber_humidity_percent=50.0,
                    ambient_pressure_kpa=101.325,
                    flow_rate_ml_min=500.0,
                )
                measurements.append(measurement)

        elif quality_type == "poor":
            # Poor quality data
            for i in range(30):  # Very short
                measurement = RespirometryMeasurement(
                    timestamp=base_time + timedelta(seconds=i),
                    vo2_ml_min=2.5 + 0.5 * np.random.randn(),  # High noise
                    vco2_ml_min=2.0 + 0.4 * np.random.randn(),
                    rer=max(
                        0.1, min(2.0, 0.8 + 0.2 * np.random.randn())
                    ),  # Variable but valid RER
                    chamber_temp_c=25.0 + 3.0 * np.random.randn(),  # Unstable temp
                    chamber_humidity_percent=50.0,
                    ambient_pressure_kpa=101.325,
                    flow_rate_ml_min=None,  # Missing flow data
                )
                measurements.append(measurement)

        return SableSessionData(
            session_id="test_session",
            subject_id="test_subject",
            start_time=base_time,
            end_time=base_time + timedelta(seconds=len(measurements) - 1),
            measurements=measurements,
            metadata={},
        )

    def test_good_quality_assessment(self):
        """Test assessment of good quality data"""
        session = self.create_session_with_quality("good")
        assessment = QualityAssessment.assess_data_quality(session)

        assert assessment["quality_score"] >= 80
        assert assessment["quality_rating"] in ["Good", "Excellent"]
        assert len(assessment["issues"]) <= 1  # Should have few issues

    def test_poor_quality_assessment(self):
        """Test assessment of poor quality data"""
        session = self.create_session_with_quality("poor")
        assessment = QualityAssessment.assess_data_quality(session)

        assert assessment["quality_score"] < 70
        assert assessment["quality_rating"] in ["Poor", "Very Poor"]
        assert len(assessment["issues"]) > 2  # Should have multiple issues

    def test_quality_recommendations(self):
        """Test quality-based recommendations"""
        session = self.create_session_with_quality("poor")
        assessment = QualityAssessment.assess_data_quality(session)
        recommendations = QualityAssessment.recommend_processing(assessment)

        assert len(recommendations) > 0
        # Should include relevant recommendations for poor data
        rec_text = " ".join(recommendations)
        # Check for any reasonable recommendation - not necessarily smoothing
        assert any(
            keyword in rec_text.lower()
            for keyword in ["smoothing", "temperature", "filtering", "validation"]
        )

    def test_empty_session_assessment(self):
        """Test assessment with no measurements"""
        empty_session = SableSessionData(
            session_id="empty",
            subject_id="empty",
            start_time=datetime.now(),
            end_time=datetime.now(),
            measurements=[],
            metadata={},
        )

        assessment = QualityAssessment.assess_data_quality(empty_session)
        assert assessment["quality_score"] == 0
        assert "No measurements found" in assessment["issues"]


class TestIntegration:
    """Integration tests for the complete respirometry system"""

    def test_full_workflow(self):
        """Test complete import-process-export workflow"""
        # Create temporary test data
        temp_dir = tempfile.mkdtemp()
        csv_file = Path(temp_dir) / "test.csv"

        # Create test CSV file
        with open(csv_file, "w") as f:
            f.write("# Subject_ID: INTEGRATION_TEST\n")
            f.write("# Experiment: Full workflow test\n")

            base_time = datetime.now()
            for i in range(60):  # 1 minute of data
                timestamp = (base_time + timedelta(seconds=i)).strftime("%H:%M:%S")
                vo2 = 2.5 + 0.1 * np.sin(i * 0.1)
                vco2 = 2.0 + 0.08 * np.sin(i * 0.1)
                f.write(f"{timestamp},{vo2:.3f},{vco2:.3f},25.0,50.0,101.3,500.0\n")

        # Step 1: Import data
        importer = SableSystemsImporter(baseline_correction=True)
        session = importer.import_session(csv_file)

        # Step 2: Process data
        parser = GenericRespirometryParser()
        summary = parser.calculate_summary(session)
        quality = QualityAssessment.assess_data_quality(session)

        # Step 3: Apply processing based on quality
        if quality["quality_score"] < 90:
            processed_session = parser.smooth_measurements(session, window_size=5)
        else:
            processed_session = session

        # Step 4: Export results
        output_file = Path(temp_dir) / "processed_output.json"
        importer.export_session(processed_session, output_file, format="json")

        # Verify workflow
        assert session.subject_id == "INTEGRATION_TEST"
        assert len(session.measurements) == 60
        assert summary.measurement_count == 60
        assert "quality_score" in quality
        assert output_file.exists()

        # Verify exported data integrity
        import json

        with open(output_file, "r") as f:
            exported_data = json.load(f)
            assert exported_data["subject_id"] == "INTEGRATION_TEST"
            assert len(exported_data["measurements"]) == 60


@pytest.mark.performance
class TestPerformance:
    """Performance tests for respirometry import"""

    def test_large_file_import(self):
        """Test import performance with large files"""
        import time

        # Create large test file
        temp_dir = tempfile.mkdtemp()
        large_csv = Path(temp_dir) / "large_test.csv"

        with open(large_csv, "w") as f:
            f.write("# Subject_ID: PERF_TEST\n")

            base_time = datetime.now()
            for i in range(10000):  # ~3 hours at 1Hz
                timestamp = (base_time + timedelta(seconds=i)).strftime("%H:%M:%S")
                vo2 = 2.5 + 0.1 * np.sin(i * 0.001)
                vco2 = 2.0 + 0.08 * np.sin(i * 0.001)
                f.write(f"{timestamp},{vo2:.3f},{vco2:.3f},25.0,50.0,101.3,500.0\\n")

        # Time the import
        importer = SableSystemsImporter()
        start_time = time.time()
        session = importer.import_session(large_csv)
        import_time = time.time() - start_time

        # Verify performance and correctness
        assert len(session.measurements) == 10000
        assert import_time < 5.0  # Should import within 5 seconds
        assert session.mean_vo2 > 0

    def test_batch_import_performance(self):
        """Test batch import performance"""
        import time

        temp_dir = tempfile.mkdtemp()

        # Create multiple files
        for file_num in range(10):
            csv_file = Path(temp_dir) / f"batch_test_{file_num}.csv"
            with open(csv_file, "w") as f:
                f.write(f"# Subject_ID: BATCH_{file_num}\n")

                for i in range(100):  # Small files for batch test
                    timestamp = f"{i//3600:02d}:{(i % 3600)//60:02d}:{i % 60:02d}"
                    vo2 = 2.5 + 0.1 * np.random.randn()
                    vco2 = 2.0 + 0.08 * np.random.randn()
                    f.write(
                        f"{timestamp},{vo2:.3f},{vco2:.3f},25.0,50.0,101.3,500.0\\n"
                    )

        # Time batch import
        importer = SableSystemsImporter()
        start_time = time.time()
        sessions = importer.batch_import(temp_dir)
        batch_time = time.time() - start_time

        # Verify performance
        assert len(sessions) == 10
        assert batch_time < 2.0  # Should complete within 2 seconds
        assert all(len(s.measurements) == 100 for s in sessions)
