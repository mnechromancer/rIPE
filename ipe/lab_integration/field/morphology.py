"""
DATA-003: Field Data Connectors - Morphology Data

This module implements functionality for importing and processing
morphometric data from field collections and museum specimens.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class MorphologicalMeasurement:
    """Represents a single morphological measurement"""

    measurement_id: str
    specimen_id: str
    trait_name: str
    value: float
    unit: str
    measurement_method: str  # "digital_caliper", "ruler", "image_analysis", etc.
    precision: Optional[float] = None  # measurement precision/accuracy
    observer: Optional[str] = None
    date_measured: Optional[datetime] = None
    notes: Optional[str] = None

    def __post_init__(self):
        """Validate measurement data"""
        if self.value < 0:
            raise ValueError(f"Measurement value cannot be negative: {self.value}")
        if not self.trait_name:
            raise ValueError("Trait name is required")


@dataclass
class Specimen:
    """Represents a biological specimen with metadata"""

    specimen_id: str
    species: str
    collection_date: Optional[datetime] = None
    collection_location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation_m: Optional[float] = None
    sex: Optional[str] = None  # "M", "F", "U" (unknown)
    age_class: Optional[str] = None  # "adult", "juvenile", "subadult"
    life_stage: Optional[str] = None  # "breeding", "non-breeding", "molt"
    collector: Optional[str] = None
    collection_method: Optional[str] = None
    preservation_method: Optional[str] = None
    catalog_number: Optional[str] = None
    museum_code: Optional[str] = None
    tissue_samples: List[str] = field(default_factory=list)
    measurements: List[MorphologicalMeasurement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_measurement(self, measurement: MorphologicalMeasurement):
        """Add a morphological measurement to this specimen"""
        measurement.specimen_id = self.specimen_id
        self.measurements.append(measurement)

    def get_measurement(self, trait_name: str) -> Optional[MorphologicalMeasurement]:
        """Get measurement for a specific trait"""
        for measurement in self.measurements:
            if measurement.trait_name == trait_name:
                return measurement
        return None

    def get_measurement_value(self, trait_name: str) -> Optional[float]:
        """Get measurement value for a specific trait"""
        measurement = self.get_measurement(trait_name)
        return measurement.value if measurement else None


@dataclass
class MorphometricDataset:
    """Contains morphometric data from multiple specimens"""

    dataset_id: str
    specimens: Dict[str, Specimen]
    trait_definitions: Dict[str, Dict[str, Any]]  # trait_name -> metadata
    study_metadata: Dict[str, Any]
    created_date: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Create indices for analysis"""
        self._species_index = {}
        self._trait_index = {}

        for specimen in self.specimens.values():
            # Index by species
            if specimen.species not in self._species_index:
                self._species_index[specimen.species] = []
            self._species_index[specimen.species].append(specimen)

            # Index by trait
            for measurement in specimen.measurements:
                trait_name = measurement.trait_name
                if trait_name not in self._trait_index:
                    self._trait_index[trait_name] = []
                self._trait_index[trait_name].append(measurement)

    def get_specimens_by_species(self, species: str) -> List[Specimen]:
        """Get all specimens of a given species"""
        return self._species_index.get(species, [])

    def get_measurements_by_trait(
        self, trait_name: str
    ) -> List[MorphologicalMeasurement]:
        """Get all measurements for a specific trait"""
        return self._trait_index.get(trait_name, [])

    def get_trait_statistics(
        self, trait_name: str, species: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate statistics for a trait"""
        if species:
            specimens = self.get_specimens_by_species(species)
            values = [s.get_measurement_value(trait_name) for s in specimens]
        else:
            measurements = self.get_measurements_by_trait(trait_name)
            values = [m.value for m in measurements]

        # Filter out None values
        values = [v for v in values if v is not None]

        if not values:
            return {}

        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
        }

    def get_available_traits(self) -> List[str]:
        """Get list of all available traits"""
        return list(self._trait_index.keys())

    def get_available_species(self) -> List[str]:
        """Get list of all species in dataset"""
        return list(self._species_index.keys())


class MorphologyDataImporter:
    """
    Importer for morphometric data from various sources

    Supports:
    - Museum specimen databases
    - Field collection data sheets
    - Image analysis results
    - Multi-observer data with quality control
    """

    def __init__(self):
        """Initialize morphology data importer"""
        self.standard_traits = self._load_standard_traits()
        self.unit_conversions = self._load_unit_conversions()

    def _load_standard_traits(self) -> Dict[str, Dict[str, Any]]:
        """Load standard morphological trait definitions"""
        return {
            # Body size measurements
            "body_mass": {
                "description": "Total body mass",
                "standard_unit": "g",
                "precision": 0.1,
                "category": "body_size",
            },
            "total_length": {
                "description": "Total body length from nose to tail tip",
                "standard_unit": "mm",
                "precision": 1.0,
                "category": "body_size",
            },
            "tail_length": {
                "description": "Tail length from base to tip",
                "standard_unit": "mm",
                "precision": 1.0,
                "category": "body_size",
            },
            "hind_foot_length": {
                "description": "Hind foot length excluding claw",
                "standard_unit": "mm",
                "precision": 0.5,
                "category": "body_size",
            },
            "ear_length": {
                "description": "External ear length",
                "standard_unit": "mm",
                "precision": 0.5,
                "category": "body_size",
            },
            # Skull measurements
            "skull_length": {
                "description": "Greatest skull length",
                "standard_unit": "mm",
                "precision": 0.1,
                "category": "skull",
            },
            "skull_width": {
                "description": "Greatest skull width",
                "standard_unit": "mm",
                "precision": 0.1,
                "category": "skull",
            },
            "mandible_length": {
                "description": "Mandible length",
                "standard_unit": "mm",
                "precision": 0.1,
                "category": "skull",
            },
            # Wing measurements (for birds/bats)
            "wing_chord": {
                "description": "Unflattened wing chord",
                "standard_unit": "mm",
                "precision": 1.0,
                "category": "flight",
            },
            "wingspan": {
                "description": "Wing span",
                "standard_unit": "mm",
                "precision": 5.0,
                "category": "flight",
            },
        }

    def _load_unit_conversions(self) -> Dict[str, Dict[str, float]]:
        """Load unit conversion factors"""
        return {
            "length": {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "ft": 304.8},
            "mass": {"g": 1.0, "kg": 1000.0, "mg": 0.001, "oz": 28.35, "lb": 453.6},
        }

    def import_specimen_csv(
        self, filepath: Union[str, Path], dataset_id: str
    ) -> MorphometricDataset:
        """
        Import specimen data from CSV

        Expected columns:
        - specimen_id (required)
        - species (required)
        - Additional specimen metadata columns
        - Morphometric measurement columns

        Args:
            filepath: Path to CSV file
            dataset_id: Dataset identifier

        Returns:
            MorphometricDataset object
        """
        filepath = Path(filepath)
        df = pd.read_csv(filepath)

        # Validate required columns
        if "specimen_id" not in df.columns:
            raise ValueError("CSV must contain 'specimen_id' column")
        if "species" not in df.columns:
            raise ValueError("CSV must contain 'species' column")

        specimens = {}
        trait_definitions = {}

        for _, row in df.iterrows():
            specimen_id = str(row["specimen_id"])

            # Parse collection date
            collection_date = None
            if "collection_date" in df.columns:
                try:
                    collection_date = pd.to_datetime(row["collection_date"])
                except Exception:
                    pass

            # Create specimen
            specimen = Specimen(
                specimen_id=specimen_id,
                species=str(row["species"]),
                collection_date=collection_date,
                collection_location=row.get("collection_location"),
                latitude=self._safe_float(row.get("latitude")),
                longitude=self._safe_float(row.get("longitude")),
                elevation_m=self._safe_float(row.get("elevation_m")),
                sex=row.get("sex"),
                age_class=row.get("age_class"),
                life_stage=row.get("life_stage"),
                collector=row.get("collector"),
                collection_method=row.get("collection_method"),
                preservation_method=row.get("preservation_method"),
                catalog_number=row.get("catalog_number"),
                museum_code=row.get("museum_code"),
            )

            # Add morphometric measurements
            for column in df.columns:
                # Skip metadata columns
                if column in [
                    "specimen_id",
                    "species",
                    "collection_date",
                    "collection_location",
                    "latitude",
                    "longitude",
                    "elevation_m",
                    "sex",
                    "age_class",
                    "life_stage",
                    "collector",
                    "collection_method",
                    "preservation_method",
                    "catalog_number",
                    "museum_code",
                ]:
                    continue

                value = self._safe_float(row[column])
                if value is not None:
                    # Determine unit from column name or use default
                    unit = self._infer_unit(column)

                    # Create measurement
                    measurement = MorphologicalMeasurement(
                        measurement_id=f"{specimen_id}_{column}",
                        specimen_id=specimen_id,
                        trait_name=column,
                        value=value,
                        unit=unit,
                        measurement_method="unknown",
                        date_measured=datetime.now(),
                    )

                    specimen.add_measurement(measurement)

                    # Add to trait definitions
                    if column not in trait_definitions:
                        trait_definitions[column] = self.standard_traits.get(
                            column,
                            {
                                "description": f"Measurement of {column}",
                                "standard_unit": unit,
                                "category": "morphometric",
                            },
                        )

            specimens[specimen_id] = specimen

        return MorphometricDataset(
            dataset_id=dataset_id,
            specimens=specimens,
            trait_definitions=trait_definitions,
            study_metadata={
                "source_file": str(filepath),
                "import_date": datetime.now().isoformat(),
                "total_specimens": len(specimens),
                "species_count": len(set(s.species for s in specimens.values())),
                "trait_count": len(trait_definitions),
            },
        )

    def import_measurements_csv(
        self,
        filepath: Union[str, Path],
        specimens_dict: Dict[str, Specimen],
        dataset_id: str,
    ) -> MorphometricDataset:
        """
        Import measurements from a separate CSV file

        Expected columns:
        - specimen_id
        - trait_name
        - value
        - unit (optional)
        - measurement_method (optional)
        - observer (optional)
        - date_measured (optional)

        Args:
            filepath: Path to measurements CSV file
            specimens_dict: Dictionary of existing specimens
            dataset_id: Dataset identifier

        Returns:
            Updated MorphometricDataset
        """
        filepath = Path(filepath)
        df = pd.read_csv(filepath)

        # Validate required columns
        required_columns = ["specimen_id", "trait_name", "value"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column")

        trait_definitions = {}

        for _, row in df.iterrows():
            specimen_id = str(row["specimen_id"])

            if specimen_id not in specimens_dict:
                print(
                    f"Warning: Specimen {specimen_id} not found, skipping measurement"
                )
                continue

            # Parse measurement date
            date_measured = None
            if "date_measured" in df.columns:
                try:
                    date_measured = pd.to_datetime(row["date_measured"])
                except Exception:
                    pass

            # Create measurement
            trait_name = str(row["trait_name"])
            value = float(row["value"])
            unit = row.get("unit", self._infer_unit(trait_name))

            measurement = MorphologicalMeasurement(
                measurement_id=(
                    f"{specimen_id}_{trait_name}_"
                    f"{len(specimens_dict[specimen_id].measurements)}"
                ),
                specimen_id=specimen_id,
                trait_name=trait_name,
                value=value,
                unit=unit,
                measurement_method=row.get("measurement_method", "unknown"),
                observer=row.get("observer"),
                date_measured=date_measured,
                notes=row.get("notes"),
            )

            specimens_dict[specimen_id].add_measurement(measurement)

            # Add to trait definitions
            if trait_name not in trait_definitions:
                trait_definitions[trait_name] = self.standard_traits.get(
                    trait_name,
                    {
                        "description": f"Measurement of {trait_name}",
                        "standard_unit": unit,
                        "category": "morphometric",
                    },
                )

        return MorphometricDataset(
            dataset_id=dataset_id,
            specimens=specimens_dict,
            trait_definitions=trait_definitions,
            study_metadata={
                "measurements_file": str(filepath),
                "import_date": datetime.now().isoformat(),
                "total_specimens": len(specimens_dict),
                "species_count": len(set(s.species for s in specimens_dict.values())),
                "trait_count": len(trait_definitions),
            },
        )

    def standardize_units(self, dataset: MorphometricDataset) -> MorphometricDataset:
        """
        Standardize measurement units across the dataset

        Args:
            dataset: Original dataset

        Returns:
            Dataset with standardized units
        """
        standardized_specimens = {}

        for specimen_id, specimen in dataset.specimens.items():
            # Create copy of specimen
            new_specimen = Specimen(
                specimen_id=specimen.specimen_id,
                species=specimen.species,
                collection_date=specimen.collection_date,
                collection_location=specimen.collection_location,
                latitude=specimen.latitude,
                longitude=specimen.longitude,
                elevation_m=specimen.elevation_m,
                sex=specimen.sex,
                age_class=specimen.age_class,
                life_stage=specimen.life_stage,
                collector=specimen.collector,
                collection_method=specimen.collection_method,
                preservation_method=specimen.preservation_method,
                catalog_number=specimen.catalog_number,
                museum_code=specimen.museum_code,
                tissue_samples=specimen.tissue_samples.copy(),
                metadata=specimen.metadata.copy(),
            )

            # Standardize measurements
            for measurement in specimen.measurements:
                trait_def = dataset.trait_definitions.get(measurement.trait_name, {})
                standard_unit = trait_def.get("standard_unit", measurement.unit)

                # Convert to standard unit
                converted_value = self._convert_units(
                    measurement.value, measurement.unit, standard_unit
                )

                standardized_measurement = MorphologicalMeasurement(
                    measurement_id=measurement.measurement_id,
                    specimen_id=measurement.specimen_id,
                    trait_name=measurement.trait_name,
                    value=converted_value,
                    unit=standard_unit,
                    measurement_method=measurement.measurement_method,
                    precision=measurement.precision,
                    observer=measurement.observer,
                    date_measured=measurement.date_measured,
                    notes=measurement.notes,
                )

                new_specimen.add_measurement(standardized_measurement)

            standardized_specimens[specimen_id] = new_specimen

        return MorphometricDataset(
            dataset_id=f"{dataset.dataset_id}_standardized",
            specimens=standardized_specimens,
            trait_definitions=dataset.trait_definitions,
            study_metadata={
                **dataset.study_metadata,
                "units_standardized": True,
                "standardization_date": datetime.now().isoformat(),
            },
        )

    def quality_control(
        self, dataset: MorphometricDataset, outlier_threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        Perform quality control checks on morphometric data

        Args:
            dataset: Dataset to check
            outlier_threshold: Z-score threshold for outlier detection

        Returns:
            Quality control report
        """
        report = {
            "dataset_id": dataset.dataset_id,
            "check_date": datetime.now().isoformat(),
            "total_specimens": len(dataset.specimens),
            "total_measurements": sum(
                len(s.measurements) for s in dataset.specimens.values()
            ),
            "issues": [],
            "outliers": {},
            "missing_data": {},
            "summary": {},
        }

        # Check for missing critical data
        missing_species = [
            s.specimen_id for s in dataset.specimens.values() if not s.species
        ]
        if missing_species:
            report["issues"].append(
                f"Missing species data: {len(missing_species)} specimens"
            )
            report["missing_data"]["species"] = missing_species

        # Check for duplicate measurements
        measurement_keys = {}
        duplicates = []
        for specimen in dataset.specimens.values():
            for measurement in specimen.measurements:
                key = (measurement.specimen_id, measurement.trait_name)
                if key in measurement_keys:
                    duplicates.append(key)
                measurement_keys[key] = measurement

        if duplicates:
            report["issues"].append(f"Duplicate measurements: {len(duplicates)} cases")
            report["duplicates"] = duplicates

        # Outlier detection for each trait
        # Collect all measurements by trait directly from specimens to include any newly added measurements
        trait_measurements = {}
        for specimen in dataset.specimens.values():
            for measurement in specimen.measurements:
                trait_name = measurement.trait_name
                if trait_name not in trait_measurements:
                    trait_measurements[trait_name] = []
                trait_measurements[trait_name].append(measurement)

        for trait_name, measurements in trait_measurements.items():
            values = [m.value for m in measurements]

            if len(values) < 5:  # Need minimum sample size
                continue

            mean_val = np.mean(values)
            std_val = np.std(values)

            if std_val == 0:  # Avoid division by zero
                continue

            outliers = []
            for measurement in measurements:
                z_score = abs(measurement.value - mean_val) / std_val
                if z_score > outlier_threshold:
                    outliers.append(
                        {
                            "specimen_id": measurement.specimen_id,
                            "trait": trait_name,
                            "value": measurement.value,
                            "z_score": z_score,
                        }
                    )

            if outliers:
                report["outliers"][trait_name] = outliers

        # Data completeness by trait
        trait_completeness = {}
        for trait_name, measurements in trait_measurements.items():
            completeness = len(measurements) / len(dataset.specimens)
            trait_completeness[trait_name] = completeness

        report["summary"] = {
            "traits_with_outliers": len(report["outliers"]),
            "total_outliers": sum(
                len(outliers) for outliers in report["outliers"].values()
            ),
            "trait_completeness": trait_completeness,
            "mean_completeness": (
                np.mean(list(trait_completeness.values())) if trait_completeness else 0
            ),
        }

        return report

    def export_dataset(
        self,
        dataset: MorphometricDataset,
        output_path: Union[str, Path],
        format: str = "csv",
        include_statistics: bool = True,
    ):
        """
        Export morphometric dataset to file

        Args:
            dataset: Dataset to export
            output_path: Output file path
            format: Export format ('csv', 'json', 'wide_csv')
            include_statistics: Whether to include summary statistics
        """
        output_path = Path(output_path)

        if format.lower() == "csv":
            self._export_long_csv(dataset, output_path, include_statistics)
        elif format.lower() == "wide_csv":
            self._export_wide_csv(dataset, output_path, include_statistics)
        elif format.lower() == "json":
            self._export_json(dataset, output_path, include_statistics)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if pd.isna(value) or value is None or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _infer_unit(self, trait_name: str) -> str:
        """Infer unit from trait name"""
        trait_name_lower = trait_name.lower()

        if any(keyword in trait_name_lower for keyword in ["mass", "weight"]):
            return "g"
        elif any(
            keyword in trait_name_lower
            for keyword in ["length", "width", "height", "diameter"]
        ):
            return "mm"
        else:
            return "mm"  # Default for morphometric measurements

    def _convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert between measurement units"""
        if from_unit == to_unit:
            return value

        # Determine measurement type
        length_units = self.unit_conversions["length"]
        mass_units = self.unit_conversions["mass"]

        if from_unit in length_units and to_unit in length_units:
            # Convert via mm
            mm_value = value * length_units[from_unit]
            return mm_value / length_units[to_unit]
        elif from_unit in mass_units and to_unit in mass_units:
            # Convert via grams
            g_value = value * mass_units[from_unit]
            return g_value / mass_units[to_unit]
        else:
            print(f"Warning: Cannot convert from {from_unit} to {to_unit}")
            return value

    def _export_long_csv(
        self, dataset: MorphometricDataset, output_path: Path, include_statistics: bool
    ):
        """Export dataset in long format (one row per measurement)"""
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            if include_statistics:
                writer.writerow([f"# Dataset: {dataset.dataset_id}"])
                writer.writerow([f"# Created: {dataset.created_date}"])
                writer.writerow([f"# Specimens: {len(dataset.specimens)}"])
                writer.writerow([f"# Species: {len(dataset.get_available_species())}"])
                writer.writerow([f"# Traits: {len(dataset.get_available_traits())}"])

            # Header
            header = [
                "specimen_id",
                "species",
                "trait_name",
                "value",
                "unit",
                "measurement_method",
                "observer",
                "date_measured",
                "collection_date",
                "collection_location",
                "latitude",
                "longitude",
                "sex",
                "age_class",
                "catalog_number",
                "museum_code",
            ]
            writer.writerow(header)

            # Data
            for specimen in dataset.specimens.values():
                for measurement in specimen.measurements:
                    row = [
                        measurement.specimen_id,
                        specimen.species,
                        measurement.trait_name,
                        measurement.value,
                        measurement.unit,
                        measurement.measurement_method,
                        measurement.observer or "",
                        (
                            measurement.date_measured.isoformat()
                            if measurement.date_measured
                            else ""
                        ),
                        (
                            specimen.collection_date.isoformat()
                            if specimen.collection_date
                            else ""
                        ),
                        specimen.collection_location or "",
                        specimen.latitude or "",
                        specimen.longitude or "",
                        specimen.sex or "",
                        specimen.age_class or "",
                        specimen.catalog_number or "",
                        specimen.museum_code or "",
                    ]
                    writer.writerow(row)

    def _export_wide_csv(
        self, dataset: MorphometricDataset, output_path: Path, include_statistics: bool
    ):
        """Export dataset in wide format (one row per specimen)"""
        # Get all traits
        all_traits = sorted(dataset.get_available_traits())

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            if include_statistics:
                writer.writerow([f"# Dataset: {dataset.dataset_id}"])
                writer.writerow([f"# Specimens: {len(dataset.specimens)}"])

            # Header
            header = [
                "specimen_id",
                "species",
                "collection_date",
                "collection_location",
                "latitude",
                "longitude",
                "sex",
                "age_class",
                "catalog_number",
                "museum_code",
            ]
            header.extend(all_traits)
            writer.writerow(header)

            # Data
            for specimen in dataset.specimens.values():
                row = [
                    specimen.specimen_id,
                    specimen.species,
                    (
                        specimen.collection_date.isoformat()
                        if specimen.collection_date
                        else ""
                    ),
                    specimen.collection_location or "",
                    specimen.latitude or "",
                    specimen.longitude or "",
                    specimen.sex or "",
                    specimen.age_class or "",
                    specimen.catalog_number or "",
                    specimen.museum_code or "",
                ]

                # Add trait values
                for trait in all_traits:
                    value = specimen.get_measurement_value(trait)
                    row.append(value if value is not None else "")

                writer.writerow(row)

    def _export_json(
        self, dataset: MorphometricDataset, output_path: Path, include_statistics: bool
    ):
        """Export dataset to JSON"""
        data = {
            "dataset_id": dataset.dataset_id,
            "created_date": dataset.created_date.isoformat(),
            "study_metadata": dataset.study_metadata,
            "trait_definitions": dataset.trait_definitions,
            "specimens": {},
        }

        if include_statistics:
            # Add summary statistics
            data["statistics"] = {}
            for trait in dataset.get_available_traits():
                trait_stats = dataset.get_trait_statistics(trait)
                if trait_stats:
                    data["statistics"][trait] = trait_stats

        # Add specimens
        for specimen_id, specimen in dataset.specimens.items():
            specimen_data = {
                "species": specimen.species,
                "collection_date": (
                    specimen.collection_date.isoformat()
                    if specimen.collection_date
                    else None
                ),
                "collection_location": specimen.collection_location,
                "coordinates": {
                    "latitude": specimen.latitude,
                    "longitude": specimen.longitude,
                    "elevation_m": specimen.elevation_m,
                },
                "metadata": {
                    "sex": specimen.sex,
                    "age_class": specimen.age_class,
                    "life_stage": specimen.life_stage,
                    "collector": specimen.collector,
                    "collection_method": specimen.collection_method,
                    "preservation_method": specimen.preservation_method,
                    "catalog_number": specimen.catalog_number,
                    "museum_code": specimen.museum_code,
                    "tissue_samples": specimen.tissue_samples,
                },
                "measurements": {},
            }

            # Add measurements
            for measurement in specimen.measurements:
                specimen_data["measurements"][measurement.trait_name] = {
                    "value": measurement.value,
                    "unit": measurement.unit,
                    "method": measurement.measurement_method,
                    "precision": measurement.precision,
                    "observer": measurement.observer,
                    "date": (
                        measurement.date_measured.isoformat()
                        if measurement.date_measured
                        else None
                    ),
                    "notes": measurement.notes,
                }

            data["specimens"][specimen_id] = specimen_data

        with open(output_path, "w") as jsonfile:
            json.dump(data, jsonfile, indent=2, default=str)
