"""
API-003: Output Formatters

Utilities for formatting simulation data into various output formats
including CSV, JSON, HDF5, and publication-ready figures.
"""

import pandas as pd
import numpy as np
import json
import h5py
import io
import zipfile
from typing import Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)


class DataFormatter:
    """Base class for data formatting operations."""

    def __init__(self):
        self.supported_formats = ["csv", "json", "hdf5", "xlsx"]

    def validate_format(self, format_type: str) -> bool:
        """Validate if format is supported."""
        return format_type.lower() in self.supported_formats

    def format_data(
        self, data: Dict[str, Any], format_type: str, output_path: Optional[str] = None
    ) -> Union[str, bytes, io.BytesIO]:
        """Format data according to specified format."""
        if not self.validate_format(format_type):
            raise ValueError(f"Unsupported format: {format_type}")

        if format_type == "csv":
            return self._to_csv(data)
        elif format_type == "json":
            return self._to_json(data)
        elif format_type == "hdf5":
            return self._to_hdf5(data, output_path)
        elif format_type == "xlsx":
            return self._to_excel(data)
        else:
            raise ValueError(f"Format not implemented: {format_type}")

    def _to_csv(self, data: Dict[str, Any]) -> io.BytesIO:
        """Convert data to CSV format."""
        buffer = io.BytesIO()

        # Create a zip file containing multiple CSV files
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    csv_content = value.to_csv(index=False)
                    zf.writestr(f"{key}.csv", csv_content)
                elif isinstance(value, dict):
                    # Convert dict to JSON and save as text file
                    json_content = json.dumps(value, indent=2)
                    zf.writestr(f"{key}.json", json_content)
                elif isinstance(value, list):
                    # Convert list to simple CSV
                    if value and isinstance(value[0], dict):
                        # List of dictionaries -> DataFrame -> CSV
                        df = pd.DataFrame(value)
                        csv_content = df.to_csv(index=False)
                        zf.writestr(f"{key}.csv", csv_content)
                    else:
                        # Simple list -> single column CSV
                        df = pd.DataFrame({key: value})
                        csv_content = df.to_csv(index=False)
                        zf.writestr(f"{key}.csv", csv_content)

        buffer.seek(0)
        return buffer

    def _to_json(self, data: Dict[str, Any]) -> str:
        """Convert data to JSON format."""
        # Convert pandas DataFrames to dictionaries
        json_data = {}
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                json_data[key] = value.to_dict("records")
            elif isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif pd.isna(value):  # Handle NaN values
                json_data[key] = None
            else:
                json_data[key] = value

        return json.dumps(json_data, indent=2, default=self._json_serializer)

    def _to_hdf5(self, data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Convert data to HDF5 format."""
        if output_path is None:
            import tempfile

            output_path = tempfile.mktemp(suffix=".h5")

        with h5py.File(output_path, "w") as hf:
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    # Create a group for each DataFrame
                    grp = hf.create_group(key)
                    for col in value.columns:
                        col_data = value[col].values
                        # Handle different data types
                        if col_data.dtype == "object":
                            # Convert object columns to strings
                            col_data = col_data.astype(str)
                        grp.create_dataset(col, data=col_data)

                    # Store column names as attribute
                    grp.attrs["columns"] = [
                        col.encode("utf-8") for col in value.columns
                    ]
                    grp.attrs["index"] = value.index.tolist()

                elif isinstance(value, (list, tuple)):
                    # Store lists as datasets
                    if value:  # Only if not empty
                        hf.create_dataset(key, data=np.array(value))

                elif isinstance(value, dict):
                    # Store dictionaries as attributes in a group
                    grp = hf.create_group(key)
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float, str, bool)):
                            grp.attrs[sub_key] = sub_value
                        elif isinstance(sub_value, (list, tuple)):
                            grp.create_dataset(sub_key, data=np.array(sub_value))

                elif isinstance(value, (int, float, str, bool)):
                    # Store simple values as attributes
                    hf.attrs[key] = value

        return output_path

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy and pandas objects."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        return str(obj)


class StatisticalSummaryFormatter:
    """Specialized formatter for statistical summaries."""

    def __init__(self):
        self.summary_fields = [
            "mean",
            "median",
            "std",
            "var",
            "min",
            "max",
            "quantiles",
            "skewness",
            "kurtosis",
            "count",
        ]

    def generate_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistical summary."""
        summary = {
            "metadata": {
                "generated_at": pd.Timestamp.now().isoformat(),
                "data_types": {key: str(type(value)) for key, value in data.items()},
            },
            "numerical_summary": {},
            "categorical_summary": {},
            "data_quality": {},
        }

        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                summary["numerical_summary"][key] = self._analyze_dataframe(value)
                summary["data_quality"][key] = self._assess_data_quality(value)
            elif isinstance(value, (list, np.ndarray)):
                if len(value) > 0:
                    arr = np.array(value)
                    if np.issubdtype(arr.dtype, np.number):
                        summary["numerical_summary"][key] = self._analyze_numeric_array(
                            arr
                        )
            elif isinstance(value, dict):
                summary["metadata"][f"{key}_keys"] = list(value.keys())

        return summary

    def _analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a pandas DataFrame."""
        analysis = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numerical_columns": {},
            "categorical_columns": {},
        }

        # Analyze numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            series = df[col].dropna()
            if len(series) > 0:
                analysis["numerical_columns"][col] = {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "quantiles": {
                        "0.25": float(series.quantile(0.25)),
                        "0.75": float(series.quantile(0.75)),
                    },
                    "missing_count": int(df[col].isna().sum()),
                    "unique_count": int(series.nunique()),
                }

        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            series = df[col].dropna()
            if len(series) > 0:
                value_counts = series.value_counts()
                analysis["categorical_columns"][col] = {
                    "unique_count": int(series.nunique()),
                    "top_values": value_counts.head(10).to_dict(),
                    "missing_count": int(df[col].isna().sum()),
                }

        return analysis

    def _analyze_numeric_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """Analyze a numeric array."""
        arr_clean = arr[~np.isnan(arr)]
        if len(arr_clean) == 0:
            return {"error": "All values are NaN"}

        return {
            "mean": float(np.mean(arr_clean)),
            "median": float(np.median(arr_clean)),
            "std": float(np.std(arr_clean)),
            "min": float(np.min(arr_clean)),
            "max": float(np.max(arr_clean)),
            "count": int(len(arr_clean)),
            "missing_count": int(len(arr) - len(arr_clean)),
        }

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics."""
        total_cells = df.size
        missing_cells = df.isna().sum().sum()

        return {
            "total_cells": int(total_cells),
            "missing_cells": int(missing_cells),
            "missing_percentage": (
                float(missing_cells / total_cells * 100) if total_cells > 0 else 0
            ),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        }


# Export the main classes
__all__ = ["DataFormatter", "StatisticalSummaryFormatter"]
