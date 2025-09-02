"""
Prometheus metrics collection for IPE (Integrated Phenotypic Evolution) platform.
Provides comprehensive monitoring of simulation performance, system resources,
and business metrics.
"""

import time
import psutil
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    Enum,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from sqlalchemy import func
from sqlalchemy.orm import Session

# Import IPE models for database metrics
try:
    from ipe.data.models import (
        Simulation,
        Organism,
        PhysiologyMeasurement,
        EnvironmentalCondition,
    )
except ImportError:
    # Handle case where models aren't available yet
    Simulation = Organism = PhysiologyMeasurement = EnvironmentalCondition = None


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""

    enable_simulation_metrics: bool = True
    enable_system_metrics: bool = True
    enable_database_metrics: bool = True
    enable_api_metrics: bool = True
    collection_interval: int = 60  # seconds
    histogram_buckets: List[float] = field(
        default_factory=lambda: [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
        ]
    )


class IPEMetrics:
    """
    Central metrics collector for IPE platform.
    Integrates with Prometheus for monitoring and alerting.
    """

    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        registry: Optional[CollectorRegistry] = None,
    ):
        self.config = config or MetricsConfig()
        self.registry = registry or CollectorRegistry()

        # Initialize metrics
        self._init_simulation_metrics()
        self._init_system_metrics()
        self._init_database_metrics()
        self._init_api_metrics()
        self._init_business_metrics()

        # Track collection state
        self.last_collection = None
        self.collection_errors = 0

    def _init_simulation_metrics(self):
        """Initialize simulation-related metrics."""
        # Simulation execution metrics
        self.simulation_duration = Histogram(
            "ipe_simulation_duration_seconds",
            "Time taken to complete simulations",
            buckets=self.config.histogram_buckets,
            registry=self.registry,
        )

        self.simulation_status = Enum(
            "ipe_simulation_status",
            "Current status of simulations",
            states=["initialized", "running", "completed", "failed", "paused"],
            registry=self.registry,
        )

        self.simulations_total = Counter(
            "ipe_simulations_total",
            "Total number of simulations started",
            ["status"],
            registry=self.registry,
        )

        self.organisms_generated = Counter(
            "ipe_organisms_generated_total",
            "Total number of organisms generated in simulations",
            ["species"],
            registry=self.registry,
        )

        self.generation_time = Histogram(
            "ipe_generation_time_seconds",
            "Time taken to process each generation",
            buckets=self.config.histogram_buckets,
            registry=self.registry,
        )

        # Evolution metrics
        self.fitness_score = Histogram(
            "ipe_organism_fitness_score",
            "Distribution of organism fitness scores",
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        self.mutation_rate = Gauge(
            "ipe_mutation_rate",
            "Current mutation rate in active simulations",
            ["simulation_id"],
            registry=self.registry,
        )

        self.population_size = Gauge(
            "ipe_population_size",
            "Current population size in active simulations",
            ["simulation_id", "species"],
            registry=self.registry,
        )

    def _init_system_metrics(self):
        """Initialize system resource metrics."""
        # CPU metrics
        self.cpu_usage = Gauge(
            "ipe_cpu_usage_percent", "CPU usage percentage", registry=self.registry
        )

        self.cpu_cores = Info("ipe_cpu_info", "CPU information", registry=self.registry)

        # Memory metrics
        self.memory_usage = Gauge(
            "ipe_memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],  # total, available, used
            registry=self.registry,
        )

        self.memory_usage_percent = Gauge(
            "ipe_memory_usage_percent",
            "Memory usage percentage",
            registry=self.registry,
        )

        # Disk metrics
        self.disk_usage = Gauge(
            "ipe_disk_usage_bytes",
            "Disk usage in bytes",
            ["device", "type"],  # used, free, total
            registry=self.registry,
        )

        # Network metrics
        self.network_bytes = Counter(
            "ipe_network_bytes_total",
            "Total network bytes transferred",
            ["interface", "direction"],  # sent, received
            registry=self.registry,
        )

    def _init_database_metrics(self):
        """Initialize database-related metrics."""
        # Connection metrics
        self.db_connections = Gauge(
            "ipe_database_connections",
            "Number of active database connections",
            ["state"],  # active, idle
            registry=self.registry,
        )

        # Query performance
        self.db_query_duration = Histogram(
            "ipe_database_query_duration_seconds",
            "Database query execution time",
            ["table", "operation"],
            buckets=self.config.histogram_buckets,
            registry=self.registry,
        )

        # Data volume metrics
        self.db_table_size = Gauge(
            "ipe_database_table_size_bytes",
            "Size of database tables",
            ["table"],
            registry=self.registry,
        )

        self.db_row_count = Gauge(
            "ipe_database_rows",
            "Number of rows in database tables",
            ["table"],
            registry=self.registry,
        )

        # TimescaleDB specific metrics
        self.hypertable_chunks = Gauge(
            "ipe_timescale_chunks",
            "Number of chunks in TimescaleDB hypertables",
            ["hypertable"],
            registry=self.registry,
        )

        self.compression_ratio = Gauge(
            "ipe_timescale_compression_ratio",
            "Compression ratio for TimescaleDB chunks",
            ["hypertable"],
            registry=self.registry,
        )

    def _init_api_metrics(self):
        """Initialize API performance metrics."""
        # Request metrics
        self.http_requests_total = Counter(
            "ipe_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.http_request_duration = Histogram(
            "ipe_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            buckets=self.config.histogram_buckets,
            registry=self.registry,
        )

        self.http_request_size = Histogram(
            "ipe_http_request_size_bytes",
            "HTTP request size",
            ["method", "endpoint"],
            registry=self.registry,
        )

        self.http_response_size = Histogram(
            "ipe_http_response_size_bytes",
            "HTTP response size",
            ["method", "endpoint"],
            registry=self.registry,
        )

        # WebSocket metrics
        self.websocket_connections = Gauge(
            "ipe_websocket_connections",
            "Number of active WebSocket connections",
            registry=self.registry,
        )

        self.websocket_messages = Counter(
            "ipe_websocket_messages_total",
            "Total WebSocket messages",
            ["direction"],  # sent, received
            registry=self.registry,
        )

    def _init_business_metrics(self):
        """Initialize business/domain-specific metrics."""
        # User activity
        self.active_users = Gauge(
            "ipe_active_users",
            "Number of active users",
            ["timeframe"],  # daily, weekly, monthly
            registry=self.registry,
        )

        # Scientific metrics
        self.experiments_completed = Counter(
            "ipe_experiments_completed_total",
            "Total number of completed experiments",
            ["type"],  # simulation, analysis, export
            registry=self.registry,
        )

        self.data_points_processed = Counter(
            "ipe_data_points_processed_total",
            "Total number of data points processed",
            ["source"],  # simulation, import, calculation
            registry=self.registry,
        )

        # Platform health
        self.application_status = Enum(
            "ipe_application_status",
            "Overall application status",
            states=["healthy", "degraded", "unhealthy"],
            registry=self.registry,
        )

        self.last_backup = Gauge(
            "ipe_last_backup_timestamp",
            "Timestamp of last successful backup",
            registry=self.registry,
        )

    def collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)

            cpu_info = {
                "cores": str(psutil.cpu_count()),
                "cores_physical": str(psutil.cpu_count(logical=False)),
                "frequency": str(
                    psutil.cpu_freq().current if psutil.cpu_freq() else "unknown"
                ),
            }
            self.cpu_cores.info(cpu_info)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.memory_usage.labels("total").set(memory.total)
            self.memory_usage.labels("available").set(memory.available)
            self.memory_usage.labels("used").set(memory.used)
            self.memory_usage_percent.set(memory.percent)

            # Disk metrics
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    device = partition.device.replace("/", "_")
                    self.disk_usage.labels(device, "total").set(usage.total)
                    self.disk_usage.labels(device, "used").set(usage.used)
                    self.disk_usage.labels(device, "free").set(usage.free)
                except PermissionError:
                    continue

            # Network metrics
            network = psutil.net_io_counters(pernic=True)
            for interface, stats in network.items():
                self.network_bytes.labels(interface, "sent")._value._value = (
                    stats.bytes_sent
                )
                self.network_bytes.labels(interface, "received")._value._value = (
                    stats.bytes_recv
                )

        except Exception as e:
            self.collection_errors += 1
            print(f"Error collecting system metrics: {e}")

    def collect_database_metrics(self, db_session: Session):
        """Collect database-related metrics."""
        if not db_session or not Simulation:
            return

        try:
            # Table row counts
            tables = {
                "simulations": Simulation,
                "organisms": Organism,
                "physiology_measurements": PhysiologyMeasurement,
                "environmental_conditions": EnvironmentalCondition,
            }

            for table_name, model in tables.items():
                if model:
                    count = db_session.query(func.count(model.id)).scalar()
                    self.db_row_count.labels(table_name).set(count)

            # Simulation status distribution
            status_counts = (
                db_session.query(Simulation.status, func.count(Simulation.id))
                .group_by(Simulation.status)
                .all()
            )

            for status, count in status_counts:
                self.simulations_total.labels(status)._value._value = count

            # Active simulations metrics
            active_sims = (
                db_session.query(Simulation)
                .filter(Simulation.status == "running")
                .all()
            )

            for sim in active_sims:
                self.mutation_rate.labels(str(sim.id)).set(sim.mutation_rate)

                # Population size by species
                species_counts = (
                    db_session.query(Organism.species_id, func.count(Organism.id))
                    .filter(Organism.simulation_id == sim.id, Organism.is_alive is True)
                    .group_by(Organism.species_id)
                    .all()
                )

                for species, count in species_counts:
                    self.population_size.labels(str(sim.id), species).set(count)

        except Exception as e:
            self.collection_errors += 1
            print(f"Error collecting database metrics: {e}")

    def record_simulation_event(
        self, simulation_id: str, event_type: str, duration: Optional[float] = None
    ):
        """Record simulation-related events."""
        if event_type == "completed" and duration:
            self.simulation_duration.observe(duration)

        self.simulations_total.labels(event_type).inc()

    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float,
        request_size: int = 0,
        response_size: int = 0,
    ):
        """Record API request metrics."""
        self.http_requests_total.labels(method, endpoint, str(status)).inc()
        self.http_request_duration.labels(method, endpoint).observe(duration)

        if request_size > 0:
            self.http_request_size.labels(method, endpoint).observe(request_size)
        if response_size > 0:
            self.http_response_size.labels(method, endpoint).observe(response_size)

    def collect_all_metrics(self, db_session: Optional[Session] = None):
        """Collect all available metrics."""
        start_time = time.time()

        if self.config.enable_system_metrics:
            self.collect_system_metrics()

        if self.config.enable_database_metrics and db_session:
            self.collect_database_metrics(db_session)

        # Update application status based on collection success
        if self.collection_errors == 0:
            self.application_status.state("healthy")
        elif self.collection_errors < 5:
            self.application_status.state("degraded")
        else:
            self.application_status.state("unhealthy")

        self.last_collection = datetime.now()
        collection_duration = time.time() - start_time

        print(f"Metrics collection completed in {collection_duration:.3f}s")

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get the content type for metrics response."""
        return CONTENT_TYPE_LATEST


# Global metrics instance
metrics = IPEMetrics()


def get_metrics_instance() -> IPEMetrics:
    """Get the global metrics instance."""
    return metrics


def setup_metrics(config: Optional[MetricsConfig] = None) -> IPEMetrics:
    """Setup and configure the global metrics instance."""
    global metrics
    metrics = IPEMetrics(config)
    return metrics