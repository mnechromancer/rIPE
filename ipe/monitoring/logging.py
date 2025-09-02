"""
Structured logging system for IPE (Integrated Phenotypic Evolution) platform.
Provides comprehensive logging with JSON formatting, correlation IDs, and
integration with monitoring systems.
"""

import json
import logging
import logging.config
import sys
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from threading import local
import os

# Thread-local storage for correlation IDs
_context = local()


@dataclass
class LogConfig:
    """Configuration for the logging system."""

    level: str = "INFO"
    format_type: str = "json"  # json or text
    include_correlation_id: bool = True
    include_caller_info: bool = True
    log_to_file: bool = True
    log_file_path: str = "/var/log/ipe/application.log"
    log_file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_file_backup_count: int = 5
    enable_console_output: bool = True
    sensitive_fields: list = field(
        default_factory=lambda: ["password", "token", "secret", "key"]
    )


class CorrelationIDFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record):
        record.correlation_id = getattr(_context, "correlation_id", None)
        return True


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def __init__(self, include_caller_info=True, sensitive_fields=None):
        super().__init__()
        self.include_caller_info = include_caller_info
        self.sensitive_fields = sensitive_fields or []

    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available
        if hasattr(record, "correlation_id") and record.correlation_id:
            log_entry["correlation_id"] = record.correlation_id

        # Add caller information if enabled
        if self.include_caller_info:
            log_entry["caller"] = {
                "filename": record.filename,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "correlation_id",
            ]:
                log_entry[key] = self._sanitize_value(value)

        return json.dumps(log_entry, default=str)

    def _sanitize_value(self, value):
        """Sanitize sensitive information from log values."""
        if isinstance(value, dict):
            return {
                k: "[REDACTED]" if k.lower() in self.sensitive_fields else v
                for k, v in value.items()
            }
        elif isinstance(value, str) and any(
            field in value.lower() for field in self.sensitive_fields
        ):
            return "[REDACTED]"
        return value


class IPELogger:
    """
    Central logging system for IPE platform with structured logging capabilities.
    """

    def __init__(self, config: Optional[LogConfig] = None):
        self.config = config or LogConfig()
        self.loggers = {}
        self._setup_logging()

    def _setup_logging(self):
        """Setup the logging configuration."""
        # Create log directory if it doesn't exist
        if self.config.log_to_file:
            log_dir = os.path.dirname(self.config.log_file_path)
            os.makedirs(log_dir, exist_ok=True)

        # Configure root logger
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": JSONFormatter,
                    "include_caller_info": self.config.include_caller_info,
                    "sensitive_fields": self.config.sensitive_fields,
                },
                "text": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "filters": {
                "correlation_id": {
                    "()": CorrelationIDFilter,
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.config.level,
                    "formatter": self.config.format_type,
                    "filters": ["correlation_id"],
                    "stream": sys.stdout,
                }
            },
            "loggers": {
                "ipe": {
                    "level": self.config.level,
                    "handlers": ["console"],
                    "propagate": False,
                }
            },
            "root": {"level": self.config.level, "handlers": ["console"]},
        }

        # Add file handler if enabled
        if self.config.log_to_file:
            logging_config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": self.config.level,
                "formatter": self.config.format_type,
                "filters": ["correlation_id"],
                "filename": self.config.log_file_path,
                "maxBytes": self.config.log_file_max_bytes,
                "backupCount": self.config.log_file_backup_count,
            }
            logging_config["loggers"]["ipe"]["handlers"].append("file")
            logging_config["root"]["handlers"].append("file")

        # Remove console handler if disabled
        if not self.config.enable_console_output:
            logging_config["loggers"]["ipe"]["handlers"].remove("console")
            logging_config["root"]["handlers"].remove("console")

        logging.config.dictConfig(logging_config)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for the specified name."""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(f"ipe.{name}")
        return self.loggers[name]

    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """Context manager for correlation ID."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        old_correlation_id = getattr(_context, "correlation_id", None)
        _context.correlation_id = correlation_id

        try:
            yield correlation_id
        finally:
            _context.correlation_id = old_correlation_id

    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current thread."""
        _context.correlation_id = correlation_id

    def get_correlation_id(self) -> Optional[str]:
        """Get correlation ID for current thread."""
        return getattr(_context, "correlation_id", None)


class SimulationLogger:
    """Specialized logger for simulation events."""

    def __init__(self, ipe_logger: IPELogger):
        self.logger = ipe_logger.get_logger("simulation")
        self.ipe_logger = ipe_logger

    def simulation_started(self, simulation_id: str, config: Dict[str, Any]):
        """Log simulation start event."""
        self.logger.info(
            "Simulation started",
            extra={
                "event_type": "simulation_started",
                "simulation_id": simulation_id,
                "config": config,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def simulation_completed(
        self, simulation_id: str, duration: float, results: Dict[str, Any]
    ):
        """Log simulation completion event."""
        self.logger.info(
            "Simulation completed",
            extra={
                "event_type": "simulation_completed",
                "simulation_id": simulation_id,
                "duration_seconds": duration,
                "results": results,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def simulation_failed(self, simulation_id: str, error: Exception, duration: float):
        """Log simulation failure event."""
        self.logger.error(
            "Simulation failed",
            extra={
                "event_type": "simulation_failed",
                "simulation_id": simulation_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "duration_seconds": duration,
                "timestamp": datetime.utcnow().isoformat(),
            },
            exc_info=True,
        )

    def generation_completed(
        self,
        simulation_id: str,
        generation: int,
        population_size: int,
        avg_fitness: float,
    ):
        """Log generation completion."""
        self.logger.debug(
            "Generation completed",
            extra={
                "event_type": "generation_completed",
                "simulation_id": simulation_id,
                "generation": generation,
                "population_size": population_size,
                "average_fitness": avg_fitness,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def mutation_event(
        self,
        simulation_id: str,
        organism_id: str,
        mutation_type: str,
        effect: Dict[str, Any],
    ):
        """Log mutation event."""
        self.logger.debug(
            "Mutation occurred",
            extra={
                "event_type": "mutation",
                "simulation_id": simulation_id,
                "organism_id": organism_id,
                "mutation_type": mutation_type,
                "effect": effect,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


class APILogger:
    """Specialized logger for API events."""

    def __init__(self, ipe_logger: IPELogger):
        self.logger = ipe_logger.get_logger("api")
        self.ipe_logger = ipe_logger

    def request_received(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        request_id: str,
        user_id: Optional[str] = None,
    ):
        """Log incoming API request."""
        self.logger.info(
            "API request received",
            extra={
                "event_type": "api_request_received",
                "method": method,
                "path": path,
                "request_id": request_id,
                "user_id": user_id,
                "user_agent": headers.get("User-Agent"),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def request_completed(
        self, request_id: str, status_code: int, duration: float, response_size: int
    ):
        """Log API request completion."""
        log_level = logging.INFO if status_code < 400 else logging.WARNING
        self.logger.log(
            log_level,
            "API request completed",
            extra={
                "event_type": "api_request_completed",
                "request_id": request_id,
                "status_code": status_code,
                "duration_seconds": duration,
                "response_size_bytes": response_size,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def authentication_event(
        self, event_type: str, user_id: Optional[str], success: bool, ip_address: str
    ):
        """Log authentication events."""
        log_level = logging.INFO if success else logging.WARNING
        self.logger.log(
            log_level,
            f"Authentication {event_type}",
            extra={
                "event_type": f"auth_{event_type}",
                "user_id": user_id,
                "success": success,
                "ip_address": ip_address,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def rate_limit_exceeded(self, ip_address: str, endpoint: str, limit: int):
        """Log rate limiting events."""
        self.logger.warning(
            "Rate limit exceeded",
            extra={
                "event_type": "rate_limit_exceeded",
                "ip_address": ip_address,
                "endpoint": endpoint,
                "limit": limit,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


class DatabaseLogger:
    """Specialized logger for database events."""

    def __init__(self, ipe_logger: IPELogger):
        self.logger = ipe_logger.get_logger("database")
        self.ipe_logger = ipe_logger

    def query_executed(
        self,
        query_type: str,
        table: str,
        duration: float,
        rows_affected: Optional[int] = None,
    ):
        """Log database query execution."""
        self.logger.debug(
            "Database query executed",
            extra={
                "event_type": "db_query_executed",
                "query_type": query_type,
                "table": table,
                "duration_seconds": duration,
                "rows_affected": rows_affected,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def migration_event(
        self, migration_name: str, direction: str, success: bool, duration: float
    ):
        """Log database migration events."""
        log_level = logging.INFO if success else logging.ERROR
        self.logger.log(
            log_level,
            f"Database migration {direction}",
            extra={
                "event_type": "db_migration",
                "migration_name": migration_name,
                "direction": direction,
                "success": success,
                "duration_seconds": duration,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def backup_event(
        self, backup_type: str, success: bool, file_size: Optional[int] = None
    ):
        """Log database backup events."""
        log_level = logging.INFO if success else logging.ERROR
        self.logger.log(
            log_level,
            f"Database backup {backup_type}",
            extra={
                "event_type": "db_backup",
                "backup_type": backup_type,
                "success": success,
                "file_size_bytes": file_size,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


class SecurityLogger:
    """Specialized logger for security events."""

    def __init__(self, ipe_logger: IPELogger):
        self.logger = ipe_logger.get_logger("security")
        self.ipe_logger = ipe_logger

    def suspicious_activity(
        self, activity_type: str, ip_address: str, details: Dict[str, Any]
    ):
        """Log suspicious security activity."""
        self.logger.warning(
            f"Suspicious activity detected: {activity_type}",
            extra={
                "event_type": "security_suspicious_activity",
                "activity_type": activity_type,
                "ip_address": ip_address,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def security_violation(
        self,
        violation_type: str,
        user_id: Optional[str],
        ip_address: str,
        details: Dict[str, Any],
    ):
        """Log security violations."""
        self.logger.error(
            f"Security violation: {violation_type}",
            extra={
                "event_type": "security_violation",
                "violation_type": violation_type,
                "user_id": user_id,
                "ip_address": ip_address,
                "details": details,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


# Global logger instance
_global_logger = None


def setup_logging(config: Optional[LogConfig] = None) -> IPELogger:
    """Setup and configure the global logging system."""
    global _global_logger
    _global_logger = IPELogger(config)
    return _global_logger


def get_logger(name: str = "main") -> logging.Logger:
    """Get a logger instance."""
    if _global_logger is None:
        setup_logging()
    return _global_logger.get_logger(name)


def get_simulation_logger() -> SimulationLogger:
    """Get the simulation logger."""
    if _global_logger is None:
        setup_logging()
    return SimulationLogger(_global_logger)


def get_api_logger() -> APILogger:
    """Get the API logger."""
    if _global_logger is None:
        setup_logging()
    return APILogger(_global_logger)


def get_database_logger() -> DatabaseLogger:
    """Get the database logger."""
    if _global_logger is None:
        setup_logging()
    return DatabaseLogger(_global_logger)


def get_security_logger() -> SecurityLogger:
    """Get the security logger."""
    if _global_logger is None:
        setup_logging()
    return SecurityLogger(_global_logger)


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager for correlation ID."""
    if _global_logger is None:
        setup_logging()
    with _global_logger.correlation_context(correlation_id) as cid:
        yield cid
