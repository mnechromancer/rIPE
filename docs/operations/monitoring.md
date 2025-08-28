# IPE Monitoring and Observability

## Overview

This document covers comprehensive monitoring, observability, and alerting strategies for the IPE platform in production environments.

## Monitoring Stack Architecture

```
┌─────────────────────────────────────────────────────┐
│                 Alerting Layer                      │
│         AlertManager + PagerDuty + Slack           │
├─────────────────────────────────────────────────────┤
│              Visualization Layer                    │
│              Grafana Dashboards                     │
├─────────────────────────────────────────────────────┤
│               Metrics Layer                         │
│    Prometheus + Custom Metrics + Exporters         │
├─────────────────────────────────────────────────────┤
│                Logs Layer                           │
│          ELK Stack / Loki + Promtail               │
├─────────────────────────────────────────────────────┤
│               Tracing Layer                         │
│              Jaeger + OpenTelemetry                │
├─────────────────────────────────────────────────────┤
│              Application Layer                      │
│         IPE Services + Instrumentation             │
└─────────────────────────────────────────────────────┘
```

## Key Performance Indicators (KPIs)

### Business Metrics
- **Simulation Success Rate**: Percentage of simulations completing successfully
- **User Engagement**: Active users per day/week/month
- **Research Output**: Publications and citations using IPE
- **Data Export Volume**: Amount of data exported by users

### Technical Metrics
- **API Response Time**: P50, P95, P99 latencies
- **Throughput**: Requests per second
- **Error Rate**: Percentage of failed requests
- **System Uptime**: Availability percentage (target: 99.9%)

### Resource Metrics
- **CPU Utilization**: Per service and overall
- **Memory Usage**: Current usage and growth trends
- **Disk I/O**: Read/write operations and latency
- **Network Traffic**: Ingress/egress bandwidth

## Metrics Collection

### Application Metrics

#### Custom IPE Metrics

```python
# ipe/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Enum

# Simulation metrics
SIMULATIONS_TOTAL = Counter(
    'ipe_simulations_created_total',
    'Total number of simulations created',
    ['user_id', 'simulation_type']
)

SIMULATION_DURATION = Histogram(
    'ipe_simulation_duration_seconds',
    'Time taken to complete simulations',
    ['simulation_type', 'success'],
    buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200, float('inf')]
)

ACTIVE_SIMULATIONS = Gauge(
    'ipe_active_simulations',
    'Number of currently running simulations'
)

SIMULATION_STATUS = Enum(
    'ipe_simulation_status',
    'Current status of simulations',
    ['simulation_id'],
    states=['created', 'running', 'completed', 'failed', 'cancelled']
)

# API metrics
API_REQUESTS_TOTAL = Counter(
    'ipe_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

API_REQUEST_DURATION = Histogram(
    'ipe_api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint'],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
)

# Database metrics
DB_CONNECTION_POOL = Gauge(
    'ipe_db_connections',
    'Database connection pool status',
    ['pool_name', 'status']
)

DB_QUERY_DURATION = Histogram(
    'ipe_db_query_duration_seconds',
    'Database query execution time',
    ['query_type', 'table'],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, float('inf')]
)

# Resource allocation metrics
RESOURCE_ALLOCATION = Gauge(
    'ipe_resource_allocation',
    'Current resource allocation',
    ['resource_type', 'simulation_id']
)

# Export metrics
EXPORTS_TOTAL = Counter(
    'ipe_exports_total',
    'Total data exports',
    ['format', 'data_type', 'user_id']
)

EXPORT_SIZE_BYTES = Histogram(
    'ipe_export_size_bytes',
    'Size of exported data',
    ['format'],
    buckets=[1024, 10240, 102400, 1024000, 10240000, 102400000, float('inf')]
)
```

#### Instrumentation in Application Code

```python
# ipe/api/routes/simulations.py
import time
from ipe.monitoring.metrics import (
    SIMULATIONS_TOTAL, SIMULATION_DURATION, 
    API_REQUESTS_TOTAL, API_REQUEST_DURATION
)

@router.post("", response_model=SimulationResponse)
async def create_simulation(params: SimulationParams) -> SimulationResponse:
    start_time = time.time()
    
    try:
        # Create simulation logic
        simulation = await simulation_service.create(params)
        
        # Record metrics
        SIMULATIONS_TOTAL.labels(
            user_id=current_user.id,
            simulation_type=params.simulation_type
        ).inc()
        
        API_REQUESTS_TOTAL.labels(
            method="POST",
            endpoint="/simulations",
            status="201"
        ).inc()
        
        return simulation
        
    except Exception as e:
        API_REQUESTS_TOTAL.labels(
            method="POST",
            endpoint="/simulations", 
            status="500"
        ).inc()
        raise
        
    finally:
        duration = time.time() - start_time
        API_REQUEST_DURATION.labels(
            method="POST",
            endpoint="/simulations"
        ).observe(duration)
```

### Infrastructure Metrics

#### Prometheus Configuration

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'ipe-production'
    environment: 'production'

rule_files:
  - "alert_rules.yml"
  - "recording_rules.yml"

alertmanager:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # IPE Application
  - job_name: 'ipe-api'
    static_configs:
      - targets: 
        - 'ipe-api-1:8000'
        - 'ipe-api-2:8000'
        - 'ipe-api-3:8000'
    metrics_path: /metrics
    scrape_interval: 10s
    scrape_timeout: 5s

  # PostgreSQL
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 15s

  # Redis
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 15s

  # System metrics
  - job_name: 'node'
    static_configs:
      - targets:
        - 'node1:9100'
        - 'node2:9100'
        - 'node3:9100'
    scrape_interval: 15s

  # Nginx/Load Balancer
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
    scrape_interval: 15s

  # Docker/Container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
    scrape_interval: 15s
```

#### Recording Rules

```yaml
# prometheus/recording_rules.yml
groups:
- name: ipe_recording_rules
  interval: 30s
  rules:
  # API performance aggregations
  - record: ipe:api_request_rate_5m
    expr: rate(ipe_api_requests_total[5m])
    
  - record: ipe:api_error_rate_5m
    expr: rate(ipe_api_requests_total{status=~"5.."}[5m]) / rate(ipe_api_requests_total[5m])
    
  - record: ipe:api_latency_95p
    expr: histogram_quantile(0.95, rate(ipe_api_request_duration_seconds_bucket[5m]))
    
  # Simulation performance
  - record: ipe:simulation_completion_rate_1h
    expr: rate(ipe_simulations_total{status="completed"}[1h])
    
  - record: ipe:simulation_failure_rate_1h
    expr: rate(ipe_simulations_total{status="failed"}[1h]) / rate(ipe_simulations_total[1h])
    
  # Resource utilization
  - record: ipe:cpu_utilization_avg
    expr: 1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m]))
    
  - record: ipe:memory_utilization
    expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100
    
  # Database performance
  - record: ipe:db_connection_utilization
    expr: postgres_stat_activity_count / postgres_settings_max_connections * 100
```

## Logging Strategy

### Structured Logging

```python
# ipe/utils/logging.py
import logging
import structlog
from pythonjsonlogger import jsonlogger

def configure_logging():
    """Configure structured logging for IPE"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False
    )
    
    # Configure standard logging
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Usage in application code
logger = structlog.get_logger(__name__)

async def create_simulation(params: SimulationParams):
    logger.info(
        "simulation.create.started",
        simulation_name=params.name,
        duration=params.duration,
        population_size=params.population_size,
        user_id=current_user.id
    )
    
    try:
        simulation = await simulation_service.create(params)
        
        logger.info(
            "simulation.create.completed",
            simulation_id=simulation.id,
            duration_seconds=time.time() - start_time
        )
        
        return simulation
        
    except Exception as e:
        logger.error(
            "simulation.create.failed",
            error=str(e),
            error_type=type(e).__name__,
            simulation_params=params.dict()
        )
        raise
```

### Log Aggregation with ELK Stack

```yaml
# logstash/pipeline/ipe-logs.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "ipe-api" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [level] in ["ERROR", "CRITICAL"] {
      mutate {
        add_tag => ["alert_worthy"]
      }
    }
    
    # Extract simulation_id for correlation
    if [simulation_id] {
      mutate {
        add_field => { "trace_id" => "%{simulation_id}" }
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "ipe-logs-%{+YYYY.MM.dd}"
  }
  
  # Send alerts to monitoring system
  if "alert_worthy" in [tags] {
    http {
      url => "http://alertmanager:9093/api/v1/alerts"
      http_method => "post"
      content_type => "application/json"
      format => "json"
    }
  }
}
```

## Alerting System

### Alert Rules

```yaml
# prometheus/alert_rules.yml
groups:
- name: ipe_alerts
  rules:
  
  # Service availability alerts
  - alert: IPE_API_Down
    expr: up{job="ipe-api"} == 0
    for: 1m
    labels:
      severity: critical
      service: ipe-api
    annotations:
      summary: "IPE API instance is down"
      description: "IPE API instance {{ $labels.instance }} has been down for more than 1 minute."
      runbook_url: "https://docs.ipe.velottalab.com/runbooks/api-down"

  - alert: IPE_High_Error_Rate
    expr: ipe:api_error_rate_5m > 0.05
    for: 5m
    labels:
      severity: warning
      service: ipe-api
    annotations:
      summary: "High error rate on IPE API"
      description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"

  - alert: IPE_High_Latency
    expr: ipe:api_latency_95p > 5
    for: 5m
    labels:
      severity: warning
      service: ipe-api
    annotations:
      summary: "High latency on IPE API"
      description: "95th percentile latency is {{ $value }}s for the last 5 minutes"

  # Database alerts
  - alert: PostgreSQL_Down
    expr: up{job="postgres"} == 0
    for: 1m
    labels:
      severity: critical
      service: postgresql
    annotations:
      summary: "PostgreSQL is down"
      description: "PostgreSQL instance {{ $labels.instance }} is down"

  - alert: High_Database_Connections
    expr: ipe:db_connection_utilization > 80
    for: 2m
    labels:
      severity: warning
      service: postgresql
    annotations:
      summary: "High database connection usage"
      description: "Database connection usage is {{ $value }}%"

  - alert: Long_Running_Queries
    expr: postgres_stat_activity_max_tx_duration > 300
    for: 1m
    labels:
      severity: warning
      service: postgresql
    annotations:
      summary: "Long running database queries detected"
      description: "Longest running transaction is {{ $value }}s"

  # System resource alerts
  - alert: High_CPU_Usage
    expr: ipe:cpu_utilization_avg > 0.8
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is {{ $value | humanizePercentage }} averaged over 5 minutes"

  - alert: High_Memory_Usage
    expr: ipe:memory_utilization > 90
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}% on {{ $labels.instance }}"

  - alert: Low_Disk_Space
    expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 10
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Low disk space"
      description: "Disk space is {{ $value }}% available on {{ $labels.instance }}:{{ $labels.mountpoint }}"

  # Simulation-specific alerts
  - alert: High_Simulation_Failure_Rate
    expr: ipe:simulation_failure_rate_1h > 0.1
    for: 10m
    labels:
      severity: warning
      service: simulation-engine
    annotations:
      summary: "High simulation failure rate"
      description: "Simulation failure rate is {{ $value | humanizePercentage }} over the last hour"

  - alert: Simulation_Queue_Backlog
    expr: ipe_active_simulations > 100
    for: 15m
    labels:
      severity: warning
      service: simulation-engine
    annotations:
      summary: "Large simulation queue backlog"
      description: "{{ $value }} simulations are currently queued or running"

  # Redis alerts
  - alert: Redis_Down
    expr: up{job="redis"} == 0
    for: 1m
    labels:
      severity: critical
      service: redis
    annotations:
      summary: "Redis is down"
      description: "Redis instance {{ $labels.instance }} is down"

  - alert: Redis_High_Memory_Usage
    expr: redis_memory_used_bytes / redis_memory_max_bytes * 100 > 90
    for: 5m
    labels:
      severity: warning
      service: redis
    annotations:
      summary: "Redis high memory usage"
      description: "Redis memory usage is {{ $value }}%"
```

### AlertManager Configuration

```yaml
# alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@ipe.velottalab.com'
  smtp_auth_username: 'alerts@ipe.velottalab.com'
  smtp_auth_password: 'app_password'

route:
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
    continue: true
  - match:
      service: ipe-api
    receiver: 'api-team'
  - match:
      service: postgresql
    receiver: 'database-team'

receivers:
- name: 'default'
  email_configs:
  - to: 'ops@ipe.velottalab.com'
    subject: 'IPE Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}

- name: 'critical-alerts'
  pagerduty_configs:
  - service_key: 'YOUR_PAGERDUTY_SERVICE_KEY'
    description: 'Critical IPE Alert'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#ipe-critical'
    title: 'Critical IPE Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'api-team'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#ipe-api-alerts'
    title: 'IPE API Alert'

- name: 'database-team'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#ipe-database-alerts'
    title: 'IPE Database Alert'

inhibit_rules:
- source_match:
    severity: 'critical'
  target_match:
    severity: 'warning'
  equal: ['alertname', 'instance']
```

## Distributed Tracing

### OpenTelemetry Configuration

```python
# ipe/tracing/setup.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing(app):
    """Setup distributed tracing for IPE"""
    
    # Configure tracer
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Setup Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=6831,
        service_name="ipe-api"
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)
    
    # Instrument database
    SQLAlchemyInstrumentor().instrument()
    
    # Instrument Redis
    RedisInstrumentor().instrument()
    
    return tracer

# Usage in application code
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

async def run_simulation(simulation_id: str):
    with tracer.start_as_current_span("simulation.run") as span:
        span.set_attribute("simulation.id", simulation_id)
        span.set_attribute("simulation.type", simulation.type)
        
        try:
            # Run simulation logic
            result = await simulation_engine.run(simulation_id)
            span.set_status(trace.Status(trace.StatusCode.OK))
            return result
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(
                trace.Status(
                    trace.StatusCode.ERROR,
                    description=str(e)
                )
            )
            raise
```

## Grafana Dashboards

### Main IPE Dashboard

```json
{
  "dashboard": {
    "title": "IPE Production Overview",
    "tags": ["ipe", "production"],
    "timezone": "UTC",
    "panels": [
      {
        "title": "System Health",
        "type": "stat",
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "up{job=\"ipe-api\"}",
            "legendFormat": "API Instances",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 0.5},
                {"color": "green", "value": 1}
              ]
            }
          }
        }
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
        "targets": [
          {
            "expr": "sum(rate(ipe_api_requests_total[5m])) by (endpoint)",
            "legendFormat": "{{ endpoint }}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(ipe_api_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, sum(rate(ipe_api_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Active Simulations",
        "type": "graph",
        "gridPos": {"h": 6, "w": 8, "x": 0, "y": 12},
        "targets": [
          {
            "expr": "ipe_active_simulations",
            "legendFormat": "Running"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "gridPos": {"h": 6, "w": 8, "x": 8, "y": 12},
        "targets": [
          {
            "expr": "sum(rate(ipe_api_requests_total{status=~\"5..\"}[5m])) / sum(rate(ipe_api_requests_total[5m]))",
            "legendFormat": "Error Rate"
          }
        ]
      },
      {
        "title": "Resource Usage",
        "type": "graph",
        "gridPos": {"h": 6, "w": 8, "x": 16, "y": 12},
        "targets": [
          {
            "expr": "ipe:cpu_utilization_avg * 100",
            "legendFormat": "CPU %"
          },
          {
            "expr": "ipe:memory_utilization",
            "legendFormat": "Memory %"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
```

### Database Performance Dashboard

```json
{
  "dashboard": {
    "title": "IPE Database Performance",
    "panels": [
      {
        "title": "Connection Pool Status",
        "type": "graph",
        "targets": [
          {
            "expr": "postgres_stat_activity_count",
            "legendFormat": "Active Connections"
          },
          {
            "expr": "postgres_settings_max_connections",
            "legendFormat": "Max Connections"
          }
        ]
      },
      {
        "title": "Query Performance",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(postgres_stat_user_tables_seq_scan[5m])",
            "legendFormat": "Sequential Scans"
          },
          {
            "expr": "rate(postgres_stat_user_tables_idx_scan[5m])",
            "legendFormat": "Index Scans"
          }
        ]
      },
      {
        "title": "Lock Statistics",
        "type": "graph",
        "targets": [
          {
            "expr": "postgres_locks_count",
            "legendFormat": "{{ mode }} locks"
          }
        ]
      }
    ]
  }
}
```

## Health Checks

### Application Health Endpoints

```python
# ipe/api/health.py
from fastapi import APIRouter, HTTPException
from sqlalchemy import text
import redis
import time

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """Detailed readiness check including dependencies"""
    
    checks = {
        "database": False,
        "redis": False,
        "simulation_engine": False
    }
    
    # Check database
    try:
        result = db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
    
    # Check Redis
    try:
        redis_client = redis.Redis(host='redis', port=6379, db=0)
        redis_client.ping()
        checks["redis"] = True
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
    
    # Check simulation engine
    try:
        # Basic simulation engine validation
        from ipe.simulation.engine import SimulationEngine
        engine = SimulationEngine()
        if engine.is_healthy():
            checks["simulation_engine"] = True
    except Exception as e:
        logger.error("Simulation engine health check failed", error=str(e))
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return Response(
        content={
            "status": "ready" if all_healthy else "not ready",
            "checks": checks,
            "timestamp": time.time()
        },
        status_code=status_code
    )

@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

### Deep Health Checks

```python
# ipe/monitoring/health.py
import asyncio
from typing import Dict, Any

async def comprehensive_health_check() -> Dict[str, Any]:
    """Comprehensive system health assessment"""
    
    health_status = {
        "overall": "healthy",
        "components": {},
        "timestamp": time.time()
    }
    
    # Check all components
    component_checks = [
        check_database_health(),
        check_redis_health(),
        check_simulation_capacity(),
        check_file_system(),
        check_external_services()
    ]
    
    results = await asyncio.gather(*component_checks, return_exceptions=True)
    
    for i, result in enumerate(results):
        component_name = ["database", "redis", "simulation", "filesystem", "external"][i]
        
        if isinstance(result, Exception):
            health_status["components"][component_name] = {
                "status": "unhealthy",
                "error": str(result)
            }
            health_status["overall"] = "degraded"
        else:
            health_status["components"][component_name] = result
            
            if result.get("status") != "healthy":
                health_status["overall"] = "degraded"
    
    return health_status

async def check_simulation_capacity():
    """Check if system can handle new simulations"""
    
    try:
        # Check CPU and memory availability
        cpu_usage = get_cpu_usage()
        memory_usage = get_memory_usage()
        active_sims = get_active_simulation_count()
        
        if cpu_usage > 90 or memory_usage > 90 or active_sims > 50:
            return {
                "status": "degraded",
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "active_simulations": active_sims,
                "can_accept_new": False
            }
        
        return {
            "status": "healthy",
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "active_simulations": active_sims,
            "can_accept_new": True
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

This comprehensive monitoring setup provides full observability into the IPE platform, enabling proactive issue detection and resolution while maintaining high availability and performance standards.