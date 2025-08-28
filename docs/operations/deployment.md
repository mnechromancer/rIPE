# IPE Deployment Guide

## Overview

This guide covers production deployment of the Interactionist Phylogeny Engine (IPE) across different environments and platforms.

## Deployment Architecture

### Production Environment Stack

```
┌─────────────────────────────────────────────────────┐
│                 Load Balancer                       │
│              (Nginx/HAProxy)                        │
├─────────────────────────────────────────────────────┤
│                API Gateway                          │
│          (Multiple FastAPI instances)              │
├─────────────────────────────────────────────────────┤
│              Application Layer                      │
│  ┌─────────────┬─────────────┬─────────────────────┐ │
│  │ Simulation  │ Background  │ WebSocket           │ │
│  │ Services    │ Workers     │ Services            │ │
│  └─────────────┴─────────────┴─────────────────────┘ │
├─────────────────────────────────────────────────────┤
│               Data Layer                            │
│  ┌──────────────────────┬──────────────────────────┐ │
│  │   PostgreSQL HA      │    Redis Cluster         │ │
│  │  (Primary/Replica)   │   (Cache/Sessions)       │ │
│  └──────────────────────┴──────────────────────────┘ │
├─────────────────────────────────────────────────────┤
│             Infrastructure                          │
│   Docker Swarm / Kubernetes / Cloud Services       │
└─────────────────────────────────────────────────────┘
```

## Environment Types

### 1. Development Environment

**Purpose:** Local development and testing

**Configuration:**
```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  ipe-api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - /app/__pycache__
    ports:
      - "8000:8000"
    environment:
      - DEBUG=true
      - DATABASE_URL=postgresql://ipe:password@postgres:5432/ipe_dev
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=ipe_dev
      - POSTGRES_USER=ipe
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_dev:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_dev:
```

**Setup Commands:**
```bash
# Development setup
git clone https://github.com/mnechromancer/RIPE.git
cd RIPE
cp .env.example .env.dev
docker-compose -f docker-compose.dev.yml up -d

# Run migrations
docker-compose -f docker-compose.dev.yml exec ipe-api alembic upgrade head

# Install development tools
pip install -r requirements-dev.txt
pre-commit install
```

### 2. Staging Environment

**Purpose:** Pre-production testing and validation

**Configuration:**
```yaml
# docker-compose.staging.yml
version: '3.8'
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/staging.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - ipe-api

  ipe-api:
    image: ipe:staging
    replicas: 2
    environment:
      - ENVIRONMENT=staging
      - DATABASE_URL=postgresql://ipe:${DB_PASSWORD}@postgres:5432/ipe_staging
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=ipe_staging
      - POSTGRES_USER=ipe
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_staging:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_staging:/data

volumes:
  postgres_staging:
  redis_staging:
```

### 3. Production Environment

**Purpose:** Live production system

**Configuration:**
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/production.conf:/etc/nginx/nginx.conf
      - /etc/letsencrypt:/etc/letsencrypt:ro
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.role == manager

  ipe-api:
    image: ipe:${VERSION}
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      placement:
        constraints:
          - node.role == worker
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://ipe:${DB_PASSWORD}@postgres-primary:5432/ipe
      - REDIS_URL=redis://redis-cluster:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - JWT_SECRET=${JWT_SECRET}
    secrets:
      - db_password
      - jwt_secret

  postgres-primary:
    image: postgres:14
    environment:
      - POSTGRES_DB=ipe
      - POSTGRES_USER=ipe
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
      - POSTGRES_REPLICATION_USER=replica
      - POSTGRES_REPLICATION_PASSWORD_FILE=/run/secrets/replica_password
    volumes:
      - postgres_primary:/var/lib/postgresql/data
      - ./postgres/postgresql.conf:/etc/postgresql/postgresql.conf
    secrets:
      - db_password
      - replica_password

secrets:
  db_password:
    external: true
  jwt_secret:
    external: true
  replica_password:
    external: true

volumes:
  postgres_primary:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/postgres
```

## Cloud Platform Deployments

### AWS Deployment

#### ECS with Fargate

```yaml
# aws-ecs-task-definition.json
{
  "family": "ipe-production",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "ipe-api",
      "image": "YOUR_ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/ipe:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:ACCOUNT:secret:ipe/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ipe-production",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Terraform Configuration

```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# ECS Cluster
resource "aws_ecs_cluster" "ipe_cluster" {
  name = "ipe-production"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Application Load Balancer
resource "aws_lb" "ipe_alb" {
  name               = "ipe-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets           = var.public_subnet_ids

  enable_deletion_protection = true
}

# RDS PostgreSQL
resource "aws_db_instance" "ipe_database" {
  identifier = "ipe-production"
  
  engine         = "postgres"
  engine_version = "14.7"
  instance_class = "db.r6g.xlarge"
  
  allocated_storage     = 500
  max_allocated_storage = 2000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "ipe"
  username = "ipe_admin"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.ipe_subnet_group.name
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "ipe-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  
  tags = {
    Name = "IPE Production Database"
  }
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "ipe_redis" {
  replication_group_id = "ipe-redis"
  description          = "Redis cluster for IPE"
  
  engine               = "redis"
  node_type           = "cache.r6g.large"
  parameter_group_name = "default.redis6.x"
  
  num_cache_clusters = 2
  
  subnet_group_name  = aws_elasticache_subnet_group.ipe_cache_subnet.name
  security_group_ids = [aws_security_group.redis_sg.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name = "IPE Production Redis"
  }
}
```

### Google Cloud Platform (GCP)

#### GKE Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ipe-api
  labels:
    app: ipe-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ipe-api
  template:
    metadata:
      labels:
        app: ipe-api
    spec:
      containers:
      - name: ipe-api
        image: gcr.io/PROJECT_ID/ipe:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ipe-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ipe-secrets
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ipe-api-service
spec:
  selector:
    app: ipe-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### Azure Deployment

#### Container Instances

```yaml
# azure-container-instances.yaml
apiVersion: 2019-12-01
location: eastus
name: ipe-production
properties:
  containers:
  - name: ipe-api
    properties:
      image: iperegistry.azurecr.io/ipe:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 4
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: ENVIRONMENT
        value: production
      - name: DATABASE_URL
        secureValue: postgresql://ipe:PASSWORD@ipe-postgres.postgres.database.azure.com:5432/ipe
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
tags: 
  environment: production
  application: ipe
type: Microsoft.ContainerInstance/containerGroups
```

## Database Deployment

### PostgreSQL High Availability

#### Master-Slave Setup

```bash
# Primary server configuration (postgresql.conf)
listen_addresses = '*'
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3
synchronous_commit = on
synchronous_standby_names = 'standby1,standby2'

# Create replication user
sudo -u postgres psql
CREATE USER replica REPLICATION LOGIN ENCRYPTED PASSWORD 'replica_password';
```

```bash
# Standby server setup
# Stop PostgreSQL on standby
sudo systemctl stop postgresql

# Take base backup from primary
sudo -u postgres pg_basebackup -h PRIMARY_IP -D /var/lib/postgresql/14/main -U replica -P -v -R -X stream -C -S standby1

# Start standby
sudo systemctl start postgresql
```

#### Connection Pooling with PgBouncer

```ini
# /etc/pgbouncer/pgbouncer.ini
[databases]
ipe = host=localhost port=5432 dbname=ipe

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
admin_users = postgres
stats_users = stats, postgres

pool_mode = transaction
server_reset_query = DISCARD ALL
max_client_conn = 1000
default_pool_size = 100
min_pool_size = 10
reserve_pool_size = 10
reserve_pool_timeout = 5
max_db_connections = 50
```

### Redis Configuration

#### Redis Cluster Setup

```bash
# Redis cluster configuration (redis.conf)
port 7000
cluster-enabled yes
cluster-config-file nodes-7000.conf
cluster-node-timeout 5000
appendonly yes
```

```bash
# Create cluster
redis-cli --cluster create \
  127.0.0.1:7000 127.0.0.1:7001 \
  127.0.0.1:7002 127.0.0.1:7003 \
  127.0.0.1:7004 127.0.0.1:7005 \
  --cluster-replicas 1
```

## Load Balancing and SSL

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/ipe-production
upstream ipe_backend {
    least_conn;
    server 10.0.1.10:8000 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8000 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8000 max_fails=3 fail_timeout=30s;
    server 10.0.1.13:8000 max_fails=3 fail_timeout=30s;
}

# WebSocket upstream
upstream ipe_websocket {
    least_conn;
    server 10.0.1.10:8000;
    server 10.0.1.11:8000;
    server 10.0.1.12:8000;
    server 10.0.1.13:8000;
}

server {
    listen 80;
    server_name ipe.velottalab.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ipe.velottalab.com;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/ipe.velottalab.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ipe.velottalab.com/privkey.pem;
    ssl_session_timeout 1d;
    ssl_session_cache shared:MozTLS:10m;
    ssl_session_tickets off;

    # Intermediate configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    # API routes
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://ipe_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # WebSocket routes
    location /ws/ {
        proxy_pass http://ipe_websocket;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Static files
    location /static/ {
        alias /var/www/ipe/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Documentation
    location /docs/ {
        alias /var/www/ipe/docs/;
        index index.html;
        try_files $uri $uri/ =404;
    }
}
```

### HAProxy Configuration (Alternative)

```
# /etc/haproxy/haproxy.cfg
global
    daemon
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy

    # SSL
    ssl-default-bind-ciphers ECDH+AESGCM:DH+AESGCM:ECDH+AES256:DH+AES256:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP
    ssl-default-bind-options ssl-min-ver TLSv1.2 no-tls-tickets

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull
    option redispatch
    retries 3
    maxconn 2000

frontend ipe_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/ipe.pem
    redirect scheme https if !{ ssl_fc }
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request reject if { sc_http_req_rate(0) gt 20 }
    
    # Route to backend
    default_backend ipe_backend

backend ipe_backend
    balance roundrobin
    option httpchk GET /health
    server api1 10.0.1.10:8000 check
    server api2 10.0.1.11:8000 check
    server api3 10.0.1.12:8000 check
    server api4 10.0.1.13:8000 check

# Statistics
listen stats
    bind *:8404
    stats enable
    stats uri /stats
    stats refresh 30s
    stats admin if LOCALHOST
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "ipe_rules.yml"

alertmanager:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'ipe-api'
    static_configs:
      - targets: 
        - 'ipe-api-1:8000'
        - 'ipe-api-2:8000'
        - 'ipe-api-3:8000'
        - 'ipe-api-4:8000'
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

  - job_name: 'node'
    static_configs:
      - targets:
        - 'node1:9100'
        - 'node2:9100'
        - 'node3:9100'
```

### Alert Rules

```yaml
# ipe_rules.yml
groups:
- name: ipe_alerts
  rules:
  - alert: IPE_API_Down
    expr: up{job="ipe-api"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "IPE API instance is down"
      description: "IPE API instance {{ $labels.instance }} has been down for more than 1 minute."

  - alert: High_Response_Time
    expr: histogram_quantile(0.95, http_request_duration_seconds_bucket{job="ipe-api"}) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time on IPE API"
      description: "95th percentile latency is {{ $value }}s"

  - alert: High_Memory_Usage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is above 90% on {{ $labels.instance }}"

  - alert: Database_Connection_High
    expr: postgres_stat_activity_count > 90
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High number of database connections"
      description: "Database has {{ $value }} active connections"
```

### ELK Stack Configuration

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.7.1
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.7.1
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
      - ./logstash/config:/usr/share/logstash/config
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.7.1
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "IPE Production Dashboard",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket{job=\"ipe-api\"})",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, http_request_duration_seconds_bucket{job=\"ipe-api\"})",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Active Simulations",
        "type": "singlestat",
        "targets": [
          {
            "expr": "ipe_active_simulations_total",
            "legendFormat": "Active"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"ipe-api\"}[5m])",
            "legendFormat": "{{ method }} {{ status }}"
          }
        ]
      }
    ]
  }
}
```

## Backup and Recovery

### Database Backup Strategy

```bash
#!/bin/bash
# backup-database.sh

DB_HOST="localhost"
DB_NAME="ipe"
DB_USER="ipe"
BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/ipe_backup_${DATE}.sql"

# Create backup directory
mkdir -p ${BACKUP_DIR}

# Full backup
pg_dump -h ${DB_HOST} -U ${DB_USER} -d ${DB_NAME} -v -f ${BACKUP_FILE}

# Compress backup
gzip ${BACKUP_FILE}

# Upload to S3 (optional)
aws s3 cp ${BACKUP_FILE}.gz s3://ipe-backups/postgres/

# Clean up old backups (keep 30 days)
find ${BACKUP_DIR} -name "ipe_backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

### Application Data Backup

```bash
#!/bin/bash
# backup-application.sh

BACKUP_DIR="/backups/application"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/var/lib/ipe"

# Create backup
tar -czf ${BACKUP_DIR}/ipe_app_${DATE}.tar.gz \
  --exclude='*.log' \
  --exclude='tmp/*' \
  --exclude='cache/*' \
  ${APP_DIR}

# Upload to cloud storage
aws s3 cp ${BACKUP_DIR}/ipe_app_${DATE}.tar.gz s3://ipe-backups/application/

echo "Application backup completed"
```

### Disaster Recovery Plan

```bash
#!/bin/bash
# disaster-recovery.sh

# 1. Restore database
echo "Restoring database..."
psql -h ${NEW_DB_HOST} -U ${DB_USER} -d template1 -c "DROP DATABASE IF EXISTS ipe;"
psql -h ${NEW_DB_HOST} -U ${DB_USER} -d template1 -c "CREATE DATABASE ipe;"
gunzip -c ${LATEST_BACKUP} | psql -h ${NEW_DB_HOST} -U ${DB_USER} -d ipe

# 2. Update configuration
echo "Updating configuration..."
sed -i "s/DATABASE_URL=.*/DATABASE_URL=${NEW_DATABASE_URL}/" .env

# 3. Restore application data
echo "Restoring application data..."
tar -xzf ${APP_BACKUP} -C /

# 4. Start services
echo "Starting services..."
docker-compose up -d

# 5. Verify health
echo "Checking health..."
curl -f http://localhost:8000/health || exit 1

echo "Disaster recovery completed successfully"
```

This comprehensive deployment guide provides the foundation for running IPE in production environments with high availability, monitoring, and disaster recovery capabilities.