#!/bin/bash
# Database setup script for IPE (Integrated Phenotypic Evolution) platform
# Sets up PostgreSQL with TimescaleDB, runs migrations, and configures backups

set -euo pipefail

# Configuration
DB_NAME="${DB_NAME:-ipe_db}"
DB_USER="${DB_USER:-ipe_user}"
DB_PASSWORD="${DB_PASSWORD:-ipe_password}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
POSTGRES_VERSION="${POSTGRES_VERSION:-15}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/ipe}"
ENVIRONMENT="${ENVIRONMENT:-development}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if PostgreSQL is running
check_postgres() {
    if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" >/dev/null 2>&1; then
        log_error "PostgreSQL is not running or not accessible at $DB_HOST:$DB_PORT"
        exit 1
    fi
    log_success "PostgreSQL is running"
}

# Function to check if TimescaleDB is available
check_timescaledb() {
    local check_query="SELECT * FROM pg_extension WHERE extname = 'timescaledb';"
    if psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -d postgres -t -c "$check_query" | grep -q timescaledb; then
        log_success "TimescaleDB extension is available"
    else
        log_warning "TimescaleDB extension not found - installing..."
        install_timescaledb
    fi
}

# Function to install TimescaleDB (Ubuntu/Debian)
install_timescaledb() {
    if [[ "$EUID" -ne 0 ]]; then
        log_error "TimescaleDB installation requires root privileges"
        exit 1
    fi
    
    log_info "Installing TimescaleDB..."
    
    # Add TimescaleDB repository
    curl -fsSL https://packagecloud.io/timescale/timescaledb/gpgkey | apt-key add -
    echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main" > /etc/apt/sources.list.d/timescaledb.list
    
    # Update and install
    apt-get update
    apt-get install -y timescaledb-2-postgresql-$POSTGRES_VERSION
    
    # Configure TimescaleDB
    timescaledb-tune --quiet --yes
    
    # Restart PostgreSQL
    systemctl restart postgresql
    
    log_success "TimescaleDB installed successfully"
}

# Function to create database and user
setup_database() {
    log_info "Setting up database and user..."
    
    # Check if database exists
    if psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        log_warning "Database $DB_NAME already exists"
    else
        log_info "Creating database $DB_NAME..."
        psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "CREATE DATABASE $DB_NAME;"
        log_success "Database $DB_NAME created"
    fi
    
    # Check if user exists
    if psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -t -c "SELECT 1 FROM pg_roles WHERE rolname='$DB_USER'" | grep -q 1; then
        log_warning "User $DB_USER already exists"
    else
        log_info "Creating user $DB_USER..."
        psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "CREATE USER $DB_USER WITH PASSWORD '$DB_PASSWORD';"
        log_success "User $DB_USER created"
    fi
    
    # Grant privileges
    log_info "Granting privileges..."
    psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
    psql -h "$DB_HOST" -p "$DB_PORT" -U postgres -d "$DB_NAME" -c "GRANT ALL ON SCHEMA public TO $DB_USER;"
    
    log_success "Database setup completed"
}

# Function to enable TimescaleDB extension
enable_timescaledb() {
    log_info "Enabling TimescaleDB extension..."
    
    export PGPASSWORD="$DB_PASSWORD"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
    
    # Verify extension is enabled
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT * FROM pg_extension WHERE extname = 'timescaledb';" | grep -q timescaledb; then
        log_success "TimescaleDB extension enabled"
    else
        log_error "Failed to enable TimescaleDB extension"
        exit 1
    fi
}

# Function to run migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Set environment variables for Alembic
    export DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
    
    # Check if alembic is available
    if ! command -v alembic &> /dev/null; then
        log_info "Installing Alembic..."
        pip install alembic psycopg2-binary
    fi
    
    # Initialize Alembic if needed
    if [[ ! -f "alembic.ini" ]]; then
        log_error "alembic.ini not found. Please run this script from the project root directory."
        exit 1
    fi
    
    # Run migrations
    if alembic upgrade head; then
        log_success "Migrations completed successfully"
    else
        log_error "Migration failed"
        exit 1
    fi
}

# Function to create backup directory and setup backup strategy
setup_backups() {
    log_info "Setting up backup strategy..."
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    chown postgres:postgres "$BACKUP_DIR"
    
    # Create backup script
    cat > "$BACKUP_DIR/backup-ipe.sh" << EOF
#!/bin/bash
# Automated backup script for IPE database

BACKUP_DIR="$BACKUP_DIR"
DB_NAME="$DB_NAME"
DATE=\$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="\$BACKUP_DIR/ipe_backup_\$DATE.sql"

# Create backup
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d \$DB_NAME > \$BACKUP_FILE

# Compress backup
gzip \$BACKUP_FILE

# Remove backups older than 30 days
find \$BACKUP_DIR -name "ipe_backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: \$BACKUP_FILE.gz"
EOF
    
    chmod +x "$BACKUP_DIR/backup-ipe.sh"
    
    # Add to crontab for daily backups at 2 AM
    if ! crontab -l 2>/dev/null | grep -q "backup-ipe.sh"; then
        (crontab -l 2>/dev/null; echo "0 2 * * * $BACKUP_DIR/backup-ipe.sh") | crontab -
        log_success "Daily backup scheduled at 2 AM"
    else
        log_warning "Backup cron job already exists"
    fi
    
    log_success "Backup strategy configured"
}

# Function to optimize PostgreSQL configuration for TimescaleDB
optimize_postgres() {
    log_info "Optimizing PostgreSQL configuration for TimescaleDB..."
    
    local postgres_conf="/etc/postgresql/$POSTGRES_VERSION/main/postgresql.conf"
    
    if [[ -f "$postgres_conf" ]]; then
        # Backup original configuration
        cp "$postgres_conf" "$postgres_conf.backup.$(date +%Y%m%d)"
        
        # Apply TimescaleDB optimizations
        cat >> "$postgres_conf" << EOF

# IPE Platform TimescaleDB Optimizations
shared_preload_libraries = 'timescaledb'
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 8MB
min_wal_size = 1GB
max_wal_size = 4GB

# TimescaleDB specific settings
timescaledb.max_background_workers = 8
EOF
        
        log_info "PostgreSQL configuration updated. Restart required."
        
        if systemctl is-active --quiet postgresql; then
            log_warning "PostgreSQL restart recommended for optimal performance"
            read -p "Restart PostgreSQL now? (y/N): " -r
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                systemctl restart postgresql
                log_success "PostgreSQL restarted"
            fi
        fi
    else
        log_warning "PostgreSQL configuration file not found at $postgres_conf"
    fi
}

# Function to verify setup
verify_setup() {
    log_info "Verifying database setup..."
    
    export PGPASSWORD="$DB_PASSWORD"
    
    # Check database connection
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT version();" >/dev/null 2>&1; then
        log_success "Database connection successful"
    else
        log_error "Database connection failed"
        exit 1
    fi
    
    # Check TimescaleDB
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "SELECT * FROM timescaledb_information.hypertables;" >/dev/null 2>&1; then
        log_success "TimescaleDB is working"
    else
        log_error "TimescaleDB verification failed"
        exit 1
    fi
    
    # Check tables
    local table_count=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
    if [[ "$table_count" -gt 0 ]]; then
        log_success "Database tables created ($table_count tables)"
    else
        log_error "No tables found in database"
        exit 1
    fi
    
    log_success "Database setup verification completed"
}

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Database setup script for IPE platform

OPTIONS:
    --db-name NAME          Database name (default: ipe_db)
    --db-user USER          Database user (default: ipe_user)
    --db-password PASS      Database password (default: ipe_password)
    --db-host HOST          Database host (default: localhost)
    --db-port PORT          Database port (default: 5432)
    --backup-dir DIR        Backup directory (default: /var/backups/ipe)
    --environment ENV       Environment (development|test|production)
    --skip-install          Skip TimescaleDB installation
    --skip-optimize         Skip PostgreSQL optimization
    --skip-backups          Skip backup setup
    --verify-only           Only verify existing setup
    -h, --help              Show this help message

EXAMPLES:
    $0                                      # Full setup with defaults
    $0 --environment production            # Production setup
    $0 --verify-only                       # Verify existing setup
    $0 --skip-install --skip-optimize      # Setup without system changes

EOF
}

# Parse command line arguments
SKIP_INSTALL=false
SKIP_OPTIMIZE=false
SKIP_BACKUPS=false
VERIFY_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --db-name)
            DB_NAME="$2"
            shift 2
            ;;
        --db-user)
            DB_USER="$2"
            shift 2
            ;;
        --db-password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        --db-host)
            DB_HOST="$2"
            shift 2
            ;;
        --db-port)
            DB_PORT="$2"
            shift 2
            ;;
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --skip-optimize)
            SKIP_OPTIMIZE=true
            shift
            ;;
        --skip-backups)
            SKIP_BACKUPS=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
log_info "Starting IPE database setup..."
log_info "Environment: $ENVIRONMENT"
log_info "Database: $DB_NAME on $DB_HOST:$DB_PORT"

if [[ "$VERIFY_ONLY" == true ]]; then
    check_postgres
    verify_setup
    exit 0
fi

# Execute setup steps
check_postgres

if [[ "$SKIP_INSTALL" == false ]]; then
    check_timescaledb
fi

setup_database
enable_timescaledb
run_migrations

if [[ "$SKIP_OPTIMIZE" == false && "$ENVIRONMENT" == "production" ]]; then
    optimize_postgres
fi

if [[ "$SKIP_BACKUPS" == false ]]; then
    setup_backups
fi

verify_setup

log_success "IPE database setup completed successfully!"
log_info ""
log_info "Connection details:"
log_info "  Host: $DB_HOST"
log_info "  Port: $DB_PORT"
log_info "  Database: $DB_NAME"
log_info "  User: $DB_USER"
log_info ""
log_info "To connect:"
log_info "  psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"