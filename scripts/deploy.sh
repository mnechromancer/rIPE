#!/bin/bash
# Deployment script for IPE (Integrated Phenotypic Evolution) platform
# Supports staging and production deployments with zero-downtime and rollback capabilities

set -euo pipefail

# Default configuration
ENVIRONMENT="staging"
IMAGE_TAG="latest"
REGISTRY="ghcr.io"
REPOSITORY_NAME="mnechromancer/ripe/ipe"
BACKUP_ONLY=false
ROLLBACK=false
PREVIOUS_VERSION=""
ZERO_DOWNTIME=false
VERIFY_HEALTH=false
DRY_RUN=false
TIMEOUT=300

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

# Configuration based on environment
set_environment_config() {
    case "$ENVIRONMENT" in
        "staging")
            DEPLOY_HOST="staging-server"
            DEPLOY_USER="deploy"
            COMPOSE_FILE="docker-compose.staging.yml"
            HEALTH_URL="https://staging.ipe.example.com/health"
            DB_BACKUP_PATH="/var/backups/ipe-staging"
            SERVICE_NAME="ipe-staging"
            ;;
        "production")
            DEPLOY_HOST="prod-server"
            DEPLOY_USER="deploy"
            COMPOSE_FILE="docker-compose.production.yml"
            HEALTH_URL="https://api.ipe.example.com/health"
            DB_BACKUP_PATH="/var/backups/ipe-production"
            SERVICE_NAME="ipe-production"
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

# Display usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deployment script for IPE platform

OPTIONS:
    -e, --environment ENV       Target environment: staging or production (default: staging)
    -t, --image-tag TAG         Docker image tag to deploy (default: latest)
    -r, --registry REGISTRY     Container registry URL (default: ghcr.io)
    -n, --repository-name NAME  Repository name (default: mnechromancer/ripe/ipe)
    --backup-only               Only create backup, don't deploy
    --backup-name NAME          Custom backup name
    --rollback                  Rollback to previous version
    --previous-version VERSION  Specific version to rollback to
    --zero-downtime             Use zero-downtime deployment strategy
    --verify-health             Verify health after deployment
    --dry-run                   Show what would be done without executing
    --timeout SECONDS           Health check timeout (default: 300)
    -h, --help                  Show this help message

EXAMPLES:
    $0                                          # Deploy latest to staging
    $0 -e production -t v1.2.3                 # Deploy v1.2.3 to production
    $0 --rollback -e production                # Rollback production
    $0 --backup-only --backup-name "pre-v1.2"  # Create backup only
    $0 --zero-downtime --verify-health         # Zero-downtime with health checks

EOF
}

# Create database backup
create_backup() {
    local backup_name="${1:-backup-$(date +%Y%m%d-%H%M%S)}"
    local backup_file="$DB_BACKUP_PATH/${backup_name}.sql"
    
    log_info "Creating database backup: $backup_name"
    
    # Ensure backup directory exists
    ssh "$DEPLOY_USER@$DEPLOY_HOST" "sudo mkdir -p $DB_BACKUP_PATH"
    
    # Create backup via remote script
    ssh "$DEPLOY_USER@$DEPLOY_HOST" << EOF
        docker-compose -f $COMPOSE_FILE exec -T db pg_dump -U ipe_user ipe_db > $backup_file
        gzip $backup_file
        echo "Backup created: ${backup_file}.gz"
EOF
    
    log_success "Database backup completed"
    return 0
}

# Verify deployment health
verify_health() {
    log_info "Verifying deployment health..."
    
    local retries=0
    local max_retries=$((TIMEOUT / 10))
    
    while [ $retries -lt $max_retries ]; do
        if curl -f -s "$HEALTH_URL" >/dev/null 2>&1; then
            log_success "Health check passed"
            return 0
        fi
        
        retries=$((retries + 1))
        log_info "Health check attempt $retries/$max_retries..."
        sleep 10
    done
    
    log_error "Health check failed after $max_retries attempts"
    return 1
}

# Get current running version
get_current_version() {
    ssh "$DEPLOY_USER@$DEPLOY_HOST" << 'EOF'
        if docker-compose -f $COMPOSE_FILE ps -q ipe-api >/dev/null 2>&1; then
            docker inspect $(docker-compose -f $COMPOSE_FILE ps -q ipe-api) \
                --format='{{index .Config.Labels "org.opencontainers.image.version"}}'
        else
            echo "none"
        fi
EOF
}

# Deploy application
deploy() {
    local full_image_name="$REGISTRY/$REPOSITORY_NAME:$IMAGE_TAG"
    
    log_info "Starting deployment to $ENVIRONMENT"
    log_info "Image: $full_image_name"
    
    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN - No actual deployment will occur"
        return 0
    fi
    
    # Create backup unless it's staging
    if [ "$ENVIRONMENT" = "production" ]; then
        create_backup "pre-deploy-$(date +%Y%m%d-%H%M%S)"
    fi
    
    # Get current version for potential rollback
    local current_version
    current_version=$(get_current_version)
    log_info "Current version: $current_version"
    
    # Upload deployment files
    log_info "Uploading deployment configuration..."
    scp "$COMPOSE_FILE" "$DEPLOY_USER@$DEPLOY_HOST:~/"
    
    # Execute deployment
    if [ "$ZERO_DOWNTIME" = true ]; then
        deploy_zero_downtime "$full_image_name"
    else
        deploy_standard "$full_image_name"
    fi
    
    # Verify health if requested
    if [ "$VERIFY_HEALTH" = true ]; then
        if ! verify_health; then
            log_error "Deployment health check failed"
            if [ "$ENVIRONMENT" = "production" ]; then
                log_warning "Initiating automatic rollback..."
                rollback_deployment "$current_version"
            fi
            exit 1
        fi
    fi
    
    log_success "Deployment completed successfully"
}

# Standard deployment (with brief downtime)
deploy_standard() {
    local image_name="$1"
    
    log_info "Executing standard deployment..."
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" << EOF
        # Set image tag for compose file
        export IMAGE_TAG="$IMAGE_TAG"
        export REGISTRY="$REGISTRY"
        export REPOSITORY_NAME="$REPOSITORY_NAME"
        
        # Pull new image
        docker pull $image_name
        
        # Stop services
        docker-compose -f $COMPOSE_FILE stop ipe-api
        
        # Update and start services
        docker-compose -f $COMPOSE_FILE up -d
        
        # Clean up old images
        docker image prune -f
EOF
}

# Zero-downtime deployment (blue-green style)
deploy_zero_downtime() {
    local image_name="$1"
    
    log_info "Executing zero-downtime deployment..."
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" << EOF
        # Set image tag for compose file
        export IMAGE_TAG="$IMAGE_TAG"
        export REGISTRY="$REGISTRY"
        export REPOSITORY_NAME="$REPOSITORY_NAME"
        
        # Pull new image
        docker pull $image_name
        
        # Start new container alongside existing one
        docker-compose -f $COMPOSE_FILE up -d --scale ipe-api=2 --no-recreate
        
        # Wait for new container to be ready
        sleep 30
        
        # Check if new container is healthy
        NEW_CONTAINER=\$(docker-compose -f $COMPOSE_FILE ps -q ipe-api | tail -n 1)
        if docker exec \$NEW_CONTAINER curl -f http://localhost:8000/health; then
            # Stop old container
            OLD_CONTAINER=\$(docker-compose -f $COMPOSE_FILE ps -q ipe-api | head -n 1)
            docker stop \$OLD_CONTAINER
            docker rm \$OLD_CONTAINER
            
            # Scale back to 1 instance
            docker-compose -f $COMPOSE_FILE up -d --scale ipe-api=1
        else
            echo "New container failed health check"
            exit 1
        fi
        
        # Clean up old images
        docker image prune -f
EOF
}

# Rollback deployment
rollback_deployment() {
    local version="${1:-$PREVIOUS_VERSION}"
    
    if [ -z "$version" ] || [ "$version" = "none" ]; then
        log_error "No previous version specified for rollback"
        exit 1
    fi
    
    log_warning "Rolling back to version: $version"
    
    if [ "$DRY_RUN" = true ]; then
        log_warning "DRY RUN - No actual rollback will occur"
        return 0
    fi
    
    local rollback_image="$REGISTRY/$REPOSITORY_NAME:$version"
    
    ssh "$DEPLOY_USER@$DEPLOY_HOST" << EOF
        # Set image tag for rollback
        export IMAGE_TAG="$version"
        export REGISTRY="$REGISTRY"
        export REPOSITORY_NAME="$REPOSITORY_NAME"
        
        # Pull rollback image
        docker pull $rollback_image
        
        # Execute rollback
        docker-compose -f $COMPOSE_FILE stop ipe-api
        docker-compose -f $COMPOSE_FILE up -d
EOF
    
    # Verify rollback health
    if [ "$VERIFY_HEALTH" = true ]; then
        if verify_health; then
            log_success "Rollback completed successfully"
        else
            log_error "Rollback health check failed"
            exit 1
        fi
    fi
}

# Main execution function
main() {
    set_environment_config
    
    log_info "IPE Platform Deployment Script"
    log_info "Environment: $ENVIRONMENT"
    log_info "Target: $DEPLOY_HOST"
    
    if [ "$BACKUP_ONLY" = true ]; then
        create_backup "${BACKUP_NAME:-backup-$(date +%Y%m%d-%H%M%S)}"
        exit 0
    fi
    
    if [ "$ROLLBACK" = true ]; then
        rollback_deployment
        exit 0
    fi
    
    deploy
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -n|--repository-name)
            REPOSITORY_NAME="$2"
            shift 2
            ;;
        --backup-only)
            BACKUP_ONLY=true
            shift
            ;;
        --backup-name)
            BACKUP_NAME="$2"
            shift 2
            ;;
        --rollback)
            ROLLBACK=true
            shift
            ;;
        --previous-version)
            PREVIOUS_VERSION="$2"
            shift 2
            ;;
        --zero-downtime)
            ZERO_DOWNTIME=true
            shift
            ;;
        --verify-health)
            VERIFY_HEALTH=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
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

# Validate environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    log_error "Environment must be 'staging' or 'production'"
    exit 1
fi

# Execute main function
main