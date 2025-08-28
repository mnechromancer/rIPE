#!/bin/bash
# Docker build script for IPE (Integrated Phenotypic Evolution) platform
# Supports development and production builds with size optimization

set -euo pipefail

# Default values
TARGET="production"
IMAGE_NAME="ipe"
IMAGE_TAG="latest"
REGISTRY=""
PUSH=false
BUILD_ARGS=""
CACHE_FROM=""
PLATFORM="linux/amd64"

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

# Display usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Docker build script for IPE platform

OPTIONS:
    -t, --target TARGET     Build target: development or production (default: production)
    -n, --name NAME         Image name (default: ipe)
    -g, --tag TAG           Image tag (default: latest)
    -r, --registry REGISTRY Registry URL for pushing (optional)
    -p, --push              Push to registry after build
    -a, --build-arg ARG     Additional build arguments (can be used multiple times)
    -c, --cache-from IMAGE  Use cache from specified image
    --platform PLATFORM    Target platform (default: linux/amd64)
    -h, --help              Show this help message

EXAMPLES:
    $0                                          # Build production image
    $0 -t development                           # Build development image
    $0 -t production -n ipe -g v1.0.0 -p       # Build and push production image
    $0 --cache-from ipe:latest                 # Build with cache
    $0 --platform linux/arm64                  # Build for ARM64

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -g|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -a|--build-arg)
            BUILD_ARGS="${BUILD_ARGS} --build-arg $2"
            shift 2
            ;;
        -c|--cache-from)
            CACHE_FROM="--cache-from $2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
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

# Validate target
if [[ "$TARGET" != "development" && "$TARGET" != "production" ]]; then
    log_error "Invalid target: $TARGET. Must be 'development' or 'production'"
    exit 1
fi

# Set full image name
if [[ -n "$REGISTRY" ]]; then
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running or not accessible"
    exit 1
fi

# Check if Dockerfile exists
if [[ ! -f "Dockerfile" ]]; then
    log_error "Dockerfile not found in current directory"
    exit 1
fi

log_info "Building IPE Docker image..."
log_info "Target: $TARGET"
log_info "Image: $FULL_IMAGE_NAME"
log_info "Platform: $PLATFORM"

# Build the image
BUILD_COMMAND="docker build \
    --target $TARGET \
    --platform $PLATFORM \
    --tag $FULL_IMAGE_NAME \
    $CACHE_FROM \
    $BUILD_ARGS \
    ."

log_info "Running: $BUILD_COMMAND"

if eval $BUILD_COMMAND; then
    log_success "Build completed successfully"
else
    log_error "Build failed"
    exit 1
fi

# Check image size
IMAGE_SIZE=$(docker images --format "table {{.Size}}" "$FULL_IMAGE_NAME" | tail -n 1)
log_info "Image size: $IMAGE_SIZE"

# Warn if production image is too large
if [[ "$TARGET" == "production" ]]; then
    SIZE_BYTES=$(docker inspect "$FULL_IMAGE_NAME" --format='{{.Size}}')
    MAX_SIZE_BYTES=$((1024*1024*1024))  # 1GB in bytes
    
    if [[ $SIZE_BYTES -gt $MAX_SIZE_BYTES ]]; then
        log_warning "Production image size ($IMAGE_SIZE) exceeds 1GB target"
    else
        log_success "Production image size is within 1GB target"
    fi
fi

# Push to registry if requested
if [[ "$PUSH" == true ]]; then
    if [[ -z "$REGISTRY" ]]; then
        log_error "Cannot push without registry specified"
        exit 1
    fi
    
    log_info "Pushing image to registry..."
    if docker push "$FULL_IMAGE_NAME"; then
        log_success "Push completed successfully"
    else
        log_error "Push failed"
        exit 1
    fi
fi

# Display final information
log_success "Docker build process completed"
log_info "Image: $FULL_IMAGE_NAME"
log_info "Size: $IMAGE_SIZE"

# Show how to run the container
log_info ""
log_info "To run the container:"
if [[ "$TARGET" == "development" ]]; then
    echo "  docker-compose up -d"
    echo "  docker run -it --rm -p 8000:8000 $FULL_IMAGE_NAME"
else
    echo "  docker run -d -p 8000:8000 --name ipe $FULL_IMAGE_NAME"
fi

log_info ""
log_info "To inspect the image:"
echo "  docker inspect $FULL_IMAGE_NAME"
echo "  docker history $FULL_IMAGE_NAME"