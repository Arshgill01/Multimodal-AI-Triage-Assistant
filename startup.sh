#!/bin/bash
# Frostbyte Triage Demo Launcher
# Starts all three services: Python, Rust, and frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required environment
check_env() {
    log_info "Checking environment..."

    # Check Python service dependencies
    if [ ! -f "preprocessing_service.py" ]; then
        log_error "preprocessing_service.py not found"
        exit 1
    fi

    # Check Rust backend
    if [ ! -d "backend" ]; then
        log_error "backend directory not found"
        exit 1
    fi

    # Check frontend
    if [ ! -d "frontend" ]; then
        log_error "frontend directory not found"
        exit 1
    fi

    log_success "Environment structure OK"
}

# Start Python sidecar
start_python() {
    log_info "Starting Python preprocessing service (port 8000)..."
    cd "$SCRIPT_DIR"

    # Check if port is in use
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warn "Port 8000 already in use - skipping Python start"
    else
        # Set environment
        export TRIAGE_DATA_DIR="${TRIAGE_DATA_DIR:-.}"
        export GEMINI_API_KEY="${GEMINI_API_KEY:-}"

        # Start in background
        nohup python3 -m uvicorn preprocessing_service:app --host 0.0.0.0 --port 8000 > python_service.log 2>&1 &
        PYTHON_PID=$!
        echo $PYTHON_PID > .python.pid
        log_success "Python service started (PID: $PYTHON_PID)"
    fi
}

# Start Rust backend
start_rust() {
    log_info "Starting Rust backend (port 3001)..."
    cd "$SCRIPT_DIR/backend"

    # Check if port is in use
    if lsof -Pi :3001 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warn "Port 3001 already in use - skipping Rust start"
    else
        # Set environment
        export TRIAGE_MODEL_PATH="${TRIAGE_MODEL_PATH:-../triage_multimodal_model.txt}"
        export TRIAGE_PYTHON_URL="${TRIAGE_PYTHON_URL:-http://localhost:8000}"
        export TRIAGE_AUDIT_DB="${TRIAGE_AUDIT_DB:-../triage_audit.db}"

        # Start in background
        nohup cargo run --release > rust_backend.log 2>&1 &
        RUST_PID=$!
        echo $RUST_PID > .rust.pid
        log_success "Rust backend started (PID: $RUST_PID)"
    fi
}

# Start frontend
start_frontend() {
    log_info "Starting Next.js frontend (port 3000)..."
    cd "$SCRIPT_DIR/frontend"

    # Check if port is in use
    if lsof -Pi :3000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warn "Port 3000 already in use - skipping frontend start"
    else
        # Set environment
        export NEXT_PUBLIC_RUST_API="${NEXT_PUBLIC_RUST_API:-http://localhost:3001}"
        export NEXT_PUBLIC_PYTHON_API="${NEXT_PUBLIC_PYTHON_API:-http://localhost:8000}"

        # Start in background
        nohup npm run dev > frontend.log 2>&1 &
        FRONTEND_PID=$!
        echo $FRONTEND_PID > .frontend.pid
        log_success "Frontend started (PID: $FRONTEND_PID)"
    fi
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."

    local max_attempts=30
    local attempt=0

    # Wait for Python
    log_info "Checking Python service..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            log_success "Python service ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    if [ $attempt -eq $max_attempts ]; then
        log_warn "Python service may not be ready yet"
    fi

    # Wait for Rust
    attempt=0
    log_info "Checking Rust backend..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:3001/health >/dev/null 2>&1; then
            log_success "Rust backend ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    if [ $attempt -eq $max_attempts ]; then
        log_warn "Rust backend may not be ready yet"
    fi

    # Wait for frontend
    attempt=0
    log_info "Checking Next.js frontend..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:3000 >/dev/null 2>&1; then
            log_success "Frontend ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    if [ $attempt -eq $max_attempts ]; then
        log_warn "Frontend may not be ready yet"
    fi
}

# Print status
print_status() {
    echo ""
    echo "========================================"
    echo "🎯 Frostbyte Triage System Ready"
    echo "========================================"
    echo ""
    echo "Frontend:  http://localhost:3000"
    echo "Rust:     http://localhost:3001"
    echo "Python:   http://localhost:8000"
    echo ""
    echo "Health checks:"
    curl -s http://localhost:3001/health | jq -c '.status' 2>/dev/null || echo "  Rust: unknown"
    curl -s http://localhost:8000/health | jq -c '.status' 2>/dev/null || echo "  Python: unknown"
    echo ""
    echo "========================================"
    echo ""
    echo "To stop all services:"
    echo "  ./shutdown.sh"
    echo ""
}

# Main
main() {
    echo ""
    echo "========================================"
    echo "🚀 Frostbyte Triage Startup"
    echo "========================================"
    echo ""

    check_env
    start_python
    start_rust
    start_frontend
    wait_for_services
    print_status
}

# Run
main