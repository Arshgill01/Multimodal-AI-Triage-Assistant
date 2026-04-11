#!/bin/bash
# Frostbyte Triage Shutdown Script
# Stops all three services

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Kill services using PIDs if they exist
kill_service() {
    local name=$1
    local pid_file=$2

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Stopping $name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            sleep 1
            # Force kill if still alive
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
            log_success "$name stopped"
        else
            log_warn "$name was not running (stale PID file)"
        fi
        rm -f "$pid_file"
    else
        log_warn "$name PID file not found"
    fi
}

# Also kill by port if PID file missing
kill_by_port() {
    local port=$1
    local name=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_info "Killing $name on port $port..."
        lsof -ti :$port | xargs kill 2>/dev/null || true
        log_success "$name stopped"
    else
        log_warn "$name not running on port $port"
    fi
}

echo ""
echo "========================================"
echo "🛑 Frostbyte Triage Shutdown"
echo "========================================"
echo ""

# Stop services in reverse order
kill_service "Frontend" ".frontend.pid"
kill_by_port 3000 "Frontend"

kill_service "Rust backend" ".rust.pid"
kill_by_port 3001 "Rust backend"

kill_service "Python service" ".python.pid"
kill_by_port 8000 "Python service"

# Clean up log files
rm -f python_service.log rust_backend.log frontend.log

echo ""
log_success "All services stopped"
echo ""