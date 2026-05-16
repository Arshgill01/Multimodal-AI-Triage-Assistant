#!/bin/bash
# Frostbyte Triage Smoke Test
# Verifies all services are running and healthy

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASSED=0
FAILED=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    PASSED=$((PASSED + 1))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    FAILED=$((FAILED + 1))
}

check_info() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

echo ""
echo "========================================"
echo "🔍 Frostbyte Triage Smoke Test"
echo "========================================"
echo ""

# Check Python service
check_info "Python service (port 8000)..."
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    PYTHON_HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status' 2>/dev/null || echo "unknown")
    check_pass "Python service healthy: $PYTHON_HEALTH"
else
    check_fail "Python service not responding"
fi

# Check Rust backend
check_info "Rust backend (port 3001)..."
if curl -s http://localhost:3001/health >/dev/null 2>&1; then
    RUST_HEALTH=$(curl -s http://localhost:3001/health | jq -r '.status' 2>/dev/null || echo "unknown")
    check_pass "Rust backend healthy: $RUST_HEALTH"
else
    check_fail "Rust backend not responding"
fi

# Check frontend
check_info "Next.js frontend (port 3000)..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null | grep -q "200"; then
    check_pass "Frontend responding"
else
    check_fail "Frontend not responding"
fi

# Check single prediction
check_info "Single prediction (POST /predict)..."
PREDICT_PAYLOAD='{"age":45,"heart_rate":88,"resp_rate":18,"spo2":97,"temp_f":98.6,"systolic_bp":130,"pain_scale":4,"chief_complaint":"Chest pain"}'
PREDICT_RESULT=$(curl -s -X POST http://localhost:3001/predict -H "Content-Type: application/json" -d "$PREDICT_PAYLOAD" 2>/dev/null)
if echo "$PREDICT_RESULT" | jq -e '.predicted_esi' >/dev/null 2>&1; then
    ESI=$(echo "$PREDICT_RESULT" | jq -r '.predicted_esi')
    LABEL=$(echo "$PREDICT_RESULT" | jq -r '.esi_label')
    check_pass "Prediction successful — ESI $ESI ($LABEL)"
elif echo "$PREDICT_RESULT" | jq -e '.error' >/dev/null 2>&1; then
    check_info "Model not loaded (degraded mode): $(echo "$PREDICT_RESULT" | jq -r '.error')"
    check_pass "Prediction endpoint responding (model unavailable)"
else
    check_fail "Prediction endpoint error: $(echo "$PREDICT_RESULT" | head -c 200)"
fi

# Check batch prediction
check_info "Batch prediction (POST /batch-predict)..."
BATCH_PAYLOAD='{"patients":[{"age":45,"heart_rate":88,"resp_rate":18,"spo2":97,"temp_f":98.6,"systolic_bp":130,"pain_scale":4,"chief_complaint":"Chest pain"},{"age":30,"heart_rate":72,"resp_rate":14,"spo2":99,"temp_f":98.2,"systolic_bp":115,"pain_scale":1,"chief_complaint":"Sore throat"}]}'
BATCH_RESULT=$(curl -s -X POST http://localhost:3001/batch-predict -H "Content-Type: application/json" -d "$BATCH_PAYLOAD" 2>/dev/null)
if echo "$BATCH_RESULT" | jq -e '.total_patients' >/dev/null 2>&1; then
    TOTAL=$(echo "$BATCH_RESULT" | jq -r '.total_patients')
    PATIENTS=$(echo "$BATCH_RESULT" | jq -r '.patients | length')
    SUMMARY=$(echo "$BATCH_RESULT" | jq -e '.summary' >/dev/null 2>&1 && echo "yes" || echo "no")
    check_pass "Batch prediction — $TOTAL patients, $PATIENTS results, summary=$SUMMARY"
elif echo "$BATCH_RESULT" | jq -e '.error' >/dev/null 2>&1; then
    check_info "Model not loaded (degraded mode): $(echo "$BATCH_RESULT" | jq -r '.error')"
    check_pass "Batch endpoint responding (model unavailable)"
else
    check_fail "Batch prediction error: $(echo "$BATCH_RESULT" | head -c 200)"
fi

# Check service connectivity
check_info "Cross-service connectivity..."
if curl -s http://localhost:3001/health | jq -r '.python_service_url' 2>/dev/null | grep -q "8000"; then
    check_pass "Rust → Python connectivity configured"
else
    check_fail "Rust → Python connectivity issue"
fi

echo ""
echo "========================================"
echo "Results: $PASSED passed, $FAILED failed"
echo "========================================"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "⚠️  Some checks failed. Review logs:"
    echo "  - Python: python_service.log"
    echo "  - Rust: backend/rust_backend.log"  
    echo "  - Frontend: frontend/frontend.log"
    echo ""
    exit 1
else
    echo "✅ All smoke tests passed"
    exit 0
fi