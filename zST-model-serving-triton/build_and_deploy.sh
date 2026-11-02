#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# build_and_deploy.sh — Build and start the Triton fraud-detection container
#
# Usage (from project root or this directory):
#   cd zST-model-serving-triton
#   bash build_and_deploy.sh
#
# Environment variables (optional overrides):
#   IMAGE_NAME     docker image name              (default: triton-fraud-detection)
#   IMAGE_TAG      docker image tag               (default: latest)
#   CONTAINER_NAME running container name         (default: triton-server)
#   BASE_IMAGE     Triton base image to use       (default: IBM Z ICR image)
#   ONNX_MODEL     path to the unified ONNX file  (default: ../saved_model/...)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve script directory so the script works from any CWD ─────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Configuration ──────────────────────────────────────────────────────────────
IMAGE_NAME="${IMAGE_NAME:-triton-fraud-detection}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-triton-server}"
BASE_IMAGE="${BASE_IMAGE:-icr.io/ibmz/ibmz-accelerated-for-nvidia-triton-inference-server:latest}"
TARGETARCH="s390x"
ONNX_MODEL="${ONNX_MODEL:-${SCRIPT_DIR}/../saved_model/fraud_detection_unified.onnx}"
MODEL_REPO="${SCRIPT_DIR}/model_repository"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Triton Fraud Detection — Build & Deploy                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Image        : ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Base image   : ${BASE_IMAGE}"
echo "  ONNX model   : ${ONNX_MODEL}"
echo "  Model repo   : ${MODEL_REPO}"
echo ""

# ── Step 1: Validate ONNX model exists ────────────────────────────────────────
echo "Step 1: Validating ONNX model..."
if [[ ! -f "${ONNX_MODEL}" ]]; then
  echo "ERROR: ONNX model not found at '${ONNX_MODEL}'"
  echo "       Run the training notebook first, or set ONNX_MODEL=/path/to/model.onnx"
  exit 1
fi
echo "  ✓ Found: ${ONNX_MODEL}"

# ── Step 2: Set up Triton model repository ────────────────────────────────────
echo ""
echo "Step 2: Setting up Triton model repository..."
python3 serve.py \
  --model-path "${ONNX_MODEL}" \
  --repo-dir   "${MODEL_REPO}"

# ── Step 3: Build Docker image ────────────────────────────────────────────────
echo ""
echo "Step 3: Building Docker image..."
docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" --build-arg TARGETARCH="${TARGETARCH}" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  .
echo "  ✓ Image built: ${IMAGE_NAME}:${IMAGE_TAG}"

# ── Step 4: Stop and remove any existing container ───────────────────────────
echo ""
echo "Step 4: Removing existing container (if any)..."
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# ── Step 5: Start Triton container ────────────────────────────────────────────
echo ""
echo "Step 5: Starting Triton Inference Server container..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "${MODEL_REPO}:/models:ro" \
  "${IMAGE_NAME}:${IMAGE_TAG}"

echo "  ✓ Container '${CONTAINER_NAME}' started"

# ── Step 6: Wait for readiness ────────────────────────────────────────────────
echo ""
echo "Step 6: Waiting for Triton to become ready..."
MAX_WAIT=120
WAITED=0
until curl -sf http://localhost:8000/v2/health/ready > /dev/null 2>&1; do
  if [[ ${WAITED} -ge ${MAX_WAIT} ]]; then
    echo "ERROR: Triton did not become ready within ${MAX_WAIT}s"
    echo "       Check logs with:  docker logs ${CONTAINER_NAME}"
    exit 1
  fi
  printf "."
  sleep 3
  WAITED=$((WAITED + 3))
done
echo ""
echo "  ✓ Triton is ready (${WAITED}s)"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Deployment complete!                                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  HTTP  (REST) : http://localhost:8000"
echo "  gRPC         : localhost:8001"
echo "  Metrics      : http://localhost:8002/metrics"
echo ""
echo "  Run tests:"
echo "    pip install tritonclient[http] numpy"
echo "    python3 test_triton_client.py"
echo ""
echo "  Stream logs:"
echo "    docker logs -f ${CONTAINER_NAME}"
echo ""
echo "  Stop server:"
echo "    docker stop ${CONTAINER_NAME} && docker rm ${CONTAINER_NAME}"
