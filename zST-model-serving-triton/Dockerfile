# ─────────────────────────────────────────────────────────────────────────────
# Triton Inference Server — APP Fraud LSTM (ONNX backend)
#
# Targets:
#   - IBM z16+ / LinuxONE 4+ (linux/s390x):
#       icr.io/ibmz/ibmz-accelerated-for-nvidia-triton-inference-server:1.6.2
#       Builds and installs the IBM Telum plugin EP from source so Triton's
#       ONNX Runtime backend can offload ops to the on-chip Telum AI Unit.
#
#   - x86_64 dev / CI (linux/amd64):
#       nvcr.io/nvidia/tritonserver:24.01-py3
#       Telum EP build steps are skipped automatically on non-s390x.
#
# Build (default — IBM Z):
#   docker build -t triton-fraud-detection .
#
# Build (x86_64 dev machine):
#   docker build \
#     --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.01-py3 \
#     -t triton-fraud-detection .
#
# Run:
#   docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
#     -v $(pwd)/model_repository:/models triton-fraud-detection
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=icr.io/ibmz/ibmz-accelerated-for-nvidia-triton-inference-server:latest

# ── Stage 1: Build ONNX Runtime and Telum EP plugin on s390x ─────────────────
FROM ${BASE_IMAGE} AS builder
USER root
ARG TARGETARCH=s390x

ENV ORT_REPO=https://github.com/microsoft/onnxruntime.git \
    ORT_REF=main \
    TELUM_EP_REPO=https://github.com/IBM/onnxruntime-ep-telum.git \
    TELUM_EP_REF=main

RUN mkdir -p /opt/telum_ep /opt/wheels

RUN if [ "${TARGETARCH}" = "s390x" ]; then \
      set -eux; \
      if which microdnf >/dev/null 2>&1; then \
        microdnf install -y git cmake gcc-c++ make ninja-build python3-devel python3-pip patch which tar gzip; \
      elif which apt-get >/dev/null 2>&1; then \
        apt-get update -qq && apt-get install -y --no-install-recommends git cmake ninja-build g++ make python3-dev python3-pip patch; \
      fi; \
      pip3 install --no-cache-dir wheel packaging setuptools numpy; \
      git clone --depth 1 --branch "${ORT_REF}" --recursive "${ORT_REPO}" /tmp/onnxruntime; \
      /tmp/onnxruntime/build.sh \
        --allow_running_as_root \
        --config Release \
        --build_wheel \
        --skip_tests \
        --parallel 2 \
        --compile_no_warning_as_error \
        --skip_submodule_sync; \
      cp /tmp/onnxruntime/build/Linux/Release/dist/*.whl /opt/wheels/; \
      git clone --depth 1 --branch "${TELUM_EP_REF}" "${TELUM_EP_REPO}" /tmp/onnxruntime-ep-telum; \
      make -C /tmp/onnxruntime-ep-telum build \
        ONNXRUNTIME_INCLUDE_DIR=/tmp/onnxruntime/include/onnxruntime/core/session \
        TELUM_EP_ENABLE_ZDNN=ON; \
      cp /tmp/onnxruntime-ep-telum/build/libtelum_plugin_ep.so /opt/telum_ep/libtelum_plugin_ep.so; \
    fi

# ── Stage 2: Final Triton image ──────────────────────────────────────────────
FROM ${BASE_IMAGE}

LABEL maintainer="IBM Z Payments Fraud TIS" \
      description="Triton Inference Server — APP Fraud LSTM (Python backend + ONNX Runtime + Telum EP)"

USER root

ENV TELUM_EP_LIBRARY_PATH=/opt/telum_ep/libtelum_plugin_ep.so

# Copy built Telum EP plugin and ONNX Runtime wheel from builder stage
COPY --from=builder /opt/telum_ep /opt/telum_ep
COPY --from=builder /opt/wheels   /opt/wheels

# Install onnxruntime wheel if present
RUN if compgen -G "/opt/wheels/*.whl" > /dev/null; then \
      pip3 install --no-cache-dir /opt/wheels/*.whl; \
    fi && rm -rf /opt/wheels

# ── Python helper scripts ─────────────────────────────────────────────────────
WORKDIR /workspace

COPY serve.py              /workspace/serve.py
COPY test_triton_client.py /workspace/test_triton_client.py

# ── Model repository mount-point ──────────────────────────────────────────────
RUN mkdir -p /models /workspace && chown -R 1000:1000 /models /workspace /opt/telum_ep
VOLUME ["/models"]

USER 1000

# ── Triton ports ──────────────────────────────────────────────────────────────
# 8000 → HTTP (REST)
# 8001 → gRPC
# 8002 → Prometheus metrics
EXPOSE 8000 8001 8002

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -sf http://localhost:8000/v2/health/ready || exit 1

# ── Default command ───────────────────────────────────────────────────────────
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
