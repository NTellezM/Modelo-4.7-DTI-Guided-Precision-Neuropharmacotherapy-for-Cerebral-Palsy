# =============================================================================
# dti_mapper — Multi-stage Docker build
#
# Stage 1: Compile Rust extension with maturin
# Stage 2: Slim Python image with compiled wheel
#
# Build:  docker build -t dti_mapper .
# Run:    docker run --rm -v $(pwd)/data:/data dti_mapper \
#           map --dti /data/dti.nii.gz --mesh /data/mesh.xdmf --output /data/tensors.h5
# =============================================================================

# --- Stage 1: Build ---
FROM rust:1.78-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-pip python3-venv pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install maturin
RUN pip3 install --break-system-packages maturin

# Copy Rust sources first (better cache utilization)
COPY Cargo.toml rust-toolchain.toml README.md ./
COPY src/ src/

# Copy Python sources
COPY pyproject.toml ./
COPY python/ python/

# Build the wheel
RUN maturin build --release --strip --out /wheels


# --- Stage 2: Runtime ---
FROM python:3.11-slim

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    nibabel>=5.0 \
    meshio>=5.3 \
    h5py>=3.9 \
    scipy>=1.11 \
    click>=8.1

# Install the built wheel
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# Copy example data generator
COPY data/generate_synthetic.py /opt/dti_mapper/data/

WORKDIR /workspace

ENTRYPOINT ["dti-mapper"]
CMD ["--help"]
