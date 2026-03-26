FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    numpy nibabel meshio scipy scikit-image gmsh

COPY geom_preprocessing/ geom_preprocessing/

ENTRYPOINT ["python", "-m", "geom_preprocessing.cli"]
