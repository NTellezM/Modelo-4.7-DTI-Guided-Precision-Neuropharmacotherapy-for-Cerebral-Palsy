FROM continuumio/miniconda3:latest

RUN conda install -c conda-forge -y \
    fenics=2019.1.0 \
    meshio \
    h5py \
    nibabel \
    scipy \
    numpy \
    matplotlib \
    && conda clean -afy

COPY scripts/ /opt/fenics_solver/scripts/
ENV PYTHONPATH="/opt/fenics_solver/scripts:${PYTHONPATH}"

WORKDIR /work

ENTRYPOINT ["python3"]
CMD ["scripts/run_pipeline.py", "--help"]
