.PHONY: build test mesh tensors simulate simulate-addiction test-synthetic run-patient clean help

SHELL := /bin/bash
DATA_DIR := $(PWD)/data
OUTPUT_DIR := $(PWD)/output

help:
	@echo "Modelo 4.7 — Pipeline de simulación"
	@echo ""
	@echo "  make build              Construir imágenes Docker"
	@echo "  make test-synthetic     Pipeline completo con datos sintéticos"
	@echo "  make test-addiction     Test de plasticidad + MPC"
	@echo "  make mesh               Paso 1: generar malla"
	@echo "  make tensors            Paso 2: mapear tensores DTI"
	@echo "  make simulate           Paso 3: simulación normal"
	@echo "  make simulate-addiction  Paso 3b: simulación con adicción"
	@echo "  make run-patient ID=x   Pipeline completo para paciente"
	@echo "  make clean              Limpiar outputs"

# --- Build ---
build:
	cd dti_mapper && maturin develop --release
	docker build -t modelo47-fenics fenics_solver/

# --- Datos sintéticos ---
data/synthetic/synthetic_dti.nii.gz:
	python scripts/generate_synthetic.py

# --- Paso 1: Malla ---
mesh: data/synthetic/synthetic_dti.nii.gz
	@mkdir -p $(OUTPUT_DIR)/mesh_fenics
	docker run --rm --entrypoint "" -v $(PWD):/work -w /work modelo47-fenics \
		python3 fenics_solver/scripts/convert_mesh.py \
		--input data/synthetic/synthetic_mesh.xdmf \
		--output-dir $(OUTPUT_DIR)/mesh_fenics

# --- Paso 2: Tensores DTI ---
tensors: mesh
	cd dti_mapper && python python/examples/map_tensors.py \
		--synthetic --output $(OUTPUT_DIR)/tensors/tensors.h5

# --- Paso 3: Simulación normal ---
simulate: mesh
	docker run --rm --entrypoint "" -v $(PWD):/work -w /work modelo47-fenics \
		python3 fenics_solver/scripts/run_pipeline.py \
		--mesh-input data/synthetic/synthetic_mesh.xdmf \
		--synthetic --output $(OUTPUT_DIR)/results \
		--dt 0.05 --T-final 5.0

# --- Paso 3b: Simulación con adicción ---
simulate-addiction: mesh
	docker run --rm --entrypoint "" -v $(PWD):/work -w /work modelo47-fenics \
		python3 fenics_solver/scripts/run_pipeline.py \
		--mesh-input data/synthetic/synthetic_mesh.xdmf \
		--synthetic --addiction --nac-tag 3 \
		--output $(OUTPUT_DIR)/results_addiction \
		--dt 0.05 --T-final 100.0

# --- Test completo sintético ---
test-synthetic: data/synthetic/synthetic_dti.nii.gz simulate
	@echo "Pipeline sintético completado → $(OUTPUT_DIR)/results/"

# --- Test de módulos de adicción ---
test-addiction:
	docker run --rm --entrypoint "" -v $(PWD):/work -w /work modelo47-fenics \
		python3 fenics_solver/scripts/test_addiction.py

# --- Pipeline paciente ---
run-patient:
	@test -n "$(ID)" || (echo "Uso: make run-patient ID=patient_001" && exit 1)
	python scripts/run_pipeline_patient.py --config config/pipeline_config.json --patient-id $(ID)

# --- Limpieza ---
clean:
	rm -rf $(OUTPUT_DIR)/
	rm -rf data/synthetic/*.xdmf data/synthetic/*.h5 data/synthetic/*.nii.gz
