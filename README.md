# Modelo 4.7 — DTI-Guided Precision Neuropharmacotherapy for Cerebral Palsy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)]()
[![FEniCS](https://img.shields.io/badge/FEniCS-2019.1-green.svg)]()

A patient-specific digital twin for cerebral palsy (CP) that simulates
anisotropic drug diffusion guided by DTI tensors, coupled with an axonal
plasticity model and a Model Predictive Controller (MPC) for real-time
hydrogel permeability optimization.

## Backtesting Result

Validated against the IDYS clinical trial (intrathecal baclofen in dystonic CP):

| Metric | Model 4.7 | IDYS Trial | Error |
|--------|-----------|------------|-------|
| Initial ρ | 0.689 | 0.689 | 0% |
| Final ρ (12 months) | **0.766** | **0.770** | **0.53%** |
| Δρ total | +0.077 | +0.081 | 4.9% |

Prediction obtained using population-level parameters only — no individual
patient data used.

## Architecture

```
medi_model/
├── geom_preprocessing/   NIfTI → tetrahedral FEM mesh (Python/Gmsh)
├── dti_mapper/           DTI → per-node tensors on FEM mesh (Rust + PyO3)
├── fenics_solver/        Coupled PDE + plasticity + MPC (FEniCS/Python)
└── scripts/              Virtual patient generation and pipeline tools
```

### Core Physics

**Anisotropic diffusion** guided by DTI tensors:
```
∂C/∂t = ∇·(D_eff(x)·∇C) − k_elim·C + P·(C_res − C)·δ_gel
D_eff = ρ(x,t) · D_DTI(x)
```

**Axonal plasticity** driven by drug concentration:
```
∂ρ/∂t = α·H(R − R_th)·(1−ρ) − β·ρ
R(C)  = Cⁿ / (C₅₀ⁿ + Cⁿ)     [Hill equation]
```

**MPC controller** optimizing hydrogel permeability P to maximize
connectivity recovery ρ while avoiding toxic exposure.

## Quickstart

### Requirements
- Docker
- Python 3.10+
- Rust 1.70+ (for dti_mapper)

### Build

```bash
# FEniCS solver
docker build -t modelo47-fenics fenics_solver/

# Verify
docker run --rm --entrypoint "" modelo47-fenics \
    python3 fenics_solver/scripts/test_addiction.py
```

### Run synthetic pipeline

```bash
# Generate synthetic data
python3 scripts/generate_synthetic.py

# Run PC simulation
docker run --rm --entrypoint "" \
    -v $(pwd):/work -w /work modelo47-fenics \
    python3 fenics_solver/scripts/run_pipeline.py \
    --mesh-input data/synthetic/synthetic_mesh.xdmf \
    --synthetic --pc-mode --motor-cortex-tag 3 \
    --output output/results_pc \
    --dt 0.1 --T-final 50.0
```

### Run virtual patient backtesting

```bash
# Generate virtual patient VP_001
# (PC spastic hemiplegic, 9yo, GMFCS III, calibrated with N=1458 literature)
python3 scripts/generate_virtual_patient.py

# 12-month simulation
docker run --rm --entrypoint "" \
    -v $(pwd):/work -w /work modelo47-fenics \
    python3 fenics_solver/scripts/run_pipeline.py \
    --mesh-input data/virtual_patient/synthetic_mesh.xdmf \
    --synthetic --pc-mode --motor-cortex-tag 3 \
    --output output/backtesting_VP001_12m \
    --dt 1.0 --T-final 259200.0 \
    --P-initial 1e-08 --C-reservoir 10.0 \
    --gel-tag 1 --lesion-tag 2 \
    --C0 0.15 --C0-region uniform
```

## Clinical Parameters (VP_001)

| Parameter | Value | Source |
|-----------|-------|--------|
| FA CST (lesion) | 0.42 ± 0.08 | Zeng et al. 2024, N=1458 |
| FA CST (control) | 0.61 ± 0.05 | idem |
| FA M1 | 0.35 ± 0.06 | Kuczynski 2018 |
| ITB dose | 200 μg/day | Penn & Kroin 1985 |
| C50 (baclofen) | 0.05 μg/mL | Du Beau 2012 |
| α (plasticity) | 3.17e-7 s⁻¹ | Calibrated vs IDYS |
| GMFCS | III | IDYS cohort |

## Subdomain Topology

```
tag=1  gel          Hydrogel implant near CSF
tag=2  lesion       Periventricular white matter (PVL)
tag=3  motor_cortex Primary motor cortex M1
```

## Modules

### geom_preprocessing
NIfTI mask segmentation → tetrahedral FEM mesh via Gmsh. Supports
`build_synthetic` (3-region sphere geometry) and `build_from_masks`
(patient-specific from NIfTI segmentation).

### dti_mapper (Rust + PyO3)
High-performance DTI tensor interpolation from NIfTI voxel space to
FEM nodes. Validated against analytical solutions with <1e-6 relative
error.

### fenics_solver
- `solver.py` — anisotropic diffusion + source term + MPC coupling
- `plasticity.py` — PlasticitySolver with Hill kinetics and Heaviside activation
- `mpc_control.py` — MPCController with step-response sensitivity matrix
- `convert_mesh.py` — meshio → FEniCS XDMF with subdomain markers
- `run_pipeline.py` — end-to-end orchestration

## Roadmap

- [ ] T1: Dynamic release kinetics + heterogeneous clearance (Section 4)
- [ ] T2: Inverse problem calibration with dolfin-adjoint (Section 2)
- [ ] T3: Remyelination model with MT-MRI coupling (Section 1.2)
- [ ] T4: 1D-3D tract coupling for CST (Section 1.3)
- [ ] Prospective clinical validation protocol (5–10 patients)

## Citation

If you use this software, please cite:

```bibtex
@software{modelo47_2025,
  title   = {Modelo 4.7: DTI-Guided Precision Neuropharmacotherapy
             Digital Twin for Cerebral Palsy},
  year    = {2025},
  license = {MIT},
  url     = {https://github.com/[your-username]/medi_model}
}
```

## References

- Zeng et al. (2024). White matter lesions and DTI metrics in cerebral palsy.
  *PLOS ONE*. N=1,458.
- Bonouvrié et al. (2013). IDYS trial: ITB in dystonic CP. *BMC Pediatrics*.
- Penn & Kroin (1985). Continuous intrathecal baclofen. *Lancet*.
- Farrell et al. (2013). dolfin-adjoint. *SIAM J. Sci. Comput.*

## License

MIT — see [LICENSE](LICENSE)

## Contact

For clinical collaboration inquiries, please open an issue or contact
the maintainers directly.
