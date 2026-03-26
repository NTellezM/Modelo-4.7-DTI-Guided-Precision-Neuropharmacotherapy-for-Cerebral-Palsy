#!/usr/bin/env python3
"""run_pipeline_patient.py — Orquesta el pipeline completo desde configuración JSON.

Uso:
    python scripts/run_pipeline_patient.py --config config/pipeline_config.json
    python scripts/run_pipeline_patient.py --config config/pipeline_config.json --skip-mesh --skip-tensors
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")


def run(cmd, name):
    logger.info("─── %s ───", name)
    logger.info("$ %s", " ".join(cmd))
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        logger.error("FALLO en %s (%.1fs):\n%s", name, elapsed, result.stderr[-500:])
        sys.exit(1)
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n")[-5:]:
            logger.info("  %s", line)
    logger.info("  OK (%.1fs)", elapsed)


def main():
    parser = argparse.ArgumentParser(description="Pipeline Modelo 4.7")
    parser.add_argument("--config", required=True)
    parser.add_argument("--patient-id", default=None, help="ID del paciente (reemplaza rutas)")
    parser.add_argument("--skip-mesh", action="store_true")
    parser.add_argument("--skip-tensors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # Override paths for specific patient
    if args.patient_id:
        pid = args.patient_id
        cfg["gel_mask"] = f"data/patients/{pid}/gel.nii.gz"
        cfg["tissue_mask"] = f"data/patients/{pid}/tissue.nii.gz"
        cfg["nac_mask"] = f"data/patients/{pid}/nac.nii.gz"
        cfg["dti_file"] = f"data/patients/{pid}/dti.nii.gz"
        cfg["results_dir"] = f"output/results/{pid}"

    for d in [cfg["mesh_dir"], cfg.get("tensor_dir", "output/tensors"), cfg["results_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("═" * 60)
    logger.info("Pipeline Modelo 4.7")
    logger.info("═" * 60)
    t_start = time.perf_counter()

    # Paso 1: Malla
    if not args.skip_mesh:
        cmd = [
            "python", "-m", "geom_preprocessing.cli", "build",
            "--gel", cfg["gel_mask"],
            "--tissue", cfg["tissue_mask"],
            "--output", cfg["mesh_dir"],
            "--refine", str(cfg.get("refine", 0.2)),
            "--coarse", str(cfg.get("coarse", 1.5)),
        ]
        if cfg.get("nac_mask"):
            cmd.extend(["--nac", cfg["nac_mask"]])
        if cfg.get("smooth"):
            cmd.append("--smooth")
        if not args.dry_run:
            run(cmd, "Generación de malla")
        else:
            logger.info("[DRY] %s", " ".join(cmd))

    # Paso 2: Tensores DTI
    if not args.skip_tensors:
        tensor_dir = cfg.get("tensor_dir", "output/tensors")
        cmd = [
            "python", "-m", "dti_mapper.cli", "map",
            "--dti", cfg["dti_file"],
            "--mesh", str(Path(cfg["mesh_dir"]) / "mesh.xdmf"),
            "--output", str(Path(tensor_dir) / "tensors.h5"),
            "--validate",
        ]
        if not args.dry_run:
            run(cmd, "Mapeo de tensores DTI")
        else:
            logger.info("[DRY] %s", " ".join(cmd))

    # Paso 3: Simulación FEniCS
    sim = cfg.get("simulation", {})
    addiction = cfg.get("addiction", {})

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{Path.cwd()}:/work", "-w", "/work",
        "modelo47-fenics",
        "python3", "fenics_solver/scripts/run_pipeline.py",
        "--mesh-input", str(Path(cfg["mesh_dir"]) / "mesh.xdmf")
             if not args.skip_mesh else "data/synthetic/synthetic_mesh.xdmf",
        "--synthetic",
        "--output", cfg["results_dir"],
        "--dt", str(sim.get("dt", 0.05)),
        "--T-final", str(sim.get("T_final", 5.0)),
        "--C0", str(sim.get("C0", 1.0)),
        "--theta", str(sim.get("theta", 1.0)),
    ]

    if addiction.get("enabled"):
        cmd.extend(["--addiction", "--nac-tag", str(addiction.get("nac_tag", 3))])

    if not args.dry_run:
        run(cmd, "Simulación FEniCS")
    else:
        logger.info("[DRY] %s", " ".join(cmd))

    elapsed = time.perf_counter() - t_start
    logger.info("═" * 60)
    logger.info("Pipeline completo en %.1fs → %s", elapsed, cfg["results_dir"])
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
