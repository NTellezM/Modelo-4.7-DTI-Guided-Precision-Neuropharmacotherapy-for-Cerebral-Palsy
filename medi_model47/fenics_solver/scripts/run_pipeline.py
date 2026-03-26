#!/usr/bin/env python3
"""run_pipeline.py — Pipeline: malla → tensores → simulación (normal o adicción).

FIX #5: La matriz S del MPC se construye ejecutando run_step_response()
        con el solver real, no con una curva analítica artificial.

Uso:
  Normal:    --synthetic --output results/
  Adicción:  --synthetic --addiction --output results_addiction/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("pipeline")


def step1_convert_mesh(mesh_input, work_dir):
    logger.info("═" * 60)
    logger.info("PASO 1: Conversión de malla")
    logger.info("═" * 60)
    from convert_mesh import convert_mesh
    mesh_dir = work_dir / "mesh_fenics"
    convert_mesh(mesh_input, mesh_dir)
    return mesh_dir


def step3_run_solver(mesh_dir, tensor_path, output_dir, params_override,
                     synthetic=False, addiction=False, nac_tag=3):
    logger.info("═" * 60)
    logger.info("PASO 3: Simulación FEniCS%s", " (ADICCIÓN)" if addiction else "")
    logger.info("═" * 60)

    try:
        import dolfin  # noqa: F401
    except ImportError:
        logger.error("FEniCS no disponible. Usa Docker.")
        sys.exit(1)

    from solver import (
        SimulationParams, load_mesh_dolfin,
        create_synthetic_tensor_field, create_isotropic_tensor_field,
        load_tensor_field, run_simulation, run_simulation_addiction,
        run_step_response, save_metrics_csv, print_summary,
    )

    params = SimulationParams(**params_override)
    params.output_dir = str(output_dir)
    Path(params.output_dir).mkdir(parents=True, exist_ok=True)
    params.to_json(str(output_dir / "params_used.json"))

    mesh, subdomains, facets = load_mesh_dolfin(str(mesh_dir))

    if tensor_path and Path(tensor_path).exists():
        D_DTI = load_tensor_field(mesh, tensor_path, scale=params.D_scale)
    elif synthetic:
        D_DTI = create_synthetic_tensor_field(mesh, base_diffusivity=params.D_isotropic)
    else:
        D_DTI = create_isotropic_tensor_field(mesh, params.D_isotropic)

    if addiction:
        _run_addiction_mode(mesh, subdomains, facets, D_DTI, params, nac_tag)
    else:
        metrics = run_simulation(mesh, D_DTI, subdomains, facets, params)
        save_metrics_csv(metrics, output_dir / "metrics.csv")
        print_summary(metrics, params)


def _run_addiction_mode(mesh, subdomains, facets, D_DTI, params, nac_tag):
    """FIX #5: Construye S desde una simulación real, no desde una curva analítica."""
    from plasticity import PlasticitySolver
    from mpc_control import MPCController
    from solver import (
        run_simulation_addiction, run_step_response,
        save_metrics_csv, print_summary,
    )

    output_dir = Path(params.output_dir)

    # Parámetros de control
    dt_control = 3600.0
    dt_plastic = 86400.0
    C50, n_hill = 0.2, 2
    horizon = 10

    # --- Plasticidad ---
    logger.info("Inicializando PlasticitySolver...")
    ps = PlasticitySolver(
        mesh, subdomains, D_DTI,
        alpha=1e-7, beta=1e-8, R_th=0.5, C50=C50,
        n_hill=n_hill, dt_plastic=dt_plastic,
        plastic_region_tag=nac_tag,
    )
    logger.info("  ρ inicial: mean=%.4f, vol_plastic=%.4e",
                ps.rho.vector().get_local().mean(), ps.vol_plastic)

    # --- FIX #5: Precomputar S ejecutando el solver real ---
    logger.info("Precomputando matriz de sensibilidad S (simulación real)...")
    base_P = params.P_initial
    delta_P = base_P * 10.0  # escalón de 10× la permeabilidad base

    q_response = run_step_response(
        mesh, D_DTI, subdomains, facets, params,
        base_P=base_P, delta_P=delta_P,
        dt_control=dt_control, horizon=horizon,
        nac_tag=nac_tag, C50=C50, n_hill=n_hill,
    )

    # Normalizar: S tiene unidades de ganancia [Δcraving / ΔP].
    # q_response es adimensional (Hill ∈ [0,1]), delta_P tiene unidades de P (m/s o 1/s
    # según la formulación). La división produce la sensibilidad del craving a cambios
    # unitarios de permeabilidad, que es lo que el QP del MPC necesita para predecir
    # el efecto de cada Δu_i sobre la salida futura.
    if delta_P > 0:
        q_response = q_response / delta_P

    # --- Construir controlador MPC con S real ---
    controller = MPCController(
        S=np.zeros((horizon, horizon)),  # placeholder, se llena abajo
        q_target=0.2, P_min=1e-9, P_max=1e-6,
        delta_P_max=1e-7, horizon=horizon,
        dt_control=dt_control, lambda_reg=0.01,
    )
    controller.build_toeplitz_matrix(q_response)
    controller.P_history.append(params.P_initial)

    logger.info("  S construida: shape=%s, norma=%.4e",
                controller.S.shape, np.linalg.norm(controller.S))

    # --- Simulación acoplada ---
    metrics = run_simulation_addiction(
        mesh, D_DTI, subdomains, facets, params,
        plasticity_solver=ps, controller=controller,
        nac_tag=nac_tag, dt_control=dt_control,
        C50=C50, n_hill=n_hill,
    )

    save_metrics_csv(metrics, output_dir / "metrics.csv")
    print_summary(metrics, params)

    np.savetxt(str(output_dir / "P_history.csv"), controller.P_history,
               delimiter=",", header="P", comments="")
    np.savetxt(str(output_dir / "q_history.csv"), controller.q_history,
               delimiter=",", header="craving", comments="")
    np.savetxt(str(output_dir / "step_response.csv"), q_response,
               delimiter=",", header="q_response", comments="")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Modelo 4.7")
    parser.add_argument("--mesh-input", required=True)
    parser.add_argument("--dti", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--addiction", action="store_true")
    parser.add_argument("--nac-tag", type=int, default=3)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--T-final", type=float, default=5.0)
    parser.add_argument("--C0", type=float, default=1.0)
    parser.add_argument("--C0-region", type=str, default=None, dest="C0_region")
    parser.add_argument("--D-iso", type=float, default=1.0e-3)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--P-initial", type=float, default=1e-8)
    parser.add_argument("--C-reservoir", type=float, default=10.0)
    parser.add_argument("--gel-tag", type=int, default=1)
    args = parser.parse_args()

    if not args.dti and not args.synthetic:
        parser.error("Especifica --dti <archivo> o --synthetic")

    work_dir = Path(args.output)
    work_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    mesh_dir = step1_convert_mesh(args.mesh_input, work_dir)

    tensor_path = None
    if args.dti:
        # dti_mapper integration would go here
        pass

    params_override = {
        "dt": args.dt, "T_final": args.T_final, "C0": args.C0,
        "D_isotropic": args.D_iso, "theta": args.theta,
        "P_initial": args.P_initial, "C_reservoir": args.C_reservoir,
        "gel_tag": args.gel_tag,
    }

    step3_run_solver(mesh_dir, tensor_path, work_dir, params_override,
                     synthetic=args.synthetic, addiction=args.addiction,
                     nac_tag=args.nac_tag)

    logger.info("Pipeline completo en %.1fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
