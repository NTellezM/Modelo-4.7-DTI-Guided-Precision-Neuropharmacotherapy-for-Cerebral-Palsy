#!/usr/bin/env python3
"""solver.py — Difusión anisotrópica DTI con permeabilidad acoplada, plasticidad y MPC.

EDP en dominio gel (Ω₁, tag=1):
    ∂C/∂t = ∇·(D_eff·∇C) − k·C + P·(C_res − C)

EDP en dominio tejido (Ω₂, tags≥2):
    ∂C/∂t = ∇·(D_eff·∇C) − k·C

El término P·(C_res − C) modela la liberación controlada del hidrogel:
  - P grande → liberación rápida (el gel se vacía)
  - P pequeño → liberación lenta (el gel retiene)
  - El MPC ajusta P para controlar el craving en el NAc
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("solver")


# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------

@dataclass
class SimulationParams:
    D_isotropic: float = 1.0e-3
    D_scale: float = 1.0
    k_elimination: float = 0.0
    C0: float = 1.0
    C0_region: str = "center"
    C0_radius: float = 2.0
    P_initial: float = 1.0e-8       # permeabilidad inicial del gel
    C_reservoir: float = 10.0        # concentración del reservorio en el gel
    gel_tag: int = 1                 # etiqueta del subdominio gel
    dt: float = 0.05
    T_final: float = 5.0
    theta: float = 1.0
    output_interval: int = 5
    output_dir: str = "results"

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @property
    def n_steps(self):
        return int(np.ceil(self.T_final / self.dt))


# ---------------------------------------------------------------------------
# Carga de malla y tensores (sin cambios)
# ---------------------------------------------------------------------------

def load_mesh_dolfin(mesh_dir):
    import dolfin as df
    mesh_dir = Path(mesh_dir)
    mesh = df.Mesh()
    with df.XDMFFile(str(mesh_dir / "mesh.xdmf")) as f:
        f.read(mesh)
    logger.info("Malla: %d vértices, %d celdas, dim=%d",
                mesh.num_vertices(), mesh.num_cells(), mesh.geometric_dimension())

    subdomains = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    sub_path = mesh_dir / "subdomains.xdmf"
    if sub_path.exists():
        try:
            with df.XDMFFile(str(sub_path)) as f:
                f.read(subdomains, "markers")
            logger.info("  Subdominios: %s", np.unique(subdomains.array()))
        except Exception:
            subdomains.set_all(1)
    else:
        subdomains.set_all(1)
    facets = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    return mesh, subdomains, facets


def create_synthetic_tensor_field(mesh, base_diffusivity=1.0e-3):
    import dolfin as df
    V_tensor = df.TensorFunctionSpace(mesh, "DG", 0)
    D_field = df.Function(V_tensor, name="DiffusionTensor")
    coords = mesh.coordinates()
    z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
    z_range = max(z_max - z_min, 1e-10)
    lambda_ax = 3.0 * base_diffusivity
    lambda_rad = 0.5 * base_diffusivity
    dof_map = V_tensor.dofmap()
    d_vec = D_field.vector().get_local()
    for cell in df.cells(mesh):
        z_norm = (cell.midpoint().z() - z_min) / z_range
        theta = 2.0 * np.pi * z_norm
        e1 = np.array([np.cos(theta), np.sin(theta), 0.0])
        e2 = np.array([-np.sin(theta), np.cos(theta), 0.0])
        e3 = np.array([0.0, 0.0, 1.0])
        D = lambda_ax * np.outer(e1, e1) + lambda_rad * np.outer(e2, e2) + lambda_rad * np.outer(e3, e3)
        d_vec[dof_map.cell_dofs(cell.index())] = D.flatten()
    D_field.vector().set_local(d_vec)
    D_field.vector().apply("insert")
    logger.info("Tensor sintético: λ_ax=%.2e, λ_rad=%.2e", lambda_ax, lambda_rad)
    return D_field


def create_isotropic_tensor_field(mesh, D_iso):
    import dolfin as df
    V_tensor = df.TensorFunctionSpace(mesh, "DG", 0)
    D_field = df.Function(V_tensor, name="DiffusionTensor")
    D_val = D_iso * np.eye(3).flatten()
    dof_map = V_tensor.dofmap()
    d_vec = D_field.vector().get_local()
    for cell in df.cells(mesh):
        d_vec[dof_map.cell_dofs(cell.index())] = D_val
    D_field.vector().set_local(d_vec)
    D_field.vector().apply("insert")
    return D_field


def load_tensor_field(mesh, tensor_path, scale=1.0):
    import dolfin as df
    tensor_path = Path(tensor_path)
    if tensor_path.suffix == ".npy":
        tensors = np.load(str(tensor_path))
    elif tensor_path.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(str(tensor_path), "r") as f:
            tensors = np.array(f["tensors"])
    else:
        raise ValueError(f"Formato no soportado: {tensor_path.suffix}")
    if tensors.ndim == 2 and tensors.shape[1] == 9:
        tensors = tensors.reshape(-1, 3, 3)
    n_t, n_v, n_c = len(tensors), mesh.num_vertices(), mesh.num_cells()
    V_tensor = df.TensorFunctionSpace(mesh, "DG", 0)
    D_field = df.Function(V_tensor, name="DiffusionTensor")
    dof_map = V_tensor.dofmap()
    if n_t == n_v:
        cell_tensors = np.zeros((n_c, 3, 3))
        for cell in df.cells(mesh):
            cell_tensors[cell.index()] = tensors[cell.entities(0)].mean(axis=0)
    elif n_t == n_c:
        cell_tensors = tensors
    else:
        raise ValueError(f"Tensores ({n_t}) ≠ vértices ({n_v}) ni celdas ({n_c})")
    d_vec = D_field.vector().get_local()
    for i in range(n_c):
        d_vec[dof_map.cell_dofs(i)] = (cell_tensors[i] * scale).flatten()
    D_field.vector().set_local(d_vec)
    D_field.vector().apply("insert")
    return D_field


# ---------------------------------------------------------------------------
# FIX #1: Forma variacional con término de permeabilidad P en el gel
# ---------------------------------------------------------------------------

def setup_variational_problem(mesh, D_field, subdomains, params, P_const=None):
    """Configura las formas bilineal y lineal.

    La forma débil incluye el término de liberación del gel:
        P · (C_res − C) · v · dx(gel_tag)

    Esto se descompone en:
        a += dt·θ·P·C·v·dx(gel_tag)           (absorción implícita)
        L += dt·P·C_res·v·dx(gel_tag)          (fuente del reservorio)
        L -= dt·(1-θ)·P·C_n·v·dx(gel_tag)     (parte explícita)

    P_const es un df.Constant mutable: cuando el MPC lo actualiza con
    P_const.assign(new_P), las formas a y L usan automáticamente el nuevo
    valor en el siguiente df.solve() sin re-ensamblar manualmente.
    """
    import dolfin as df

    V = df.FunctionSpace(mesh, "CG", 1)
    C = df.TrialFunction(V)
    v = df.TestFunction(V)
    C_n = df.Function(V, name="C_prev")
    C_sol = df.Function(V, name="C")

    dx = df.Measure("dx", domain=mesh, subdomain_data=subdomains)
    dt_c = df.Constant(params.dt)
    theta_c = df.Constant(params.theta)
    k_c = df.Constant(params.k_elimination)

    D = df.as_tensor([
        [D_field[0], D_field[1], D_field[2]],
        [D_field[3], D_field[4], D_field[5]],
        [D_field[6], D_field[7], D_field[8]],
    ])

    gel = params.gel_tag

    # --- Forma bilineal (lado izquierdo, en C) ---
    a = (
        C * v * dx                                                          # masa
        + dt_c * theta_c * df.inner(D * df.grad(C), df.grad(v)) * dx       # difusión
        + dt_c * k_c * C * v * dx                                          # eliminación
    )

    # --- Forma lineal (lado derecho, en C_n) ---
    L = (
        C_n * v * dx                                                        # masa anterior
        - dt_c * (1.0 - theta_c) * df.inner(D * df.grad(C_n), df.grad(v)) * dx  # difusión explícita
    )

    # --- Término de permeabilidad P (acoplamiento MPC ↔ solver) ---
    if P_const is not None:
        C_res = df.Constant(params.C_reservoir)

        # Parte implícita: P·C en a
        a += dt_c * theta_c * P_const * C * v * dx(gel)

        # Fuente del reservorio + parte explícita de P·C_n
        L += dt_c * P_const * C_res * v * dx(gel)
        L -= dt_c * (1.0 - theta_c) * P_const * C_n * v * dx(gel)

        logger.info("  Permeabilidad P acoplada en dx(%d): P=%.2e, C_res=%.2e",
                    gel, float(P_const), params.C_reservoir)

    # Condición inicial
    coords = mesh.coordinates()
    centroid = coords.mean(axis=0)
    if params.C0_region == "center":
        class GaussIC(df.UserExpression):
            def eval(self, value, x):
                r2 = sum((xi - ci) ** 2 for xi, ci in zip(x, centroid))
                value[0] = params.C0 * np.exp(-r2 / (2.0 * params.C0_radius ** 2))
            def value_shape(self):
                return ()
        C_n.interpolate(GaussIC(degree=1))
    else:
        C_n.interpolate(df.Constant(params.C0))

    C_sol.assign(C_n)
    mass0 = df.assemble(C_n * dx)
    logger.info("Condición inicial '%s': masa=%.4e", params.C0_region, mass0)

    return V, C_n, C_sol, a, L, dx


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

@dataclass
class StepMetrics:
    step: int
    time: float
    mass: float
    c_max: float
    c_min: float
    solve_ms: float
    rho_mean: float = 1.0
    P_current: float = 0.0
    craving: float = 0.0


def save_metrics_csv(metrics, path):
    path = Path(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "time", "mass", "c_max", "c_min", "solve_ms",
                     "rho_mean", "P_current", "craving"])
        for m in metrics:
            w.writerow([m.step, f"{m.time:.6f}", f"{m.mass:.8e}",
                        f"{m.c_max:.8e}", f"{m.c_min:.8e}", f"{m.solve_ms:.2f}",
                        f"{m.rho_mean:.6f}", f"{m.P_current:.4e}", f"{m.craving:.6e}"])
    logger.info("Métricas: %s (%d entradas)", path, len(metrics))


# ---------------------------------------------------------------------------
# Simulación normal
# ---------------------------------------------------------------------------

def run_simulation(mesh, D_field, subdomains, facets, params):
    import dolfin as df
    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    V, C_n, C_sol, a, L, dx = setup_variational_problem(mesh, D_field, subdomains, params)

    xdmf = df.XDMFFile(str(output_dir / "concentration.xdmf"))
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(C_sol, 0.0)

    metrics = []
    n_steps = params.n_steps
    logger.info("=" * 60)
    logger.info("Simulación: %d pasos, dt=%.4f, T=%.2f", n_steps, params.dt, params.T_final)
    logger.info("=" * 60)
    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_now = step * params.dt
        ts = time.perf_counter()
        df.solve(a == L, C_sol)
        solve_ms = (time.perf_counter() - ts) * 1000.0
        C_n.assign(C_sol)
        arr = C_sol.vector().get_local()
        m = StepMetrics(step=step, time=t_now, mass=df.assemble(C_sol * dx),
                        c_max=float(arr.max()), c_min=float(arr.min()), solve_ms=solve_ms)
        metrics.append(m)
        if step % params.output_interval == 0 or step == n_steps:
            xdmf.write(C_sol, t_now)
        if step % max(1, n_steps // 20) == 0 or step == 1 or step == n_steps:
            logger.info("  %4d/%d | t=%.4f | masa=%.4e | C=[%.3e,%.3e] | %.1fms",
                        step, n_steps, t_now, m.mass, m.c_min, m.c_max, solve_ms)

    xdmf.close()
    elapsed = time.perf_counter() - t0
    logger.info("Completado en %.2fs (%.1f ms/paso)", elapsed, 1000 * elapsed / n_steps)
    return metrics


# ---------------------------------------------------------------------------
# Craving y respuesta al escalón
# ---------------------------------------------------------------------------

def compute_craving(C_sol, dx_nac, vol_nac, C50=0.2, n_hill=2):
    """Ocupación promedio de receptores (Hill) en NAc: R = C^n / (C50^n + C^n)."""
    import dolfin as df
    eps = 1e-16
    C_safe = C_sol + eps
    if n_hill == 1:
        R_expr = C_safe / (C50 + C_safe)
    elif n_hill == 2:
        R_expr = C_safe * C_safe / (C50 ** 2 + C_safe * C_safe)
    else:
        Cn = C_safe ** n_hill
        R_expr = Cn / (C50 ** n_hill + Cn)
    return df.assemble(R_expr * dx_nac) / max(vol_nac, 1e-30)


def run_step_response(mesh, D_DTI, subdomains, facets, params,
                      base_P, delta_P, dt_control, horizon, nac_tag=3,
                      C50=0.2, n_hill=2):
    """Ejecuta una mini-simulación con un escalón en P y mide la respuesta del craving.

    FIX #5: Usa el solver real (no una curva sintética) para construir S.

    Returns:
        q_response: array (horizon,) — craving en cada instante de control.
    """
    import dolfin as df
    import tempfile

    logger.info("Ejecutando step response: base_P=%.2e, delta_P=%.2e, horizon=%d",
                base_P, delta_P, horizon)

    # Crear una copia de params para la mini-simulación
    tmp_dir = tempfile.mkdtemp(prefix="step_response_")
    mini_params = SimulationParams(
        D_isotropic=params.D_isotropic, D_scale=params.D_scale,
        k_elimination=params.k_elimination,
        C0=params.C0, C0_region=params.C0_region, C0_radius=params.C0_radius,
        P_initial=base_P + delta_P,  # escalón aplicado desde t=0
        C_reservoir=params.C_reservoir, gel_tag=params.gel_tag,
        dt=params.dt, T_final=horizon * dt_control, theta=params.theta,
        output_interval=999999,  # no guardar archivos
        output_dir=tmp_dir,
    )

    # P constante = base_P + delta_P durante toda la mini-simulación
    P_const = df.Constant(base_P + delta_P)

    V, C_n, C_sol, a, L, dx = setup_variational_problem(
        mesh, D_DTI, subdomains, mini_params, P_const=P_const,
    )

    # Medida para NAc
    dx_nac = df.Measure("dx", domain=mesh, subdomain_data=subdomains)(nac_tag)
    vol_nac = df.assemble(df.Constant(1.0) * dx_nac)

    # Bucle temporal, muestreando craving en cada intervalo de control
    q_response = []
    n_total = mini_params.n_steps
    steps_per_control = max(1, int(dt_control / params.dt))

    for step in range(1, n_total + 1):
        df.solve(a == L, C_sol)
        C_n.assign(C_sol)

        if step % steps_per_control == 0:
            q = compute_craving(C_sol, dx_nac, vol_nac, C50, n_hill)
            q_response.append(q)

        if len(q_response) >= horizon:
            break

    # Rellenar si faltan puntos
    while len(q_response) < horizon:
        q_response.append(q_response[-1] if q_response else 0.0)

    q_arr = np.array(q_response[:horizon])
    logger.info("  Step response: q_range=[%.4e, %.4e]", q_arr.min(), q_arr.max())
    return q_arr


# ---------------------------------------------------------------------------
# Simulación con plasticidad + MPC (FIX #1: P acoplado)
# ---------------------------------------------------------------------------

def run_simulation_addiction(mesh, D_DTI, subdomains, facets, params,
                             plasticity_solver, controller,
                             nac_tag=3, dt_control=3600.0, C50=0.2, n_hill=2):
    """Bucle temporal con P acoplado a la forma débil, plasticidad y MPC."""
    import dolfin as df

    output_dir = Path(params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # P como df.Constant mutable — las formas a y L lo referencian directamente
    P_const = df.Constant(params.P_initial)
    D_field = plasticity_solver.D_eff

    V, C_n, C_sol, a, L, dx = setup_variational_problem(
        mesh, D_field, subdomains, params, P_const=P_const,
    )

    xdmf = df.XDMFFile(str(output_dir / "concentration.xdmf"))
    xdmf.parameters["flush_output"] = True
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.write(C_sol, 0.0)

    dx_nac = df.Measure("dx", domain=mesh, subdomain_data=subdomains)(nac_tag)
    vol_nac = df.assemble(df.Constant(1.0) * dx_nac)
    logger.info("NAc (tag=%d): volumen=%.4e", nac_tag, vol_nac)

    metrics = []
    n_steps = params.n_steps
    control_every = max(1, int(dt_control / params.dt))
    plastic_every = max(1, int(plasticity_solver.dt_plastic / params.dt))

    # FIX #3: acumular R cada N pasos en vez de cada paso
    R_accum_every = max(1, min(control_every, 100))

    dt_accum_plastic = 0.0
    current_P = params.P_initial
    craving = 0.0

    logger.info("=" * 60)
    logger.info("Simulación ADICCIÓN: %d pasos, dt=%.4f, T=%.2f", n_steps, params.dt, params.T_final)
    logger.info("  P inicial: %.2e, C_reservoir: %.2e, gel_tag: %d",
                params.P_initial, params.C_reservoir, params.gel_tag)
    logger.info("  Control MPC cada %d pasos (%.0f s)", control_every, dt_control)
    logger.info("  Plasticidad cada %d pasos (%.0f s)", plastic_every, plasticity_solver.dt_plastic)
    logger.info("  R acumulación cada %d pasos", R_accum_every)
    logger.info("=" * 60)

    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        t_now = step * params.dt

        ts = time.perf_counter()
        df.solve(a == L, C_sol)
        solve_ms = (time.perf_counter() - ts) * 1000.0
        C_n.assign(C_sol)

        # FIX #3: acumular R con menor frecuencia
        if step % R_accum_every == 0:
            plasticity_solver.accumulate_R(C_sol, params.dt * R_accum_every)
        dt_accum_plastic += params.dt

        # Control MPC: actualizar P
        if step % control_every == 0 and vol_nac > 1e-30:
            craving = compute_craving(C_sol, dx_nac, vol_nac, C50, n_hill)
            current_P = controller.update(craving, current_P)
            # FIX #1: P_const.assign() propaga el cambio a las formas a y L
            P_const.assign(current_P)

        # Plasticidad
        if step % plastic_every == 0:
            plasticity_solver.update_plasticity(dt_accum_plastic)
            dt_accum_plastic = 0.0

        # Métricas
        arr = C_sol.vector().get_local()
        rho_arr = plasticity_solver.rho.vector().get_local()
        m = StepMetrics(
            step=step, time=t_now, mass=df.assemble(C_sol * dx),
            c_max=float(arr.max()), c_min=float(arr.min()), solve_ms=solve_ms,
            rho_mean=float(rho_arr.mean()), P_current=current_P, craving=craving,
        )
        metrics.append(m)

        if step % params.output_interval == 0 or step == n_steps:
            xdmf.write(C_sol, t_now)

        if step % max(1, n_steps // 20) == 0 or step == 1 or step == n_steps:
            logger.info("  %4d/%d | t=%.1f | masa=%.3e | ρ=%.4f | P=%.2e | q=%.4f | %.1fms",
                        step, n_steps, t_now, m.mass, m.rho_mean,
                        current_P, craving, solve_ms)

    xdmf.close()
    elapsed = time.perf_counter() - t0
    # Reporte por región
    try:
        dx_sub = df.Measure("dx", domain=mesh, subdomain_data=subdomains)
        for tag, name in [(1,"gel"),(2,"lesion"),(3,"M1")]:
            vol = df.assemble(df.Constant(1.0)*dx_sub(tag))
            if vol > 0:
                rho_tag = df.assemble(plasticity.rho*dx_sub(tag)) / vol
                logger.info("  rho_%s (tag=%d): %.6f | vol=%.3e", name, tag, rho_tag, vol)
    except Exception as e:
        logger.warning("Reporte por region: %s", e)
    logger.info("Completado en %.2fs", elapsed)

    with df.XDMFFile(str(output_dir / "rho.xdmf")) as f:
        f.write(plasticity_solver.rho, params.T_final)

    return metrics


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------

def print_summary(metrics, params):
    if not metrics:
        return
    first, last = metrics[0], metrics[-1]
    mass_change = (last.mass - first.mass) / max(abs(first.mass), 1e-30)
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Pasos: {len(metrics)}")
    print(f"  Tiempo: [0, {last.time:.2f}] s")
    print(f"  Masa: {first.mass:.4e} → {last.mass:.4e} ({mass_change:+.4%})")
    print(f"  C_max final: {last.c_max:.4e}")
    print(f"  ρ medio final: {last.rho_mean:.4f}")
    print(f"  P final: {last.P_current:.4e}")
    print(f"  Craving final: {last.craving:.4e}")
    print(f"  Solve medio: {np.mean([m.solve_ms for m in metrics]):.1f} ms/paso")
    print(f"  Resultados: {params.output_dir}/")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Solver de difusión anisotrópica DTI")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--tensors", type=str, default=None)
    parser.add_argument("--params", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--T-final", type=float, default=None)
    parser.add_argument("--C0", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None)
    args = parser.parse_args()

    params = SimulationParams.from_json(args.params) if args.params else SimulationParams()
    if args.dt: params.dt = args.dt
    if args.T_final: params.T_final = args.T_final
    if args.C0: params.C0 = args.C0
    if args.theta: params.theta = args.theta
    params.output_dir = args.output

    Path(params.output_dir).mkdir(parents=True, exist_ok=True)
    params.to_json(str(Path(params.output_dir) / "params_used.json"))

    mesh, subdomains, facets = load_mesh_dolfin(args.mesh)
    if args.tensors:
        D_field = load_tensor_field(mesh, args.tensors, scale=params.D_scale)
    elif args.synthetic:
        D_field = create_synthetic_tensor_field(mesh, params.D_isotropic)
    else:
        D_field = create_isotropic_tensor_field(mesh, params.D_isotropic)

    metrics = run_simulation(mesh, D_field, subdomains, facets, params)
    save_metrics_csv(metrics, Path(params.output_dir) / "metrics.csv")
    print_summary(metrics, params)


if __name__ == "__main__":
    main()
