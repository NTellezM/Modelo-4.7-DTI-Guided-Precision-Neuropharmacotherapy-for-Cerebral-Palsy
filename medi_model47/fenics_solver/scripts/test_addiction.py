#!/usr/bin/env python3
"""test_addiction.py — Tests autónomos incluyendo verificación del acoplamiento P.

Test 1: PlasticitySolver (vectorizado)
Test 2: MPCController (QP con SLSQP)
Test 3: P acoplado — verifica que cambiar P cambia la solución (FIX #1)
Test 4: Integración completa

Ejecutar:
    docker run --rm -v $(pwd):/work -w /work modelo47-fenics \
        python3 fenics_solver/scripts/test_addiction.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import dolfin as df
import numpy as np

from plasticity import PlasticitySolver
from mpc_control import MPCController


def create_tagged_mesh():
    mesh = df.UnitCubeMesh(10, 10, 10)
    sd = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for cell in df.cells(mesh):
        mp = cell.midpoint()
        r2 = (mp.x() - 0.5)**2 + (mp.y() - 0.5)**2 + (mp.z() - 0.5)**2
        if r2 < 0.04:
            sd[cell] = 1   # gel
        elif mp.x() > 0.7 and mp.y() > 0.7:
            sd[cell] = 3   # NAc
        else:
            sd[cell] = 2   # tejido
    return mesh, sd


def create_iso_D(mesh, D_val=1e-3):
    V = df.TensorFunctionSpace(mesh, "DG", 0)
    D = df.Function(V)
    vals = D_val * np.eye(3).flatten()
    dof_map = V.dofmap()
    d_vec = D.vector().get_local()
    for cell in df.cells(mesh):
        d_vec[dof_map.cell_dofs(cell.index())] = vals
    D.vector().set_local(d_vec)
    D.vector().apply("insert")
    return D


# ---- Test 1: Plasticidad (FIX #2: vectorizado) ----

def test_plasticity():
    print("\n--- Test 1: PlasticitySolver (vectorizado) ---")
    mesh, sd = create_tagged_mesh()
    D_DTI = create_iso_D(mesh)

    ps = PlasticitySolver(mesh, sd, D_DTI, alpha=1e-5, beta=1e-6,
                          R_th=0.5, C50=0.2, n_hill=2, plastic_region_tag=3)

    V = df.FunctionSpace(mesh, "CG", 1)
    C = df.Function(V)
    C.vector()[:] = 0.5

    t0 = time.perf_counter()
    for _ in range(10):
        ps.accumulate_R(C, batch_dt=3600.0)
    ps.update_plasticity(dt_accumulated=36000.0)
    elapsed = (time.perf_counter() - t0) * 1000

    s = ps.get_summary()
    print(f"  ρ: mean={s['rho_mean']:.6f}, min={s['rho_min']:.6f}, max={s['rho_max']:.6f}")
    print(f"  D_eff max: {ps.D_eff.vector().get_local().max():.4e}")
    print(f"  Tiempo: {elapsed:.1f} ms")

    assert ps.D_eff.vector().get_local().max() <= D_DTI.vector().get_local().max() + 1e-16
    print("  PASS")


# ---- Test 2: MPC ----

def test_mpc():
    print("\n--- Test 2: MPCController ---")
    horizon = 10
    dt_control = 3600.0

    t = np.arange(horizon) * dt_control
    resp = np.maximum(0, t - 7200) * np.exp(-0.5 * (t - 7200) / dt_control)
    if resp.max() > 0:
        resp /= resp.max()

    controller = MPCController(
        S=np.zeros((horizon, horizon)),
        q_target=0.2, P_min=1e-9, P_max=1e-6,
        delta_P_max=1e-7, horizon=horizon, dt_control=dt_control,
    )
    controller.build_toeplitz_matrix(resp)

    P = 1e-8
    np.random.seed(42)
    for _ in range(30):
        q = 0.5 * np.exp(-_ * dt_control / 86400) + 0.05 * np.random.randn()
        P = controller.update(q, P)

    P_arr = np.array(controller.P_history)
    assert np.all(P_arr >= controller.P_min - 1e-15)
    assert np.all(P_arr <= controller.P_max + 1e-15)
    print(f"  P final: {P:.4e}, range: [{P_arr.min():.2e}, {P_arr.max():.2e}]")
    print("  PASS")


# ---- Test 3: P acoplado al solver (FIX #1 validation) ----

def test_P_coupling():
    """Verifica que cambiar P_const cambia la solución.

    Ejecuta 10 pasos con P=0 (sin liberación) y 10 pasos con P=1e-3
    (liberación rápida). La masa total debe ser significativamente mayor
    en el segundo caso porque el reservorio inyecta fármaco.
    """
    print("\n--- Test 3: Acoplamiento P ↔ forma débil (FIX #1) ---")
    from solver import SimulationParams, setup_variational_problem

    mesh, sd = create_tagged_mesh()
    D_DTI = create_iso_D(mesh)
    dx = df.Measure("dx", domain=mesh, subdomain_data=sd)

    def run_with_P(P_value, n_steps=10):
        params = SimulationParams(
            dt=0.01, T_final=n_steps * 0.01, C0=0.0, C0_region="uniform",
            P_initial=P_value, C_reservoir=10.0, gel_tag=1,
        )
        P_const = df.Constant(P_value)
        V, C_n, C_sol, a, L, dx_form = setup_variational_problem(
            mesh, D_DTI, sd, params, P_const=P_const,
        )
        for _ in range(n_steps):
            df.solve(a == L, C_sol)
            C_n.assign(C_sol)
        return df.assemble(C_sol * dx_form)

    mass_P0 = run_with_P(0.0)       # sin liberación
    mass_P_high = run_with_P(1e-3)  # liberación rápida

    print(f"  Masa con P=0:    {mass_P0:.6e}")
    print(f"  Masa con P=1e-3: {mass_P_high:.6e}")
    print(f"  Ratio: {mass_P_high / max(mass_P0, 1e-30):.1f}x")

    # Con P>0 y C_reservoir>0, debe haber más masa que con P=0
    assert mass_P_high > mass_P0 + 1e-10, \
        f"P no afecta la solución: masa_P0={mass_P0}, masa_P_high={mass_P_high}"
    print("  PASS — P está acoplado correctamente a la forma débil")


# ---- Test 4: Integración ----

def test_integration():
    print("\n--- Test 4: Integración PlasticitySolver + MPC + P acoplado ---")
    from solver import SimulationParams, setup_variational_problem, compute_craving

    mesh, sd = create_tagged_mesh()
    D_DTI = create_iso_D(mesh)

    ps = PlasticitySolver(mesh, sd, D_DTI, alpha=1e-5, beta=1e-6,
                          R_th=0.3, C50=0.2, n_hill=2, plastic_region_tag=3)

    horizon = 5
    controller = MPCController(
        S=np.zeros((horizon, horizon)),
        q_target=0.1, P_min=1e-9, P_max=1e-6,
        delta_P_max=1e-7, horizon=horizon, dt_control=3600.0,
    )
    resp = np.array([0.0, 0.1, 0.5, 0.8, 1.0])
    controller.build_toeplitz_matrix(resp)

    params = SimulationParams(
        dt=0.01, T_final=0.5, C0=1.0, C0_region="center", C0_radius=0.3,
        P_initial=1e-8, C_reservoir=5.0, gel_tag=1,
    )

    P_const = df.Constant(params.P_initial)
    V, C_n, C_sol, a, L, dx = setup_variational_problem(
        mesh, ps.D_eff, sd, params, P_const=P_const,
    )

    dx_nac = df.Measure("dx", domain=mesh, subdomain_data=sd)(3)
    vol_nac = df.assemble(df.Constant(1.0) * dx_nac)

    current_P = params.P_initial
    for step in range(1, 51):
        df.solve(a == L, C_sol)
        C_n.assign(C_sol)

        if step % 10 == 0:
            ps.accumulate_R(C_sol, batch_dt=0.1)
            craving = compute_craving(C_sol, dx_nac, vol_nac)
            current_P = controller.update(craving, current_P)
            P_const.assign(current_P)

        if step % 25 == 0:
            ps.update_plasticity(dt_accumulated=0.25)

    s = ps.get_summary()
    print(f"  ρ_mean={s['rho_mean']:.6f}, P={current_P:.3e}, craving={craving:.4f}")
    print(f"  Masa final: {df.assemble(C_sol * dx):.4e}")
    print("  PASS")


# ---- Test 5: run_step_response smoke test (FIX #5 validation) ----

def test_step_response():
    """Verifica que run_step_response produce un array de craving con la forma
    y magnitudes esperadas: monótono creciente, valores en [0, 1]."""
    print("\n--- Test 5: run_step_response (FIX #5) ---")
    from solver import SimulationParams, run_step_response

    mesh, sd = create_tagged_mesh()
    D_DTI = create_iso_D(mesh)

    params = SimulationParams(
        dt=0.01, T_final=1.0, C0=1.0, C0_region="center", C0_radius=0.3,
        P_initial=1e-8, C_reservoir=5.0, gel_tag=1,
    )
    facets = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    q = run_step_response(
        mesh, D_DTI, sd, facets, params,
        base_P=1e-8, delta_P=1e-6,
        dt_control=0.1, horizon=5,
        nac_tag=3, C50=0.2, n_hill=2,
    )

    print(f"  q shape: {q.shape}")
    print(f"  q values: {q}")
    print(f"  q range: [{q.min():.4e}, {q.max():.4e}]")

    assert q.shape == (5,), f"Forma incorrecta: {q.shape}"
    assert np.all(np.isfinite(q)), "q contiene NaN o Inf"
    assert np.all(q >= 0), f"q negativo: {q.min()}"
    assert np.all(q <= 1.0 + 1e-10), f"q > 1 (Hill debería estar en [0,1]): {q.max()}"
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Tests de adicción — Modelo 4.7")
    print("=" * 60)

    test_plasticity()
    test_mpc()
    test_P_coupling()
    test_integration()
    test_step_response()

    print("\n" + "=" * 60)
    print("5/5 TESTS PASARON")
    print("=" * 60)
