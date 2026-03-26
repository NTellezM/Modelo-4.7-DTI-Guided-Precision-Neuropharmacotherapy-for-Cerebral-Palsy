"""mpc_control.py — Control Predictivo por Modelo (MPC) para permeabilidad adaptativa.

Usa una matriz de sensibilidad lineal S (Toeplitz) para predecir la
respuesta del craving a cambios en la permeabilidad P del hidrogel.

Resuelve en cada paso de control:
    min_{Δu}  ||e₀·1 + S·Δu||² + λ||Δu||²
    s.t.      P_min ≤ P_k + Σ Δu_i ≤ P_max
              |Δu_i| ≤ δP_max

Usa scipy.optimize.minimize (SLSQP) como solver QP portable.
"""

import numpy as np
import scipy.optimize as opt


class MPCController:
    """Controlador MPC basado en matriz de sensibilidad."""

    def __init__(self, S, q_target, P_min, P_max,
                 delta_P_max=None, horizon=None, dt_control=None,
                 lambda_reg=0.01):
        self.S = np.asarray(S, dtype=np.float64)
        self.q_target = q_target
        self.P_min = P_min
        self.P_max = P_max
        self.delta_P_max = delta_P_max
        self.horizon = horizon or self.S.shape[0]
        self.n = self.horizon
        self.dt_control = dt_control
        self.lambda_reg = lambda_reg

        self.P_history = []
        self.q_history = []

    def update(self, current_q, current_P):
        """Calcula la siguiente permeabilidad óptima.

        Parameters
        ----------
        current_q : float
            Craving medido en el instante actual.
        current_P : float
            Permeabilidad del paso anterior.

        Returns
        -------
        new_P : float
            Permeabilidad a aplicar.
        """
        n = self.n
        e0 = current_q - self.q_target

        # Matriz de coste cuadrático
        # Minimizar ||e0·1 + S·du||² + λ||du||²
        # = du^T (S^T S + λI) du + 2·e0·1^T·S·du + cte
        H = self.S.T @ self.S + self.lambda_reg * np.eye(n)
        # e0 es escalar → broadcast a vector de unos
        e0_vec = e0 * np.ones(n)
        g = 2.0 * (self.S.T @ e0_vec)

        # Restricciones: P_min ≤ current_P + cumsum(du) ≤ P_max
        cumsum_mat = np.tril(np.ones((n, n)))

        A_ub_list = [
            -cumsum_mat,  # -cumsum(du) ≤ current_P - P_min
            cumsum_mat,   #  cumsum(du) ≤ P_max - current_P
        ]
        b_ub_list = [
            (current_P - self.P_min) * np.ones(n),
            (self.P_max - current_P) * np.ones(n),
        ]

        if self.delta_P_max is not None:
            I = np.eye(n)
            A_ub_list.extend([-I, I])
            b_ub_list.extend([
                self.delta_P_max * np.ones(n),
                self.delta_P_max * np.ones(n),
            ])

        A_ub = np.vstack(A_ub_list)
        b_ub = np.hstack(b_ub_list)

        # Resolver con SLSQP (disponible en cualquier scipy)
        def objective(du):
            return 0.5 * du @ H @ du + g @ du

        constraints = [{"type": "ineq", "fun": lambda du: b_ub - A_ub @ du}]

        bounds = None
        if self.delta_P_max is not None:
            bounds = [(-self.delta_P_max, self.delta_P_max)] * n

        result = opt.minimize(
            objective, np.zeros(n),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-12},
        )

        du_opt = result.x if result.success else np.zeros(n)

        # Aplicar solo el primer incremento (receding horizon)
        delta_u0 = du_opt[0]
        new_P = np.clip(current_P + delta_u0, self.P_min, self.P_max)

        self.P_history.append(new_P)
        self.q_history.append(current_q)
        return new_P

    def build_toeplitz_matrix(self, step_response):
        """Construye la matriz de Toeplitz a partir de la respuesta al escalón.

        Parameters
        ----------
        step_response : array (horizon,)
            Respuesta del craving a un escalón unitario de P.

        Returns
        -------
        S : array (horizon, horizon)
        """
        n = len(step_response)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                S[i, j] = step_response[i - j]
        self.S = S
        return S
