"""Batched proximal operator: solves the per-type proximal problems of all types at once.

For each type `i` in the batch we minimize, over the box `[lo, hi]`,

    phi_i(y) = -S_i(y) + ||y - z_i||^2 / (2 t_i)

by a projected Newton method (Bertsekas' two-metric projection), vectorized over types.
The Hessian of `S_i` is obtained by finite differences of its analytic gradient.

A model module can use this by providing a function
`proximal_operator_batch(model, thetas, Z, T)`, which the solver calls instead of
calling `proximal_operator` once per type.
"""

from collections.abc import Callable
from typing import cast

import numpy as np

# `surplus_and_grad(Y, rows)` returns `S` (an `n`-vector) and its gradient
#   (an `(n, m)` matrix) for the types `rows` of the batch at the contracts `Y`
SurplusAndGrad = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def prox_box_batch(
    surplus_and_grad: SurplusAndGrad,
    Z: np.ndarray,
    T: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 100,
    h_fd: float = 1e-6,
) -> np.ndarray:
    """minimizes `-S_i(y) + ||y - z_i||^2 / (2 t_i)` over the box `[lo, hi]` for all `i`

    Args:
        surplus_and_grad: the surplus and its gradient, see `SurplusAndGrad`
        Z: an `(n, m)` matrix, the points `z_i`
        T: an `n`-vector, the steps `t_i`
        lo: an `m`-vector of lower bounds
        hi: an `m`-vector of upper bounds
        tol: tolerance, in units of `y`, on the projected gradient step `t_i g_i`
            and on the Newton step
        max_iter: maximum number of Newton iterations
        h_fd: step for the finite-difference Hessian

    Returns:
        the `(n, m)` matrix of minimizers
    """
    n, m = Z.shape
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    Y = np.clip(Z, lo, hi)
    eye_m = np.eye(m)

    def phi(Ya: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S, G = surplus_and_grad(Ya, rows)
        dyz = Ya - Z[rows]
        inv_t = 1.0 / T[rows]
        val = -S + 0.5 * np.sum(dyz * dyz, axis=1) * inv_t
        grad = -G + dyz * inv_t[:, None]
        return val, grad

    rows = np.arange(n)
    F, G = phi(Y, rows)
    for _ in range(max_iter):
        if rows.size == 0:
            break
        Ya, f, g = Y[rows], F[rows], G[rows]

        # stop the types whose projected gradient step, in units of `y`, is small
        t_a = T[rows][:, None]
        proj_step = np.clip(Ya - t_a * g, lo, hi) - Ya
        not_done = np.max(np.abs(proj_step), axis=1) > tol
        rows, Ya, f, g = rows[not_done], Ya[not_done], f[not_done], g[not_done]
        t_a = t_a[not_done]
        if rows.size == 0:
            break
        na = rows.size

        # Hessian by forward differences of the gradient, stepping into the box
        H = np.empty((na, m, m))
        for k in range(m):
            step = np.where(Ya[:, k] + h_fd <= hi[k], h_fd, -h_fd)
            Yk = Ya.copy()
            Yk[:, k] += step
            _, gk = phi(Yk, rows)
            H[:, :, k] = (gk - g) / step[:, None]
        H = 0.5 * (H + np.transpose(H, (0, 2, 1)))

        # variables at a bound with the gradient pointing outwards are held fixed
        eps_b = 1e-12
        at_bound = ((Ya <= lo + eps_b) & (g > 0.0)) | ((Ya >= hi - eps_b) & (g < 0.0))
        free = ~at_bound
        M = H * free[:, :, None] * free[:, None, :]
        M += at_bound[:, :, None] * eye_m[None, :, :]
        # make the reduced Hessian positive definite
        min_eig = np.linalg.eigvalsh(M)[:, 0]
        shift = np.maximum(0.0, 1e-8 - min_eig)
        M += shift[:, None, None] * eye_m[None, :, :]
        newton = -np.linalg.solve(M, (g * free)[:, :, None])[:, :, 0]
        direction = np.where(free, newton, -t_a * g)

        # backtracking line search (Armijo) along the projected arc;
        #   we keep the value and gradient at the accepted points
        alpha = np.ones(na)
        Y_new = np.clip(Ya + direction, lo, hi)
        f_new = np.empty(na)
        g_new = np.empty((na, m))
        searching = np.arange(na)
        for _ls in range(40):
            f_s, g_s = phi(Y_new[searching], rows[searching])
            f_new[searching], g_new[searching] = f_s, g_s
            decrease = np.sum(g[searching] * (Y_new[searching] - Ya[searching]), axis=1)
            # the slack allows for rounding errors in `f` near the minimum
            slack = 1e-13 * (1.0 + np.abs(f[searching]))
            ok = f_s <= f[searching] + 1e-4 * decrease + slack
            searching = searching[~ok]
            if searching.size == 0:
                break
            alpha[searching] *= 0.5
            Y_new[searching] = np.clip(
                Ya[searching] + alpha[searching, None] * direction[searching], lo, hi
            )

        step_size = np.max(np.abs(Y_new - Ya), axis=1)
        Y[rows], F[rows], G[rows] = Y_new, f_new, g_new
        rows = rows[step_size > tol]

    return cast(np.ndarray, Y)
