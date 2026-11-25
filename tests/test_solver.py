from types import SimpleNamespace

import numpy as np

from multidim_screening_plain.solver import (
    compute_utilities,
    construct_D,
    nlLambda,
    proj_K,
)
from multidim_screening_plain.utils import contracts_vector


def test_compute_utilities_fixed_point(build_model):
    """the rents must be the smallest `U >= 0` with `U_i >= Lambda_ij + U_j`"""
    model = build_model("mussarosen", "d2_m2")
    N = model.N
    # with b_i(y) = theta_i . y, contracts y = grad phi(theta) with phi convex
    #   are implementable (so the fixed point is finite); with this phi
    #   the rents build up along long chains of IC constraints
    #  we list the types from the top down, so that one sweep is not enough
    model.theta_mat = model.theta_mat[::-1].copy()
    theta = model.theta_mat
    sq_norms = np.sum(theta * theta, axis=1, keepdims=True)
    y_mat = theta * (1.0 + 0.3 * sq_norms) - 1.5
    model.add_first_best(y_mat)
    results = SimpleNamespace(model=model, SB_y=y_mat)

    _, U, _ = compute_utilities(results)

    Lambda = nlLambda(model, contracts_vector(y_mat)).reshape((N, N))
    # brute-force least fixed point
    V = np.zeros(N)
    for _ in range(2 * N):
        V = np.maximum(0.0, np.max(Lambda + V[None, :], axis=1))
    assert np.allclose(U, V, atol=1e-6)
    assert np.all(U >= -1e-12)
    assert np.all(U[:, None] >= Lambda + U[None, :] - 1e-6)


def test_proj_K(build_model):
    """the projection is feasible and counts its iterations"""
    model = build_model("mussarosen", "d1_m1")
    N = model.N
    _, gamma_proj = construct_D(N)
    rng = np.random.default_rng(7)
    w = rng.normal(size=N * N)
    v, lamb, it, converged = proj_K(model, w, np.zeros(N), gamma_proj)
    assert converged
    assert it > 0
    v_mat = v.reshape((N, N))
    assert np.all(v >= 0.0)
    # D* v <= f
    assert np.all(np.sum(v_mat, 1) - np.sum(v_mat, 0) <= model.f + 1e-4)
