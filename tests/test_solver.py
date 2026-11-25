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
    v, _lamb, it, converged = proj_K(model, w, np.zeros(N), gamma_proj)
    assert converged
    assert it > 0
    v_mat = v.reshape((N, N))
    assert np.all(v >= 0.0)
    # D* v <= f
    assert np.all(np.sum(v_mat, 1) - np.sum(v_mat, 0) <= model.f + 1e-4)


def test_nlLambda_and_JLambda(build_model):
    """the combined function must agree with `nlLambda` and `JLambda`"""
    from multidim_screening_plain.solver import JLambda, nlLambda_and_JLambda

    model = build_model("insurance", "d2_m2")
    rng = np.random.default_rng(3)
    y_mat = np.column_stack(
        (rng.uniform(0.0, 5.0, model.N), rng.uniform(0.0, 1.0, model.N))
    )
    y = contracts_vector(y_mat)
    Lambda_vals, J = nlLambda_and_JLambda(model, y)
    assert np.allclose(Lambda_vals, nlLambda(model, y), rtol=1e-14, atol=1e-14)
    assert np.allclose(J, JLambda(model, y), rtol=1e-14, atol=1e-14)


def test_twopoint_deductible_capped_at_loss(build_model):
    from multidim_screening_plain.solver import get_contract_bounds

    model = build_model("insurance", "twopoint_d1_m1_risk_deduc")
    loss = model.params[-1]
    lo, hi = get_contract_bounds(model)
    assert lo[0] == model.y_minmax[0, 0]
    assert hi[0] == min(model.y_minmax[0, 1], loss)
    # the proximal operators stay below the loss
    module = model.model_module
    theta = model.theta_mat[0]
    y = module.proximal_operator(model, theta, z=np.array([loss + 2.0]), t=1.0)
    assert y[0] <= loss
    Y = module.proximal_operator_batch(
        model, model.theta_mat, np.full((model.N, 1), loss + 2.0), np.ones(model.N)
    )
    assert np.all(loss >= Y)


def test_extrapolated_contracts_in_bounds(build_model):
    from multidim_screening_plain.solver import (
        extrapolate_contracts,
        get_contract_bounds,
    )

    model = build_model("insurance", "d2_m2")
    N = model.N
    lo, hi = get_contract_bounds(model)
    rng = np.random.default_rng(5)
    y_mat = np.column_stack((rng.uniform(0.0, 10.0, N), rng.uniform(0.0, 1.0, N)))
    y_old_mat = np.column_stack((rng.uniform(0.0, 10.0, N), rng.uniform(0.0, 1.0, N)))
    y, y_old = contracts_vector(y_mat), contracts_vector(y_old_mat)
    y_bar = extrapolate_contracts(model, y, y_old, 1.0)
    y_bar_mat = y_bar.reshape((2, N)).T
    assert np.all(y_bar_mat >= lo) and np.all(y_bar_mat <= hi)
    # inside the box, it is the usual extrapolation
    inside = np.all((2 * y_mat - y_old_mat >= lo) & (2 * y_mat - y_old_mat <= hi), 1)
    assert np.allclose(y_bar_mat[inside], 2 * y_mat[inside] - y_old_mat[inside])
