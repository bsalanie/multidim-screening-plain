"""the batched proximal operators must agree with the per-type ones"""

import numpy as np
import pytest

from multidim_screening_plain.solver import get_contract_bounds

INSTANCES = [
    "d2_m2",
    "d2_m1_deduc",
    "d2_m1_copay",
    "d1_m1_risk_deduc",
    "d1_m1_risk_copay",
    "twopoint_d1_m1_risk_deduc",
]


@pytest.mark.parametrize("instance", INSTANCES)
@pytest.mark.parametrize("t", [0.01, 0.3, 5.0])
def test_batch_prox_matches_per_type(build_model, instance, t):
    model = build_model("insurance", instance)
    module = model.model_module
    N, m = model.N, model.m
    rng = np.random.default_rng(2024)
    lo, hi = get_contract_bounds(model)
    # some points inside the box, some outside it
    width = hi - lo
    Z = lo + rng.uniform(-0.1, 1.1, size=(N, m)) * width
    if instance.startswith("twopoint"):  # deductibles below the loss
        Z = rng.uniform(-0.5, 5.5, size=(N, m))
    T = np.full(N, t)

    Y_batch = module.proximal_operator_batch(model, model.theta_mat, Z, T)

    def prox_objective(y, i):
        S_i = module.S_fun(model, y, theta=model.theta_mat[i])
        return -S_i + np.sum((y - Z[i]) ** 2) / (2.0 * t)

    for i in range(N):
        y_i = module.proximal_operator(model, model.theta_mat[i], z=Z[i], t=t)
        assert np.all(Y_batch[i] >= lo) and np.all(Y_batch[i] <= hi)
        # never worse than L-BFGS-B, which can stop early when `t` is large
        assert prox_objective(Y_batch[i], i) <= prox_objective(y_i, i) + 1e-10
        assert np.allclose(Y_batch[i], y_i, atol=1e-4), (i, Z[i], Y_batch[i], y_i)
