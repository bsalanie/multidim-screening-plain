"""checks the closed forms of `b` and `S` in the TPAC insurance models
against direct numerical integration over the distribution of losses"""

import importlib

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

S_DISP, LOADING, K = 2.0, 0.25, 0.012  # as in the configuration files
SIGMA_1D = 0.6  # the risk aversion in the one-dimensional models
L_MAX = 60.0  # beyond this the density of losses is negligible


def b_and_S_by_quadrature(sigma, delta, y0, y1):
    """`b` and `S` for the contract `R(L) = (1-y1) max(L-y0, 0)`"""

    def reimbursement(loss):
        return (1.0 - y1) * max(loss - y0, 0.0)

    def density(loss):  # density of positive losses
        return K * delta * norm.pdf((loss - delta) / S_DISP) / S_DISP

    prob_no_loss = 1.0 - K * delta * norm.cdf(delta / S_DISP)

    def integral_I(r):
        def integrand(loss):
            return np.exp(sigma * (loss - r(loss))) * density(loss)

        return (
            prob_no_loss
            + quad(integrand, 0.0, y0)[0]
            + quad(integrand, y0, L_MAX, limit=200)[0]
        )

    b = (np.log(integral_I(lambda _: 0.0)) - np.log(integral_I(reimbursement))) / sigma
    D = quad(lambda loss: reimbursement(loss) * density(loss), 0.0, L_MAX, limit=200)[0]
    return b, b - (1.0 + LOADING) * D


# (sigma, delta, deductible, copay)
POINTS = [
    (sigma, delta, y0, y1)
    for sigma in (0.2, 0.6, 1.0)
    for delta in (4.0, 5.5, 8.0)
    for y0, y1 in ((0.3, 0.0), (2.0, 0.0), (0.0, 0.4), (0.0, 0.9), (1.2, 0.5))
]


def _cases(instance):
    """the points that fit model `instance`, with its `theta` and `y`"""
    for sigma, delta, y0, y1 in POINTS:
        if instance == "d2_m2":
            yield (sigma, delta, y0, y1), [sigma, delta], [y0, y1]
        elif instance == "d2_m1_deduc" and y1 == 0.0:
            yield (sigma, delta, y0, y1), [sigma, delta], [y0]
        elif instance == "d2_m1_copay" and y0 == 0.0:
            yield (sigma, delta, y0, y1), [sigma, delta], [y1]
        elif instance == "d1_m1_risk_deduc" and sigma == SIGMA_1D and y1 == 0.0:
            yield (sigma, delta, y0, y1), [delta], [y0]
        elif instance == "d1_m1_risk_copay" and sigma == SIGMA_1D and y0 == 0.0:
            yield (sigma, delta, y0, y1), [delta], [y1]


@pytest.mark.parametrize(
    "instance",
    ["d2_m2", "d2_m1_deduc", "d2_m1_copay", "d1_m1_risk_deduc", "d1_m1_risk_copay"],
)
def test_b_S_quadrature(build_model, instance):
    model = build_model("insurance", instance)
    module = importlib.import_module(
        f"multidim_screening_plain.insurance.{instance}.insurance_{instance}"
    )
    n_cases = 0
    for point, theta, y in _cases(instance):
        b_ref, S_ref = b_and_S_by_quadrature(*point)
        theta_arr, y_arr = np.array(theta), np.array(y)
        b = float(np.squeeze(module.b_fun(model, y_arr, theta=theta_arr)))
        S = float(np.squeeze(module.S_fun(model, y_arr, theta=theta_arr)))
        assert np.isclose(b, b_ref, rtol=1e-4, atol=1e-4), (point, b, b_ref)
        assert np.isclose(S, S_ref, rtol=1e-4, atol=1e-4), (point, S, S_ref)
        n_cases += 1
    assert n_cases > 0
