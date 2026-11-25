import numpy as np
from bs_python_utils.bs_opt import check_gradient_scalar_function

from multidim_screening_plain.insurance.d1_m1_risk_copay.insurance_d1_m1_risk_copay import (
    S_fun,
    b_fun,
)
from multidim_screening_plain.insurance.d1_m1_risk_copay.insurance_d1_m1_risk_copay_values import (
    val_BC,
    val_D,
    val_I,
)

THETA = np.array([6.0])  # risk location, in [4, 8]


def _check(fun, y):
    anal, num = check_gradient_scalar_function(fun, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


def _one_type(f, model, theta):
    def fun(z, args, gr):
        return f(model, z, theta=theta, gr=gr)

    return fun


def _all_types(f, model):
    """sum over all types and contracts, so the gradient is wrt all of `y`"""

    def fun(zs, args, gr):
        if gr:
            val, grad = f(model, zs, gr=True)
            return np.sum(val), np.sum(grad, axis=1).reshape(-1)
        else:
            return np.sum(f(model, zs))

    return fun


def test_dvalBC(build_model):
    model = build_model("insurance", "d1_m1_risk_copay")
    _check(_one_type(val_BC, model, THETA), np.array([0.4]))
    _check(_all_types(val_BC, model), np.array([0.2, 0.7]))


def test_dvalD():
    delta, s, k = 6.0, 2.0, 0.012

    def DdD(z, args, gr):
        return val_D(z, delta, s, k, gr=gr)

    _check(DdD, np.array([0.3]))


def test_dvalI(build_model):
    model = build_model("insurance", "d1_m1_risk_copay")
    _check(_one_type(val_I, model, THETA), np.array([0.4]))
    _check(_all_types(val_I, model), np.array([0.2, 0.7]))


def test_db(build_model):
    model = build_model("insurance", "d1_m1_risk_copay")
    _check(_one_type(b_fun, model, THETA), np.array([0.4]))
    _check(_all_types(b_fun, model), np.array([0.2, 0.7]))


def test_dS(build_model):
    model = build_model("insurance", "d1_m1_risk_copay")
    _check(_one_type(S_fun, model, THETA), np.array([0.4]))
