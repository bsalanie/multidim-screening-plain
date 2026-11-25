import numpy as np
import pytest
from bs_python_utils.bs_opt import check_gradient_scalar_function

from multidim_screening_plain.mussarosen.d1_m1 import mussarosen_d1_m1
from multidim_screening_plain.mussarosen.d2_m2 import mussarosen_d2_m2

# (model instance, module, theta for one type, one contract, several contracts)
CASES = [
    (
        "d1_m1",
        mussarosen_d1_m1,
        np.array([1.5]),
        np.array([0.8]),
        np.array([0.3, 1.1, 1.7]),
    ),
    (
        "d2_m2",
        mussarosen_d2_m2,
        np.array([1.5, 1.2]),
        np.array([0.8, 0.4]),
        np.array([0.3, 1.1, 1.7, 0.5, 0.9, 1.4]),
    ),
]


def _check(fun, y):
    anal, num = check_gradient_scalar_function(fun, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("instance, module, theta, y, ys", CASES)
def test_db(build_model, instance, module, theta, y, ys):
    model = build_model("mussarosen", instance)

    def bdb(z, args, gr):
        return module.b_fun(model, z, theta=theta, gr=gr)

    _check(bdb, y)

    def bdbmat(zs, args, gr):
        if gr:
            val, grad = module.b_fun(model, zs, gr=True)
            return np.sum(val), np.sum(grad, axis=1).reshape(-1)
        else:
            return np.sum(module.b_fun(model, zs))

    _check(bdbmat, ys)


@pytest.mark.parametrize("instance, module, theta, y, ys", CASES)
def test_dS(build_model, instance, module, theta, y, ys):
    model = build_model("mussarosen", instance)

    def SdS(z, args, gr):
        return module.S_fun(model, z, theta=theta, gr=gr)

    _check(SdS, y)
