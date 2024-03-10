import numpy as np
from bs_python_utils.bs_opt import check_gradient_scalar_function

from multidim_screening_plain.jointtax_d2_m2 import S_fun, b_fun


def test_db(build_model):
    model = build_model
    y = np.array([1.7, 0.2])
    theta = np.array([1.5, 0.5])

    def bdb(z, args, gr):
        if gr:
            obj_grad = b_fun(model, z, theta=theta, gr=True)
            return obj_grad
        else:
            obj = b_fun(model, z, theta=theta)
            return obj

    anal, num = check_gradient_scalar_function(bdb, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


def test_dS(build_model):
    model = build_model
    y = np.array([1.7, 0.2])
    theta = np.array([1.5, 0.5])

    def SdS(z, args, gr):
        if gr:
            obj_grad = S_fun(model, z, theta=theta, gr=True)
            return obj_grad
        else:
            obj = S_fun(model, z, theta=theta)
            return obj

    anal, num = check_gradient_scalar_function(SdS, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)
