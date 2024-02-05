import numpy as np
from bs_python_utils.bs_opt import check_gradient_scalar_function
from scipy.stats import norm

from multidim_screening_plain.insurance_d2_m2 import S_fun, b_fun, db_fun, dS_fun
from multidim_screening_plain.insurance_d2_m2_values import (
    d0_val_B,
    d0_val_C,
    d1_val_C,
    val_B,
    val_C,
)
from multidim_screening_plain.utils import bs_norm_cdf


def test_norm_cdf():
    x = np.linspace(-8.0, 8.0, 100).reshape((20, 5))
    assert np.allclose(bs_norm_cdf(x), norm.cdf(x), rtol=1e-6, atol=1e-6)


def test_dvalB():
    y = np.array([0.3, 0.2])
    theta_mat1 = np.array([[0.5, -6.0]])
    sigmas = np.array([theta_mat1[0, 0]])
    deltas = np.array([theta_mat1[0, 1]])
    params = np.array([4.0, 0.25])
    s = params[0]

    def BdB(z, args, gr):
        obj = val_B(z, sigmas, deltas, s)[0, 0]
        if gr:
            grad = np.array([d0_val_B(z, sigmas, deltas, s)[0, 0], 0.0])
            return obj, grad
        return obj

    anal, num = check_gradient_scalar_function(BdB, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


def test_dvalC():
    y = np.array([0.3, 0.2])
    theta_mat1 = np.array([[0.5, -6.0]])
    sigmas = np.array([theta_mat1[0, 0]])
    deltas = np.array([theta_mat1[0, 1]])
    params = np.array([4.0, 0.25])
    s = params[0]

    def CdC(z, args, gr):
        obj = val_C(z, sigmas, deltas, s)[0, 0]
        if gr:
            grad = np.array(
                [
                    d0_val_C(z, sigmas, deltas, s)[0, 0],
                    d1_val_C(z, sigmas, deltas, s)[0, 0],
                ]
            )
            return obj, grad
        return obj

    anal, num = check_gradient_scalar_function(CdC, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


def test_db():
    y = np.array([0.3, 0.2])
    theta_mat1 = np.array([[0.5, -6.0]])
    params = np.array([4.0, 0.25])

    def bdb(z, args, gr):
        obj = b_fun(z, theta_mat1, params)[0, 0]
        if gr:
            grad = db_fun(z, theta_mat1, params)[:, 0, 0]
            return obj, grad
        return obj

    anal, num = check_gradient_scalar_function(bdb, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


def test_dS():
    y = np.array([0.3, 0.2])
    theta_mat1 = np.array([[0.5, -6.0]])
    params = np.array([4.0, 0.25])

    def SdS(z, args, gr):
        obj = S_fun(z, theta_mat1, params)[0, 0]
        if gr:
            grad = dS_fun(z, theta_mat1, params)[:, 0, 0]
            return obj, grad
        return obj

    anal, num = check_gradient_scalar_function(SdS, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)

    def SdS(z, args, gr):
        obj = S_fun(z, theta_mat1, params)[0, 0]
        if gr:
            grad = dS_fun(z, theta_mat1, params)[:, 0, 0]
            return obj, grad
        return obj

    anal, num = check_gradient_scalar_function(SdS, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)
