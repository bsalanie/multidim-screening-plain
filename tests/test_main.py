import numpy as np
from bs_python_utils.bs_opt import check_gradient_scalar_function
from scipy.stats import norm

from multidim_screening_plain.insurance_d2_m2 import S_fun, b_fun, db_fun, dS_fun
from multidim_screening_plain.insurance_d2_m2_values import (
    d0_val_B,
    d0_val_BC,
    d0_val_C,
    d1_val_C,
    val_B,
    val_C,
    val_I,
    val_I_new,
)
from multidim_screening_plain.utils import (
    add_to_each_col,
    bs_norm_cdf,
    contracts_matrix,
    contracts_vector,
    multiply_each_col,
    my_outer_add,
)


def test_multiply_each_col():
    mat = np.arange(1, 7).reshape((2, 3))
    vec = np.array([2, 3])
    res_th = np.array([[2, 4, 6], [12, 15, 18]])
    res_th2 = mat * vec.reshape((-1, 1))
    assert np.allclose(res_th, multiply_each_col(mat, vec))
    assert np.allclose(res_th, res_th2)


def test_add_to_each_col():
    mat = np.arange(1, 7).reshape((2, 3))
    vec = np.array([2, 3])
    res_th = np.array([[3, 4, 5], [7, 8, 9]])
    res_th2 = mat + vec.reshape((-1, 1))
    assert np.allclose(res_th, add_to_each_col(mat, vec))
    assert np.allclose(res_th, res_th2)


def test_my_outer_add():
    vec1 = np.array([2, 3])
    vec2 = np.array([1, 6, 9])
    res_th = np.array([[3, 8, 11], [4, 9, 12]])
    res_th2 = np.add.outer(vec1, vec2)
    assert np.allclose(res_th, my_outer_add(vec1, vec2))
    assert np.allclose(res_th, res_th2)


def test_contracts_vector():
    y_mat = np.arange(1, 11).reshape((2, 5))
    y_vec = contracts_vector(y_mat)
    assert np.allclose(y_vec, np.array([1, 6, 2, 7, 3, 8, 4, 9, 5, 10]))


def test_contracts_matrix():
    y = np.array([1, 6, 2, 7, 3, 8, 4, 9, 5, 10])
    y_mat = contracts_matrix(y, 2)
    assert np.allclose(y_mat, np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]))


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


def test_d0_val_BC():
    y = np.array([0.3, 0.2])
    theta_mat1 = np.array([[0.5, -6.0]])
    sigmas = np.array([theta_mat1[0, 0]])
    deltas = np.array([theta_mat1[0, 1]])
    params = np.array([4.0, 0.25])
    s = params[0]
    d0_B = d0_val_B(y, sigmas, deltas, s)[0, 0]
    d0_C = d0_val_C(y, sigmas, deltas, s)[0, 0]
    d0_BC = d0_val_BC(y, sigmas, deltas, s)[0, 0]
    assert np.allclose(d0_BC, d0_B + d0_C)


def test_new_val_I():
    y = np.array([0.3, 0.2])
    theta_mat1 = np.array([[0.5, -6.0]])
    sigmas = np.array([theta_mat1[0, 0]])
    deltas = np.array([theta_mat1[0, 1]])
    params = np.array([4.0, 0.25])
    s = params[0]
    value = val_I(y, sigmas, deltas, s)[0, 0]
    new_value = val_I_new(y, sigmas, deltas, s)[0, 0]
    assert np.allclose(value, new_value)
