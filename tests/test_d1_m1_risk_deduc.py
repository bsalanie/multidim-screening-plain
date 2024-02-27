from math import isclose
from pathlib import Path

import numpy as np
from bs_python_utils.bs_opt import check_gradient_scalar_function
from dotenv import dotenv_values
from pytest import fixture
from scipy.stats import norm

from multidim_screening_plain.insurance_d1_m1_risk_deduc import S_fun, b_fun
from multidim_screening_plain.insurance_d1_m1_risk_deduc_values import (
    val_BC,
    val_D,
    val_I,
)
from multidim_screening_plain.setup import setup_model
from multidim_screening_plain.utils import (
    add_to_each_col,
    bs_norm_cdf,
    bs_norm_pdf,
    contracts_matrix,
    contracts_vector,
    multiply_each_col,
    my_outer_add,
)


@fixture
def build_model():
    config = dotenv_values(Path.cwd() / "multidim_screening_plain" / "config.env")
    model = setup_model(config)
    # module = model.model_module
    # module.precalculate(model)
    return model


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
    x = 1.9
    assert isclose(bs_norm_cdf(x), norm.cdf(x), rel_tol=1e-6, abs_tol=1e-6)


def test_norm_pdf():
    x = np.linspace(-8.0, 8.0, 100).reshape((20, 5))
    assert np.allclose(bs_norm_pdf(x), norm.pdf(x), rtol=1e-6, atol=1e-6)
    x = 1.9
    assert isclose(bs_norm_cdf(x), norm.cdf(x), rel_tol=1e-6, abs_tol=1e-6)


def test_dbscdf():
    y = np.array([2.0])

    def cdfpdf(z, args, gr):
        obj = bs_norm_cdf(z)[0]
        if gr:
            grad = np.array([bs_norm_pdf(z)])
            return obj, grad
        else:
            return obj

    anal, num = check_gradient_scalar_function(cdfpdf, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


def test_dvalBC(build_model):
    model = build_model
    y = np.array([1.6])
    theta = np.array([-6.0])

    def BCdBC(z, args, gr):
        if gr:
            obj_grad = val_BC(model, z, theta=theta, gr=True)
            return obj_grad
        else:
            obj = val_BC(model, z, theta=theta)
            return obj

    anal, num = check_gradient_scalar_function(BCdBC, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


def test_dvalD():
    y = np.array([0.3])
    delta = -6.0
    s = 4.0

    def DdD(z, args, gr):
        if gr:
            obj_grad = val_D(z, delta, s, gr=True)
            return obj_grad
        else:
            obj = val_D(z, delta, s)
            return obj

    anal, num = check_gradient_scalar_function(DdD, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


def test_dvalI(build_model):
    model = build_model
    y = np.array([1.3])
    theta = np.array([-6.0])

    def IdI(z, args, gr):
        if gr:
            obj_grad = val_I(model, z, theta=theta, gr=True)
            return obj_grad
        else:
            obj = val_I(model, z, theta=theta)
            return obj

    anal, num = check_gradient_scalar_function(IdI, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)

    ys = np.array([2.9, 0.8])

    def IdImat(zs, args, gr):
        if gr:
            obj_grad = val_I(model, zs, gr=True)
            obj = np.sum(obj_grad[0])
            grad = np.sum(obj_grad[1][0, :, :], axis=0)
            return obj, grad
        else:
            obj = np.sum(val_I(model, zs))
            return obj

    anal, num = check_gradient_scalar_function(IdImat, ys, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


def test_db(build_model):
    model = build_model
    y = np.array([1.2])
    theta = np.array([-6.0])

    def bdb(z, args, gr):
        if gr:
            obj_grad = b_fun(model, z, theta=theta, gr=True)
            return obj_grad
        else:
            obj = b_fun(model, z, theta=theta)
            return obj

    anal, num = check_gradient_scalar_function(bdb, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)

    ys = np.array([2.9, 0.8])

    def bdbmat(zs, args, gr):
        if gr:
            obj_grad = b_fun(model, zs, gr=True)
            obj = np.sum(obj_grad[0])
            grad = np.sum(obj_grad[1][0, :, :], axis=0)
            return obj, grad
        else:
            obj = np.sum(b_fun(model, zs))
            return obj

    anal, num = check_gradient_scalar_function(bdbmat, ys, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


def test_dS(build_model):
    model = build_model
    y = np.array([1.3])
    theta = np.array([-6.0])

    def SdS(z, args, gr):
        if gr:
            obj_grad = S_fun(model, z, theta=theta, gr=True)
            return obj_grad
        else:
            obj = S_fun(model, z, theta=theta)
            return obj

    anal, num = check_gradient_scalar_function(SdS, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)
