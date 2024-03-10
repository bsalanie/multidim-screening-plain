import numpy as np
from bs_python_utils.bs_opt import check_gradient_scalar_function

from multidim_screening_plain.insurance_d1_m1_risk_deduc import S_fun, b_fun
from multidim_screening_plain.insurance_d1_m1_risk_deduc_values import (
    val_BC,
    val_D,
    val_I,
)


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
