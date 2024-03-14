import numpy as np
from bs_python_utils.bs_opt import check_gradient_scalar_function

from multidim_screening_plain.insurance_d2_m1_deduc_mix import S_fun, b_fun
from multidim_screening_plain.insurance_d2_m1_deduc_values_mix import (
    val_BC,
    val_D,
    val_I,
)


def test_dvalBC(build_model):
    model = build_model("insurance_d2_m1_deduc_mix")
    y = np.array([1.6])
    theta = np.array([0.5, 6.0])

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
    delta = 6.0
    s = 2.0
    k = 0.012

    def DdD(z, args, gr):
        if gr:
            obj_grad = val_D(z, delta, s, k, gr=True)
            return obj_grad
        else:
            obj = val_D(z, delta, s, k)
            return obj

    anal, num = check_gradient_scalar_function(DdD, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)


def test_dvalI(build_model):
    model = build_model("insurance_d2_m1_deduc_mix")
    y = np.array([0.3])
    theta = np.array([0.5, 6.0])

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
    model = build_model("insurance_d2_m1_deduc_mix")
    y = np.array([1.3])
    theta = np.array([0.5, 6.0])

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
    model = build_model("insurance_d2_m1_deduc_mix")
    y = np.array([1.3])
    theta = np.array([0.5, 6.0])

    def SdS(z, args, gr):
        if gr:
            obj_grad = S_fun(model, z, theta=theta, gr=True)
            return obj_grad
        else:
            obj = S_fun(model, z, theta=theta)
            return obj

    anal, num = check_gradient_scalar_function(SdS, y, args=[])
    assert np.allclose(anal, num, rtol=1e-5, atol=1e-5)
