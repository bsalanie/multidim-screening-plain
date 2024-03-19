"""computes the components of the utilities for the insurance model
with two-dimensional types (risk-aversion, risk) and straight deductible contracts
"""

from typing import Any, cast

import numpy as np

from multidim_screening_plain.classes import ScreeningModel
from multidim_screening_plain.utils import (
    H_fun,
    bs_norm_cdf,
    bs_norm_pdf,
    check_args,
)


def val_A(deltas: np.ndarray | float, s: float, k: float) -> np.ndarray | float:
    """evaluates `A(delta,s)`, the probability that the loss is less than the deductible
    for all values of `delta` in `deltas`

    Args:
        deltas: a `q`-vector of risk parameters, or a single value
        s: the dispersion of losses
        k: `p_0/delta`

    Returns:
        the values of `A(delta,s)` as a `q`-vector, or a single value
    """
    return 1.0 - k * bs_norm_cdf(deltas / s)


def val_BC(
    model: ScreeningModel,
    y: np.ndarray,
    theta: np.ndarray | None = None,
    gr: bool = False,
) -> Any:
    """evaluates the values of `B+C`

    Args:
        model: the ScreeningModel
        y:  a `k`-vector of `k` contracts
        theta: if provided, a 2-vector with the characteristics of one type
        gr: if `True`, we also return the derivatives wrt `y`

    Returns:
        if `theta` is provided, `k` should be 1;
            then we return the value of (B + C)(y,theta,s)
            for this contract and this type
        otherwise we return the values of (B + C)(y,theta,s)
            for all types and the contracts in `y` as an $(N, k)$ matrix
        If `gr` is `True`, we also return the derivatives wrt `y`.
    """
    check_args("val_BC", y, 2, 1, theta)
    params = cast(np.ndarray, model.params)
    s, _, k = params
    if theta is not None:
        y_1 = y[0]
        sigma, delta = theta[0], theta[1]
        p_0 = k * delta
        y1sig = sigma * y_1
        d1 = delta / s + s * y1sig
        cdf_d1 = bs_norm_cdf(d1)
        val_expC = np.exp(y1sig * (s * s * y1sig / 2.0 + delta))
        val_compC = p_0 * cdf_d1 * val_expC
        if not gr:
            return val_compC
        else:
            grad = np.zeros(1)
            grad[0] = p_0 * sigma * s * H_fun(d1) * val_expC
            return val_compC, grad
    else:
        y_1 = y
        theta_mat = model.theta_mat
        sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
        p_0 = k * deltas
        y1sig = np.outer(sigmas, y_1)
        d1 = s * y1sig + deltas.reshape((-1, 1)) / s
        cdf_d1 = bs_norm_cdf(d1)
        val_expC = np.exp(y1sig * (s * s * y1sig / 2.0 + deltas.reshape((-1, 1))))
        val_compC = cdf_d1 * val_expC * p_0.reshape((-1, 1))
        if not gr:
            return val_compC
        else:
            grad = np.zeros((1, sigmas.size, y_1.size))
            grad[0, :, :] = s * H_fun(d1) * val_expC * (p_0 * sigmas).reshape((-1, 1))
            return val_compC, grad


def val_D(y: np.ndarray, delta: float, s: float, k: float, gr: bool = False) -> Any:
    """evaluates `D`, the actuarial premium

    Args:
        y: a 2-vector of 1 contract
        delta: a risk location parameter
        s: the dispersion of losses
        k: `p_0/delta`
        if `gr` we also return its gradient  wrt `y`

    Returns:
        the value of `D`
            and its gradient  wrt `y` if `gr` is `True`
    """
    y_1 = y[0]
    p_0 = k * delta
    dy0s = delta / s
    s_H = s * H_fun(dy0s)
    val_comp = s_H * p_0 * (1.0 - y_1)
    if not gr:
        return val_comp
    else:
        grad = np.zeros(1)
        grad[0] = -s_H * p_0
        return val_comp, grad


def val_I(
    model: ScreeningModel,
    y: np.ndarray,
    theta: np.ndarray | None = None,
    gr: bool = False,
) -> Any:
    """computes the integral `I`, and its gradient wrt `y` if `gr` is `True`

    Args:
        model: the ScreeningModel
        y:  a `k`-vector of `k` contracts
        theta: if provided, should be a 2-vector with the characteristics of one type;
            then `k` should equal 1
        gr: if `True`, we also return the gradient

    Returns:
        if `theta` is provided, the value of `I(y,theta,s)` for this type and contract;
        otherwise, the values of `I(y,t,s)` for all types and for all contracts in `y` as an $(N, k)$ matrix
        if `gr` is `True` we also return the gradient.
    """
    check_args("val_I", y, 2, 1, theta)
    params = cast(np.ndarray, model.params)
    s, _, k = params
    if theta is not None:
        delta = theta[1]
        value_A = cast(float, val_A(delta, s, k))
        value_BC = val_BC(model, y, theta=theta, gr=gr)
        if not gr:
            return value_BC + value_A
        else:
            val, grad = value_BC
            return val + value_A, grad
    else:
        deltas = model.theta_mat[:, 1]
        s = cast(np.ndarray, model.params)[0]
        value_A2 = cast(np.ndarray, val_A(deltas, s, k))
        value_BC = val_BC(model, y, gr=gr)
        if not gr:
            return value_BC + value_A2.reshape((-1, 1))
        else:
            val, grad = value_BC
            return val + value_A2.reshape((-1, 1)), grad


def proba_claim(deltas, s, k):
    return k * deltas * bs_norm_cdf(deltas / s)


def expected_positive_loss(deltas, s):
    return s * bs_norm_pdf(deltas / s) / bs_norm_cdf(deltas / s) + deltas


def cost_non_insur(model):
    sigmas = model.theta_mat[:, 0]
    y_no_insur = np.array([1.0])
    return np.log(val_I(model, y_no_insur))[:, 0] / sigmas
