"""computes the components of the utilities for the insurance model
with a mixture of losses
with two-dimensional types (risk-aversion, risk) and (deductible, copay) contracts
"""

from typing import Any, cast

import numpy as np

from multidim_screening_plain.classes import ScreeningModel
from multidim_screening_plain.utils import (
    H_fun,
    bs_norm_cdf,
    bs_norm_pdf,
    check_args,
    split_y,
)


def val_A(deltas: np.ndarray | float, s: float, k: float) -> np.ndarray | float:
    """evaluates `A(delta,s)`, the probability that the loss is less than the deductible
    for all values of `delta` in `deltas`

    Args:
        deltas: a `q`-vector of risk parameters, or a single value
        s: the dispersion of losses
        k: `p_1/delta`

    Returns:
        the values of `A(delta,s)` as a `q`-vector, or a single value
    """
    return 1.0 - k * deltas * bs_norm_cdf(deltas / s)


def val_BC(
    model: ScreeningModel,
    y: np.ndarray,
    theta: np.ndarray | None = None,
    gr: bool = False,
) -> Any:
    """evaluates the values of `B+C`

    Args:
        model: the ScreeningModel
        y:  a `2 k`-vector of `k` contracts
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
    check_args("val_BC", y, 2, 2, theta)

    if theta is not None:
        return val_BC_1(model, y, theta, gr=gr)
    else:
        return val_BC_all(model, y, gr=gr)


def val_BC_1(
    model: ScreeningModel, y: np.ndarray, theta: np.ndarray, gr: bool = False
) -> Any:
    params = cast(np.ndarray, model.params)
    s, _, k = params
    y_0, y_1 = y[0], y[1]
    sigma, delta = theta[0], theta[1]
    argu1 = delta / s + s * sigma
    dy0s = (delta - y_0) / s
    argu2 = dy0s + s * sigma
    cdf1 = bs_norm_cdf(argu1)
    cdf2 = bs_norm_cdf(argu2)
    y1sig = sigma * y_1
    y01sig = sigma * y_0 * (1 - y_1)
    ny1sig = sigma * (1 - y_1)
    p_1 = k * delta
    val_expB = np.exp(sigma * (s * s * sigma / 2.0 + delta))
    val_compB = p_1 * (cdf1 - cdf2) * val_expB
    val_expC = np.exp(y1sig * (s * s * y1sig / 2.0 + delta) + y01sig)
    d1 = dy0s + s * y1sig
    cdf_d1 = bs_norm_cdf(d1)
    val_compC = p_1 * cdf_d1 * val_expC
    if not gr:
        return val_compB + val_compC
    else:
        pdf2 = bs_norm_pdf(argu2)
        pdf_d1 = bs_norm_pdf(d1)
        grad = np.zeros(2)
        grad[0] = p_1 * (
            pdf2 * val_expB / s + (cdf_d1 * ny1sig - pdf_d1 / s) * val_expC
        )
        grad[1] = s * p_1 * H_fun(d1) * val_expC * sigma
        return val_compB + val_compC, grad


def val_BC_all(model: ScreeningModel, y: np.ndarray, gr: bool = False) -> Any:
    params = cast(np.ndarray, model.params)
    s, _, k = params
    y_0, y_1 = split_y(y, 2)
    theta_mat = model.theta_mat
    sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
    argu1 = deltas / s + s * sigmas
    dy0s = np.subtract.outer(deltas, y_0) / s
    argu2 = dy0s + s * sigmas.reshape((-1, 1))
    cdf1a = cast(np.ndarray, bs_norm_cdf(argu1))
    cdf2 = bs_norm_cdf(argu2)
    y1sig = np.outer(sigmas, y_1)
    y01sig = np.outer(sigmas, y_0 * (1 - y_1))
    ny1sig = np.outer(sigmas, 1 - y_1)
    d1 = dy0s + s * y1sig
    cdf_d1 = bs_norm_cdf(d1)
    p_1 = k * deltas
    val_expBa = np.exp(sigmas * (s * s * sigmas / 2.0 + deltas))
    val_compB = (-cdf2 + cdf1a.reshape((-1, 1))) * (p_1 * val_expBa).reshape((-1, 1))
    val_expC = np.exp(y1sig * (s * s * y1sig / 2.0 + deltas.reshape((-1, 1))) + y01sig)
    val_compC = cdf_d1 * val_expC * p_1.reshape((-1, 1))
    if not gr:
        return val_compB + val_compC
    else:
        pdf2 = bs_norm_pdf(argu2)
        pdf_d1 = bs_norm_pdf(d1)
        grad = np.zeros((2, sigmas.size, y_0.size))
        grad[0, :, :] = (
            pdf2 * val_expBa.reshape((-1, 1)) / s
            + (cdf_d1 * ny1sig - pdf_d1 / s) * val_expC
        ) * p_1.reshape((-1, 1))
        grad[1, :, :] = s * H_fun(d1) * val_expC * (p_1 * sigmas).reshape((-1, 1))
        return val_compB + val_compC, grad


def val_D(y: np.ndarray, delta: float, s: float, k: float, gr: bool = False) -> Any:
    """evaluates `D(y,delta,s)`, the actuarial premium

    Args:
        y: a 2-vector of 1 contract
        delta: a risk location parameter
        s: the dispersion of losses
        k: is `p_1/delta`
        if `gr` we also return its gradient  wrt `y`

    Returns:
        the value of `D`
            and its gradient  wrt `y` if `gr` is `True`
    """
    y_0, y_1 = y
    dy0s = (delta - y_0) / s
    p_1 = k * delta
    s_H = s * H_fun(dy0s)
    val_comp = s_H * p_1 * (1 - y_1)
    if not gr:
        return val_comp
    else:
        grad = np.zeros(2)
        grad[0] = -bs_norm_cdf(dy0s) * (1 - y_1) * p_1
        grad[1] = -s_H * p_1
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
        y:  a `2 k`-vector of `k` contracts
        theta: if provided, should be a 2-vector with the characteristics of one type;
            then `k` should equal 1
        gr: if `True`, we also return the gradient

    Returns:
        if `theta` is provided, the value of `I(y,theta,s)` for this type and contract;
        otherwise, the values of `I(y,t,s)` for all types and for all contracts in `y` as an $(N, k)$ matrix
        if `gr` is `True` we also return the gradient.
    """
    check_args("val_I", y, 2, 2, theta)
    params = cast(np.ndarray, model.params)
    s, k = params[0], params[2]
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
        value_A2 = cast(np.ndarray, val_A(deltas, s, k))
        value_BC = val_BC(model, y, gr=gr)
        if not gr:
            return value_BC + value_A2.reshape((-1, 1))
        else:
            val, grad = value_BC
            return val + value_A2.reshape((-1, 1)), grad


def val_I_no_insurance(model: ScreeningModel, theta: np.ndarray | None = None) -> Any:
    params = cast(np.ndarray, model.params)
    s, k = params[0], params[2]
    if theta is not None:
        sigma, delta = theta[0], theta[1]
        argu1 = delta / s + s * sigma
        cdf1 = bs_norm_cdf(argu1)
        p_1 = k * delta
        val_expB = np.exp(sigma * (s * s * sigma / 2.0 + delta))
        val_compB = p_1 * cdf1 * val_expB
        delta = theta[1]
        value_A = cast(float, val_A(delta, s, k))
        return val_compB + value_A
    else:
        theta_mat = model.theta_mat
        sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
        value_A2 = cast(np.ndarray, val_A(deltas, s, k))
        argu1 = deltas / s + s * sigmas
        cdf1a = cast(np.ndarray, bs_norm_cdf(argu1))
        p_1 = k * deltas
        val_expBa = np.exp(sigmas * (s * s * sigmas / 2.0 + deltas))
        val_compBa = cdf1a * p_1 * val_expBa
        return val_compBa + value_A2


def proba_claim(deltas, s, k):
    return k * deltas * bs_norm_cdf(deltas / s)


def expected_positive_loss(deltas, s):
    return s * bs_norm_pdf(deltas / s) / bs_norm_cdf(deltas / s) + deltas


def cost_non_insur(model):
    sigmas = model.theta_mat[:, 0]
    return np.log(val_I_no_insurance(model)) / sigmas
