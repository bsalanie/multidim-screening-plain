"""computes the components of the utilities for the insurance model
with one-dimensional types (risk) and straight-deductible contracts
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

# penalties to keep minimization of `S` within bounds
coeff_qpenalty_S0 = 0.00001  # coefficient of the quadratic penalty on S for y0 large
coeff_qpenalty_S0_0 = 1_000.0  # coefficient of the quadratic penalty on S for y0<0


def val_A(deltas: np.ndarray | float, s: float, k: float) -> np.ndarray | float:
    """evaluates `A(delta,s,k)`, the probability that the loss is less than the deductible
    for all values of `delta` in `deltas`

    Args:
        deltas: a `q`-vector of risk parameters, or a single value
        s: the dispersion of losses
        k: `p_0/delta`

    Returns:
        the values of `A(delta,s,k)` as a `q`-vector, or a single value
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
        y:  a `k`-vector of `k` contracts
        theta: if provided, a 1-vector for one type
        gr: if `True`, we also return the derivatives wrt `y`

    Returns:
        if `theta` is provided, `y` should be  a 1-vector;
            then we return the value of (B + C)(y,theta,s)
            for this contract and this type
        otherwise we return the values of (B + C)(y,theta,s)
            for all types and the contracts in `y` as an $(N, k)$ matrix
        If `gr` is `True`, we also return the derivatives wrt `y`.
    """
    check_args("val_BC", y, 1, 1, theta)
    params = cast(np.ndarray, model.params)
    sigma, s, _, k = params
    if theta is not None:
        return val_BC_1(theta, y, sigma, s, k, gr=gr)
    else:
        return val_BC_all(model, y, sigma, s, k, gr=gr)


def val_BC_1(
    theta: np.ndarray, y: np.ndarray, sigma: float, s: float, k: float, gr: bool = False
) -> Any:
    """`val_BC` for one type and one contract"""
    delta, y_0 = theta[0], y[0]
    p_0 = k * delta
    argu1 = delta / s + s * sigma
    dy0s = (delta - y_0) / s
    argu2 = dy0s + s * sigma
    cdf1 = bs_norm_cdf(argu1)
    cdf2 = bs_norm_cdf(argu2)
    y0sig = sigma * y_0
    val_expB = np.exp(sigma * (s * s * sigma / 2.0 + delta))
    val_compB = p_0 * (cdf1 - cdf2) * val_expB
    val_expC = np.exp(y0sig)
    cdf_d1 = bs_norm_cdf(dy0s)
    val_compC = p_0 * cdf_d1 * val_expC
    if not gr:
        return val_compB + val_compC
    else:
        pdf2 = bs_norm_pdf(argu2)
        pdf_d1 = bs_norm_pdf(dy0s)
        grad = np.zeros(1)
        grad[0] = p_0 * (pdf2 * val_expB / s + (cdf_d1 * sigma - pdf_d1 / s) * val_expC)
        return val_compB + val_compC, grad


def val_BC_all(
    model: ScreeningModel,
    y: np.ndarray,
    sigma: float,
    s: float,
    k: float,
    gr: bool = False,
) -> Any:
    deltas = model.theta_mat[:, 0]
    p_0 = k * deltas
    argu1 = deltas / s + s * sigma
    dy0s = np.subtract.outer(deltas, y) / s
    argu2 = dy0s + s * sigma
    cdf1a = cast(np.ndarray, bs_norm_cdf(argu1))
    cdf2 = bs_norm_cdf(argu2)
    y0sig = sigma * y
    cdf_d1 = bs_norm_cdf(dy0s)
    val_expBa = np.exp(sigma * (s * s * sigma / 2.0 + deltas))
    val_compB = (-cdf2 + cdf1a.reshape((-1, 1))) * (p_0 * val_expBa).reshape((-1, 1))
    val_expC = np.exp(y0sig)
    val_compC = cdf_d1 * val_expC * p_0.reshape((-1, 1))
    if not gr:
        return val_compB + val_compC
    else:
        pdf2 = bs_norm_pdf(argu2)
        pdf_d1 = bs_norm_pdf(dy0s)
        grad = np.zeros((1, deltas.size, y.size))
        grad[0, :, :] = (
            pdf2 * val_expBa.reshape((-1, 1)) / s
            + (cdf_d1 * sigma - pdf_d1 / s) * val_expC
        ) * p_0.reshape((-1, 1))
    return val_compB + val_compC, grad


def val_D(y: np.ndarray, delta: float, s: float, k: float, gr: bool = False) -> Any:
    """evaluates `D`, the actuarial premium

    Args:
        y: a  1-vector with a deductible value
        delta: a risk location parameter
        s: the dispersion of losses
        k: `p_0/delta`
        if `gr` we also return its gradient  wrt `y`

    Returns:
        the value of `D`
            and its gradient  wrt `y` if `gr` is `True`
    """
    dy0s = (delta - y[0]) / s
    p_0 = k * delta
    s_H = s * H_fun(dy0s)
    val_comp = p_0 * s_H
    if not gr:
        return val_comp
    else:
        grad = np.zeros(1)
        grad[0] = -p_0 * bs_norm_cdf(dy0s)
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
        y:  a `k`-vector of deductible values
        theta: if provided, should be a 1-vector with the risk location of one type;
            then `k` should equal 1
        gr: if `True`, we also return the gradient

    Returns:
        if `theta` is provided, the value of `I(y,theta,s)` for this type and contract;
        otherwise, the values of `I(y,t,s)` for all types and for all contracts in `y` as an $(N, k)$ matrix
        if `gr` is `True` we also return the gradient.
    """
    check_args("val_I", y, 1, 1, theta)
    params = cast(np.ndarray, model.params)
    _, s, _, k = params
    if theta is not None:
        return val_I_1(model, y, theta, s, k, gr=gr)
    else:
        return val_I_all(model, y, s, k, gr=gr)


def val_I_1(
    model: ScreeningModel,
    y: np.ndarray,
    theta: np.ndarray,
    s: float,
    k: float,
    gr: bool = False,
) -> Any:
    """`val_I` for one type and one contract"""
    delta = theta[0]
    value_A = cast(float, val_A(delta, s, k))
    value_BC = val_BC(model, y, theta=theta, gr=gr)
    if not gr:
        return value_BC + value_A
    else:
        val, grad = value_BC
        return val + value_A, grad


def val_I_all(
    model: ScreeningModel, y: np.ndarray, s: float, k: float, gr: bool = False
) -> Any:
    """`val_I` for all types and all contracts in `y`"""
    deltas = model.theta_mat[:, 0]
    values_A = cast(np.ndarray, val_A(deltas, s, k))
    values_BC = val_BC(model, y, gr=gr)
    if not gr:
        obj = values_BC + values_A.reshape((-1, 1))
        return obj
    else:
        vals, grad = values_BC
        return vals + values_A.reshape((-1, 1)), grad


def S_penalties(y: np.ndarray, gr: bool = False) -> Any:
    """penalties to keep minimization of `S` within bounds; with gradient if `gr` is `True`

    Args:
        y:  a deductible value
        gr: whether we compute the gradient

    Returns:
        a scalar, the total value of the penalties;
        and a `1`-vector of derivatives if `gr` is `True`
    """
    y_0 = y[0]
    y_0_neg = min(y_0, 0.0)
    val_penalties = (
        coeff_qpenalty_S0 * y_0 * y_0 + coeff_qpenalty_S0_0 * y_0_neg * y_0_neg
    )
    if not gr:
        return val_penalties
    else:
        grad = np.zeros(1)
        grad[0] = 2.0 * coeff_qpenalty_S0 * y_0 + 2.0 * coeff_qpenalty_S0_0 * y_0_neg
        return val_penalties, grad


def proba_claim(deltas, s, k):
    return k * deltas * bs_norm_cdf(deltas / s)


def expected_positive_loss(deltas, s):
    return s * bs_norm_pdf(deltas / s) / bs_norm_cdf(deltas / s) + deltas


def cost_non_insur(model):
    sigma = cast(np.ndarray, model.params)[0]
    y_no_insur = np.array([20.0])
    return np.log(val_I(model, y_no_insur))[:, 0] / sigma