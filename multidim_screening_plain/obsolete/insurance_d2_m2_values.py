"""computes the components of the utilities for the insurance model
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

# penalties to keep minimization of `S` within bounds
coeff_qpenalty_S0 = 0.00001  # coefficient of the quadratic penalty on S for y0 large
coeff_qpenalty_S0_0 = 1_000.0  # coefficient of the quadratic penalty on S for y0<0
coeff_qpenalty_S1_0 = 1_000.0  # coefficient of the quadratic penalty on S for y1<0
coeff_qpenalty_S1_1 = 1_000.0  # coefficient of the quadratic penalty on S for y1>1
coeff_qpenalty_S01_0 = (
    1_000.0  # coefficient of the quadratic penalty on S for y0 + y1 small
)


def val_A(deltas: np.ndarray | float, s: float) -> np.ndarray | float:
    """evaluates `A(delta,s)`, the probability that the loss is less than the deductible
    for all values of `delta` in `deltas`

    Args:
        deltas: a `q`-vector of risk parameters, or a single value
        s: the dispersion of losses

    Returns:
        the values of `A(delta,s)` as a `q`-vector, or a single value
    """
    return bs_norm_cdf(-deltas / s)


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
        # print(f"{y=}, {theta=}")
        y_0, y_1 = y[0], y[1]
        sigma, delta = theta[0], theta[1]
        s = cast(np.ndarray, model.params)[0]
        argu1 = delta / s + s * sigma
        dy0s = (delta - y_0) / s
        argu2 = dy0s + s * sigma
        cdf1 = bs_norm_cdf(argu1)
        cdf2 = bs_norm_cdf(argu2)
        y1sig = sigma * y_1
        y01sig = sigma * y_0 * (1 - y_1)
        ny1sig = sigma * (1 - y_1)
        val_expB = np.exp(sigma * (s * s * sigma / 2.0 + delta))
        val_compB = (cdf1 - cdf2) * val_expB
        val_expC = np.exp(y1sig * (s * s * y1sig / 2.0 + delta) + y01sig)
        d1 = dy0s + s * y1sig
        cdf_d1 = bs_norm_cdf(d1)
        val_compC = cdf_d1 * val_expC
        if not gr:
            return val_compB + val_compC
        else:
            pdf2 = bs_norm_pdf(argu2)
            pdf_d1 = bs_norm_pdf(d1)
            grad = np.zeros(2)
            grad[0] = pdf2 * val_expB / s + (cdf_d1 * ny1sig - pdf_d1 / s) * val_expC
            grad[1] = s * H_fun(d1) * val_expC * sigma
            return val_compB + val_compC, grad
    else:
        # precalculated_values = model.precalculated_values
        y_0, y_1 = split_y(y, 2)
        theta_mat = model.theta_mat
        sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
        s = cast(np.ndarray, model.params)[0]
        # argu1 = precalculated_values["argu1"]
        argu1 = deltas / s + s * sigmas
        dy0s = np.subtract.outer(deltas, y_0) / s
        argu2 = dy0s + s * sigmas.reshape((-1, 1))
        cdf1a = cast(np.ndarray, bs_norm_cdf(argu1))
        # cdf1a = precalculated_values["cdf1"]
        cdf2 = bs_norm_cdf(argu2)
        y1sig = np.outer(sigmas, y_1)
        y01sig = np.outer(sigmas, y_0 * (1 - y_1))
        ny1sig = np.outer(sigmas, 1 - y_1)
        d1 = dy0s + s * y1sig
        cdf_d1 = bs_norm_cdf(d1)
        val_expBa = np.exp(sigmas * (s * s * sigmas / 2.0 + deltas))
        # val_expBa = precalculated_values["val_expB"]
        val_compB = (-cdf2 + cdf1a.reshape((-1, 1))) * val_expBa.reshape((-1, 1))
        val_expC = np.exp(
            y1sig * (s * s * y1sig / 2.0 + deltas.reshape((-1, 1))) + y01sig
        )
        val_compC = cdf_d1 * val_expC
        if not gr:
            return val_compB + val_compC
        else:
            pdf2 = bs_norm_pdf(argu2)
            pdf_d1 = bs_norm_pdf(d1)
            grad = np.zeros((2, sigmas.size, y_0.size))
            grad[0, :, :] = (
                pdf2 * val_expBa.reshape((-1, 1)) / s
                + (cdf_d1 * ny1sig - pdf_d1 / s) * val_expC
            )
            grad[1, :, :] = s * H_fun(d1) * val_expC * sigmas.reshape((-1, 1))
            return val_compB + val_compC, grad


def val_D(y: np.ndarray, delta: float, s: float, gr: bool = False) -> Any:
    """evaluates `D(y,delta,s)`, the actuarial premium

    Args:
        y: a 2-vector of 1 contract
        delta: a risk location parameter
        s: the dispersion of losses
        if `gr` we also return its gradient  wrt `y`

    Returns:
        the value of `D(y,delta,s)`
            and its gradient  wrt `y` if `gr` is `True`
    """
    y_0, y_1 = y
    dy0s = (delta - y_0) / s
    s_H = s * H_fun(dy0s)
    val_comp = s_H * (1 - y_1)
    if not gr:
        return val_comp
    else:
        grad = np.zeros(2)
        grad[0] = -bs_norm_cdf(dy0s) * (1 - y_1)
        grad[1] = -s_H
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
    # precalculated_values = model.precalculated_values
    if theta is not None:
        delta = theta[1]
        s = cast(np.ndarray, model.params)[0]
        value_A = cast(float, val_A(delta, s))
        value_BC = val_BC(model, y, theta=theta, gr=gr)
        if not gr:
            return value_BC + value_A
        else:
            val, grad = value_BC
            return val + value_A, grad
    else:
        # value_A2 = cast(np.ndarray, precalculated_values["values_A"])
        deltas = model.theta_mat[:, 1]
        s = cast(np.ndarray, model.params)[0]
        value_A2 = cast(np.ndarray, val_A(deltas, s))
        value_BC = val_BC(model, y, gr=gr)
        if not gr:
            return value_BC + value_A2.reshape((-1, 1))
        else:
            val, grad = value_BC
            return val + value_A2.reshape((-1, 1)), grad


def S_penalties(y: np.ndarray, gr: bool = False) -> Any:
    """penalties to keep minimization of `S` within bounds; with gradient if `gr` is `True`

    Args:
        y:  a 2-vector of 1 contract
        gr: whether we compute the gradient

    Returns:
        a scalar, the total value of the penalties;
        and a 2-vector of derivatives if `gr` is `True`
    """
    y_0, y_1 = y[0], y[1]
    y_0_neg = min(y_0, 0.0)
    y_1_neg = min(y_1, 0.0)
    y_1_above1 = max(y_1 - 1.0, 0.0)
    y_01_small = max(0.1 - y_0 - y_1, 0.0)
    val_penalties = (
        coeff_qpenalty_S0 * y_0 * y_0
        + coeff_qpenalty_S0_0 * y_0_neg * y_0_neg
        + coeff_qpenalty_S1_0 * y_1_neg * y_1_neg
        + coeff_qpenalty_S1_1 * y_1_above1 * y_1_above1
        + coeff_qpenalty_S01_0 * y_01_small * y_01_small
    )
    if not gr:
        return val_penalties
    else:
        grad = np.zeros(2)
        grad[0] = (
            2.0 * coeff_qpenalty_S0 * y_0
            + 2.0 * coeff_qpenalty_S0_0 * y_0_neg
            - 2.0 * coeff_qpenalty_S01_0 * y_01_small
        )
        grad[1] = (
            2.0 * coeff_qpenalty_S1_0 * y_1_neg
            + 2.0 * coeff_qpenalty_S1_1 * y_1_above1
            - 2.0 * coeff_qpenalty_S01_0 * y_01_small
        )
        return val_penalties, grad


def proba_claim(deltas, s):
    return bs_norm_cdf(deltas / s)


def expected_positive_loss(deltas, s):
    return s * bs_norm_pdf(deltas / s) / proba_claim(deltas, s) + deltas


def cost_non_insur(model):
    sigmas = model.theta_mat[:, 0]
    y_no_insur = np.array([0.0, 1.0])
    return np.log(val_I(model, y_no_insur))[:, 0] / sigmas


# def value_deductible(deduc, sigma, delta, s):
#     y = np.array([deduc, 0.0])
#     sigma_vec, delta_vec = np.array([sigma]), np.array([delta])
#     return (
#         cost_non_insur(sigma, delta, s)
#         - np.log(val_I(y, sigma_vec, delta_vec, s))[0, 0] / sigma
#     )
