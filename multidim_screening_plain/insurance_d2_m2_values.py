"""computes the components of the utilities for the insurance model
with two-dimensional types (risk-aversion, risk) and (deductible, copay) contracts
"""

from typing import Any

import numpy as np
from scipy.stats import norm

from multidim_screening_plain.utils import (
    bs_norm_cdf,
    bs_norm_pdf,
)

# penalties to keep minimization of `S` within bounds
coeff_qpenalty_S0 = 0.00001  # coefficient of the quadratic penalty on S for y0 large
coeff_qpenalty_S0_0 = 1_000.0  # coefficient of the quadratic penalty on S for y0<0
coeff_qpenalty_S1_0 = 1_000.0  # coefficient of the quadratic penalty on S for y1<0
coeff_qpenalty_S1_1 = 1_000.0  # coefficient of the quadratic penalty on S for y1>1
coeff_qpenalty_S01_0 = (
    1_000.0  # coefficient of the quadratic penalty on S for y0 + y1 small
)


def split_y(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split y into two halves of equal length (deductibles and copays)"""
    N = y.size // 2
    y_0, y_1 = y[:N], y[N:]
    return y_0, y_1


def H_fun(argu: np.ndarray) -> Any:
    """computes the function $H(x)=x\\Phi(x)+\\phi(x)$

    Args:
        argu:  must be a matrix

    Returns:
        a matrix of the same shape
    """
    # return argu * n01_cdf_mat(argu) + n01_pdf_mat(argu)
    # return argu * norm.cdf(argu) + norm.pdf(argu)
    return argu * bs_norm_cdf(argu) + bs_norm_pdf(argu)


def val_A(deltas: np.ndarray, s: float) -> Any:
    """evaluates $A(\\delta,s)$, the probability that the loss is less than the deductible
    for all values of $\\delta$ in `deltas`

    Args:
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        the value of $A(\\delta,s)$ as a $q$-vector
    """
    # return norm.cdf(-deltas / s, 0.0, 1.0)
    return bs_norm_cdf(-deltas / s)


def val_BC(
    y: np.ndarray, sigmas: np.ndarray, deltas: np.ndarray, s: float, gr: bool = False
) -> Any:
    """evaluates the values of `B+C`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses
        gr: if `True`, we also return the derivatives wrt `y`

    Returns:
        the values of $(B + C)(y,\\sigma,\\delta,s)$ as a $(q, k)$ matrix
        and if `gr` is `True`, a $(2,q,k)$ array
    """
    y_0, y_1 = split_y(y)
    sigmas_s = s * sigmas
    argu1 = deltas / s + sigmas_s
    dy0s = np.subtract.outer(deltas, y_0) / s
    argu2 = dy0s + s * sigmas.reshape((-1, 1))
    cdf1 = bs_norm_cdf(argu1)
    cdf2 = bs_norm_cdf(argu2)
    y1sig = np.outer(sigmas, y_1)
    y01sig = np.outer(sigmas, y_0 * (1 - y_1))
    ny1sig = np.outer(sigmas, 1 - y_1)
    d1 = dy0s + s * y1sig
    cdf_d1 = bs_norm_cdf(d1)
    val_expB = np.exp(sigmas * (s * sigmas_s / 2.0 + deltas))
    val_compB = (-cdf2 + cdf1.reshape((-1, 1))) * val_expB.reshape((-1, 1))
    val_expC = np.exp(y1sig * (s * s * y1sig / 2.0 + deltas.reshape((-1, 1))) + y01sig)
    val_compC = cdf_d1 * val_expC
    if not gr:
        return val_compB + val_compC
    else:
        pdf2 = bs_norm_pdf(argu2)
        pdf_d1 = bs_norm_pdf(d1)
        grad = np.zeros((2, sigmas.size, y_0.size))
        grad[0, :, :] = (
            pdf2 * val_expB.reshape((-1, 1)) / s
            + (cdf_d1 * ny1sig - pdf_d1 / s) * val_expC
        )
        grad[1, :, :] = s * H_fun(d1) * val_expC * sigmas.reshape((-1, 1))
        return val_compB + val_compC, grad


def val_D(y: np.ndarray, deltas: np.ndarray, s: float, gr: bool = False) -> Any:
    """evaluates $D(y,\\delta,s)$, the actuarial premium

    Args:
        y: a $2 k$-vector of $k$ contracts
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        the value of $D(y,\\sigma,\\delta,s)$ as a $(q, k)$ matrix
    """
    y_0, y_1 = split_y(y)
    # dy0s = my_outer_add(deltas, -y_0) / s
    dy0s = np.subtract.outer(deltas, y_0) / s
    # val_comp = s * (n01_pdf_mat(dy0s) + dy0s * n01_cdf_mat(dy0s)) * (1 - y_1)
    # val_comp = s * (norm.pdf(dy0s) + dy0s * norm.cdf(dy0s)) * (1 - y_1)
    s_H = s * H_fun(dy0s)
    val_comp = s_H * (1 - y_1)
    if not gr:
        return val_comp
    else:
        grad = np.zeros((2, deltas.size, y_0.size))
        grad[0, :, :] = -bs_norm_cdf(dy0s) * (1 - y_1)
        grad[1, :, :] = -s_H
        return val_comp, grad


def val_I(
    y: np.ndarray, sigmas: np.ndarray, deltas: np.ndarray, s: float, gr: bool = False
) -> Any:
    """computes the integral $I$ for all values in `y` and `(sigmas, deltas)`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        the value of $I(y,\\sigma,\\delta,s)$ as a $(q, k)$ matrix
    """
    value_A = bs_norm_cdf(-deltas / s)
    value_BC = val_BC(y, sigmas, deltas, s, gr)
    if not gr:
        return value_BC + value_A.reshape((-1, 1))
    else:
        val, grad = value_BC
        return val + value_A.reshape((-1, 1)), grad


def S_penalties(y: np.ndarray, gr: bool = False) -> Any:
    """penalties to keep minimization of `S` within bounds; with gradient if `gr` is `True`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        gr: whether we compute the gradient

    Returns:
        a scalar, the total value of the penalties; and a $(2, k)$ matrix of derivatives
        if `gr` is `True`
    """
    y_0, y_1 = split_y(y)
    y_0_neg = np.minimum(y_0, 0.0)
    y_1_neg = np.minimum(y_1, 0.0)
    y_1_above1 = np.maximum(y_1 - 1.0, 0.0)
    y_01_small = np.maximum(0.1 - y_0 - y_1, 0.0)
    val_penalties = (
        coeff_qpenalty_S0 * np.sum(y_0 * y_0)
        + coeff_qpenalty_S0_0 * np.sum(y_0_neg * y_0_neg)
        + coeff_qpenalty_S1_0 * np.sum(y_1_neg * y_1_neg)
        + coeff_qpenalty_S1_1 * np.sum(y_1_above1 * y_1_above1)
        + coeff_qpenalty_S01_0 * np.sum(y_01_small * y_01_small)
    )
    if not gr:
        return val_penalties
    else:
        k = y.size // 2
        grad = np.zeros((2, k))
        grad[:k] = (
            2.0 * coeff_qpenalty_S0 * y_0
            + 2.0 * coeff_qpenalty_S0_0 * y_0_neg
            - 2.0 * coeff_qpenalty_S01_0 * y_01_small
        )
        grad[k:] = (
            2.0 * coeff_qpenalty_S1_0 * y_1_neg
            + 2.0 * coeff_qpenalty_S1_1 * y_1_above1
            - 2.0 * coeff_qpenalty_S01_0 * y_01_small
        )
        return val_penalties, grad


def proba_claim(deltas, s):
    return norm.cdf(deltas / s, 0.0, 1.0)


def expected_positive_loss(deltas, s):
    return s * norm.pdf(deltas / s, 0.0, 1.0) / proba_claim(deltas, s) + deltas


def cost_non_insur(sigmas, deltas, s):
    y_no_insur = np.array([0.0, 1.0])
    return np.log(val_I(y_no_insur, sigmas, deltas, s))[:, 0] / sigmas
