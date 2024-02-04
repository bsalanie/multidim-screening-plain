"""computes the components of the utilities for the insurance model
with two-dimensional types (risk-aversion, risk) and (deductible, copay) contracts
"""

from typing import Any

import numpy as np
from scipy.stats import norm

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


def add_to_each_col(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """adds a vector to each column of a matrix

    Args:
        mat: a $(q, k)$-matrix
        vec: a $q$-vector

    Returns:
        a $(q, k)$-matrix, the result of adding `vec` to each column of `mat`
    """
    q, k = mat.shape
    c = np.empty((q, k))
    for i in range(k):
        c[:, i] = mat[:, i] + vec
    return c


def multiply_each_col(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """multiplies each column of a matrix by a vector

    Args:
        mat: a $(q, k)$-matrix
        vec: a $q$-vector

    Returns:
        a $(q, k)$-matrix, the result of multiplying each column of `mat` by  `vec`
    """
    q, k = mat.shape
    c = np.empty((q, k))
    for i in range(k):
        c[:, i] = mat[:, i] * vec
    return c


def my_outer_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """outer sum of two vectors

    Args:
        a: a $q$-vector
        b: a $k$-vector

    Returns:
        a $(q,k)$-matrix, the outer sum of `a` and `b`
    """
    q, k = a.size, b.size
    c = np.empty((q, k))
    for i in range(q):
        c[i, :] = a[i] + b
    return c


def n01_cdf_mat(a: np.ndarray) -> Any:
    """cdf of N(0,1) at values in a matrix (numba_stats.norm only takes in vectors)

    Args:
        a: a `(q,k)`-matrix

    Returns:
        the `(q,k)`-matrix $\\Phi(a)$
    """
    q, k = a.shape
    c = np.empty((q, k))
    for i in range(q):
        c[i, :] = norm.cdf(a[i, :], 0.0, 1.0)
    return c


def n01_pdf_mat(a: np.ndarray) -> Any:
    """pdf of N(0,1) at values in a matrix (numba_stats.norm only takes in vectors)

    Args:
        a: a `(q,k)`-matrix

    Returns:
        the `(q,k)`-matrix $\\phi(a)$
    """
    q, k = a.shape
    c = np.empty((q, k))
    for i in range(q):
        c[i, :] = norm.pdf(a[i, :], 0.0, 1.0)
    return c


def val_A(deltas: np.ndarray, s: float) -> Any:
    """evaluates $A(\\delta,s)$, the probability that the loss is less than the deductible
    for all values of $\\delta$ in `deltas`

    Args:
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        the value of $A(\\delta,s)$ as a $q$-vector
    """
    return norm.cdf(-deltas / s, 0.0, 1.0)


def val_B(y: np.ndarray, sigmas: np.ndarray, deltas: np.ndarray, s: float) -> Any:
    """evaluates $B(y,\\sigma,\\delta,s)$ for all values in `y` and `(sigmas, deltas)`

    Args:
        y: a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        the values of $B(y,\\sigma,\\delta,s)$ as a $(q, k)$ matrix
    """
    y_0, _ = split_y(y)
    argu1 = -deltas / s - sigmas * s
    argu2 = my_outer_add(
        argu1,
        y_0 / s,
    )
    cdf1 = norm.cdf(argu1, 0.0, 1.0)
    val_comp = multiply_each_col(
        add_to_each_col(n01_cdf_mat(argu2), -cdf1),
        np.exp((s * sigmas) ** 2 / 2 + sigmas * deltas),
    )
    return val_comp


def val_C(y: np.ndarray, sigmas: np.ndarray, deltas: np.ndarray, s: float) -> Any:
    """evaluates $C(y,\\sigma,\\delta,s)$

    Args:
        y:  a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        the value of $C(y,\\sigma,\\delta,s)$ as a $(q, k)$ matrix
    """
    y_0, y_1 = split_y(y)
    dy0s = my_outer_add(deltas, -y_0) / s
    y1sig = np.outer(sigmas, y_1)
    y01sig = np.outer(sigmas, y_0 * (1 - y_1))
    val_comp = n01_cdf_mat(dy0s + s * y1sig) * np.exp(
        (s * y1sig) ** 2 / 2 + multiply_each_col(y1sig, deltas) + y01sig
    )
    return val_comp


def val_D(y: np.ndarray, deltas: np.ndarray, s: float) -> Any:
    """evaluates $D(y,\\delta,s)$, the actuarial premium

    Args:
        y: a $2 k$-vector of $k$ contracts
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        the value of $D(y,\\sigma,\\delta,s)$ as a $(q, k)$ matrix
    """
    y_0, y_1 = split_y(y)
    dy0s = my_outer_add(deltas, -y_0) / s
    val_comp = s * (n01_pdf_mat(dy0s) + dy0s * n01_cdf_mat(dy0s)) * (1 - y_1)
    return val_comp


def val_I(y: np.ndarray, sigmas: np.ndarray, deltas: np.ndarray, s: float) -> Any:
    """computes the integral $I$ for all values in `y` and `(sigmas, deltas)`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        the value of $I(y,\\sigma,\\delta,s)$ as a $(q, k)$ matrix
    """
    return add_to_each_col(
        val_B(y, sigmas, deltas, s) + val_C(y, sigmas, deltas, s), val_A(deltas, s)
    )


def S_penalties(y: np.ndarray):
    """penalties to keep minimization of `S` within bounds

    Args:
        y:  a $2 k$-vector of $k$ contracts

    Returns:
        a scalar, the total value of the penalties
    """
    y_0, y_1 = split_y(y)
    y_0_neg = np.minimum(y_0, 0.0)
    y_1_neg = np.minimum(y_1, 0.0)
    y_1_above1 = np.maximum(y_1 - 1.0, 0.0)
    y_01_small = np.maximum(0.1 - y_0 - y_1, 0.0)
    return (
        coeff_qpenalty_S0 * np.sum(y_0 * y_0)
        + coeff_qpenalty_S0_0 * np.sum(y_0_neg * y_0_neg)
        + coeff_qpenalty_S1_0 * np.sum(y_1_neg * y_1_neg)
        + coeff_qpenalty_S1_1 * np.sum(y_1_above1 * y_1_above1)
        + coeff_qpenalty_S01_0 * np.sum(y_01_small * y_01_small)
    )


def d0_val_B(y, sigmas, deltas, s):
    """evaluates the derivative of `B` wrt `y_0`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        a $(q, k)$-matrix
    """
    y_0, _ = split_y(y)
    sigma_s = s * sigmas
    dy0s = my_outer_add(deltas, -y_0) / s
    return (
        multiply_each_col(
            n01_pdf_mat(add_to_each_col(dy0s, sigma_s)),
            np.exp(sigma_s * sigma_s / 2.0 + sigmas * deltas),
        )
        / s
    )


def d0_val_C(y, sigmas, deltas, s):
    """evaluates the derivative of `C` wrt `y_0`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        a $(q, k)$-matrix
    """
    y_0, y_1 = split_y(y)
    dy0s = my_outer_add(deltas, -y_0) / s
    y1sig = np.outer(sigmas, y_1)
    ny1sig = np.outer(sigmas, 1 - y_1)
    y01sig = np.outer(sigmas, y_0 * (1 - y_1))
    sigma1_s = s * y1sig
    argu = dy0s + sigma1_s
    return (ny1sig * n01_cdf_mat(argu) - n01_pdf_mat(argu) / s) * np.exp(
        sigma1_s * sigma1_s / 2 + multiply_each_col(y1sig, deltas) + y01sig
    )


def H_fun(argu: np.ndarray) -> Any:
    """computes the function $H(x)=x\\Phi(x)+\\phi(x)$

    Args:
        argu:  must be a matrix

    Returns:
        a matrix of the same shape
    """
    return argu * n01_cdf_mat(argu) + n01_pdf_mat(argu)


def d1_val_C(y, sigmas, deltas, s):
    """evaluates the derivative of `C` wrt `y_1`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        a $(q, k)$-matrix
    """
    y_0, y_1 = split_y(y)
    dy0s = my_outer_add(deltas, -y_0) / s
    y1sig = np.outer(sigmas, y_1)
    d1 = dy0s + s * y1sig
    y01sig = np.outer(sigmas, y_0 * (1 - y_1))
    sigma1_s = s * y1sig
    return s * multiply_each_col(
        H_fun(d1)
        * np.exp(sigma1_s * sigma1_s / 2.0 + multiply_each_col(y1sig, deltas) + y01sig),
        sigmas,
    )


def d0_val_D(y, deltas, s):
    """evaluates the derivative of `D` wrt `y_0`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        a $(q, k)$-matrix
    """
    y_0, y_1 = split_y(y)
    dy0s = my_outer_add(deltas, -y_0) / s
    return -n01_cdf_mat(dy0s) * (1 - y_1)


def d1_val_D(y, deltas, s):
    """evaluates the derivative of `D` wrt `y_1`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        a $(q, k)$-matrix
    """
    y_0, _ = split_y(y)
    dy0s = my_outer_add(deltas, -y_0) / s
    return -s * H_fun(dy0s)


def d0_b_fun(y, sigmas, deltas, s):
    """evaluates the derivative of `b` wrt `y_0`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        a $(q,k)$-matrix
    """
    return -multiply_each_col(
        (d0_val_B(y, sigmas, deltas, s) + d0_val_C(y, sigmas, deltas, s))
        / val_I(y, sigmas, deltas, s),
        1.0 / sigmas,
    )


def d1_b_fun(y, sigmas, deltas, s):
    """evaluates the derivative of `b` wrt `y_1`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        a $(q,k)$-matrix
    """
    return -multiply_each_col(
        d1_val_C(y, sigmas, deltas, s) / val_I(y, sigmas, deltas, s),
        (1.0 / sigmas),
    )


def d0_S_penalties(y: np.ndarray):
    """derivatives wrt $y_0$ of the penalties to keep minimization of `S` within bounds

    Args:
        y: a $2 k$-vector of $k$ contracts

    Returns:
        a $k$-vector, the derivative of the total penalty wrt `y_0`
    """
    y_0, y_1 = split_y(y)
    y_0_neg = np.minimum(y_0, 0.0)
    y_01_small = np.maximum(0.1 - y_0 - y_1, 0.0)
    return (
        2.0 * coeff_qpenalty_S0 * y_0
        + 2.0 * coeff_qpenalty_S0_0 * y_0_neg
        - 2.0 * coeff_qpenalty_S01_0 * y_01_small
    )


def d1_S_penalties(y: np.ndarray):
    """derivatives wrt $y_1$ of the penalties to keep minimization of `S` within bounds

    Args:
        y:  a $2 k$-vector of $k$ contracts

    Returns:
        a $k$-vector, the derivative of the total penalty wrt `y_1`
    """
    y_0, y_1 = split_y(y)
    y_1_neg = np.minimum(y_1, 0.0)
    y_1_above1 = np.maximum(y_1 - 1.0, 0.0)
    y_01_small = np.maximum(0.1 - y_0 - y_1, 0.0)
    return (
        2.0 * coeff_qpenalty_S1_0 * y_1_neg
        + 2.0 * coeff_qpenalty_S1_1 * y_1_above1
        - 2.0 * coeff_qpenalty_S01_0 * y_01_small
    )


def d0_S_fun(y, sigmas, deltas, s, loading):
    """evaluates the derivative of `S` wrt `y_0`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        a $(q,k)$-matrix
    """
    # print("y=", y)
    # print("sigmas=", sigmas)
    # print("deltas=", deltas)
    # print("d0_b_fun(y, sigmas, deltas, s)=", d0_b_fun(y, sigmas, deltas, s))
    # print("d0_val_D(y, deltas, s)=", d0_val_D(y, deltas, s))
    # print("d0_S_penalties(y)=", d0_S_penalties(y))
    d0_S_val = (
        d0_b_fun(y, sigmas, deltas, s)
        - (1.0 + loading) * d0_val_D(y, deltas, s)
        - d0_S_penalties(y)
    )
    # print(f"{d0_S_val=}")
    return d0_S_val


def d1_S_fun(y, sigmas, deltas, s, loading):
    """evaluates the derivative of `S` wrt `y_1`

    Args:
        y:  a $2 k$-vector of $k$ contracts
        sigmas: a $q$-vector of risk-aversion parameters
        deltas: a $q$-vector of risk parameters
        s: the dispersion of losses

    Returns:
        a $(q,k)$-matrix
    """
    return (
        d1_b_fun(y, sigmas, deltas, s)
        - (1.0 + loading) * d1_val_D(y, deltas, s)
        - d1_S_penalties(y)
    )


def proba_claim(deltas, s):
    return norm.cdf(deltas / s, 0.0, 1.0)


def expected_positive_loss(deltas, s):
    return s * norm.pdf(deltas / s, 0.0, 1.0) / proba_claim(deltas, s) + deltas


def cost_non_insur(sigmas, deltas, s):
    y_no_insur = np.array([0.0, 1.0])
    return np.log(val_I(y_no_insur, sigmas, deltas, s))[:, 0] / sigmas
