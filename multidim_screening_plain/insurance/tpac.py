"""Vectorized joint surplus of the TPAC insurance model, for many (type, contract) pairs.

The contract `R(L) = (1 - y_1) max(L - y_0, 0)` covers all TPAC instances:
straight deductibles have `y_1 = 0` and proportional copays have `y_0 = 0`.
The formulas are those of the Appendix of the paper and of `d2_m2/insurance_d2_m2_values.py`.
"""

import numpy as np

from multidim_screening_plain.utils import H_fun, bs_norm_cdf, bs_norm_pdf


def tpac_S_and_grad(
    sigma: np.ndarray | float,
    delta: np.ndarray,
    y_0: np.ndarray,
    y_1: np.ndarray,
    s: float,
    k: float,
    loading: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """the joint surplus `S` and its derivatives wrt `y_0` and `y_1`, elementwise

    Args:
        sigma: risk aversions (or a single one)
        delta: risk locations
        y_0: deductibles
        y_1: copays
        s: the dispersion of losses
        k: `p_1/delta`
        loading: the loading factor

    Returns:
        `S`, `dS/dy_0` and `dS/dy_1`, arrays of the common shape
    """
    p_1 = k * delta
    value_A = 1.0 - p_1 * bs_norm_cdf(delta / s)
    argu1 = delta / s + s * sigma
    dy0s = (delta - y_0) / s
    argu2 = dy0s + s * sigma
    y1sig = sigma * y_1
    d1 = dy0s + s * y1sig
    cdf1 = bs_norm_cdf(argu1)
    cdf_d1 = bs_norm_cdf(d1)
    val_expB = np.exp(sigma * (s * s * sigma / 2.0 + delta))
    val_expC = np.exp(y1sig * (s * s * y1sig / 2.0 + delta) + sigma * y_0 * (1.0 - y_1))
    val_I = value_A + p_1 * (cdf1 - bs_norm_cdf(argu2)) * val_expB
    val_I += p_1 * cdf_d1 * val_expC
    val_I_no_insurance = value_A + p_1 * cdf1 * val_expB
    dI_0 = p_1 * (
        bs_norm_pdf(argu2) * val_expB / s
        + (cdf_d1 * sigma * (1.0 - y_1) - bs_norm_pdf(d1) / s) * val_expC
    )
    dI_1 = s * p_1 * H_fun(d1) * val_expC * sigma
    b = (np.log(val_I_no_insurance) - np.log(val_I)) / sigma
    db_0 = -dI_0 / (sigma * val_I)
    db_1 = -dI_1 / (sigma * val_I)

    s_H = s * H_fun(dy0s)
    D = s_H * p_1 * (1.0 - y_1)
    dD_0 = -bs_norm_cdf(dy0s) * (1.0 - y_1) * p_1
    dD_1 = -s_H * p_1

    factor = 1.0 + loading
    S = b - factor * D
    return S, db_0 - factor * dD_0, db_1 - factor * dD_1
