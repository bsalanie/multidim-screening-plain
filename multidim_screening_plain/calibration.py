from itertools import product

import numpy as np
from scipy.stats import norm

from multidim_screening_plain.insurance_d2_m2_values import val_I


def proba_accident(delta, s):
    return norm.cdf(delta / s, 0.0, 1.0)


def expected_positive_loss(delta, s):
    return s * norm.pdf(delta / s, 0.0, 1.0) / proba_accident(delta, s) + delta


def cost_non_insur(sigma, delta, s):
    y_no_insur = np.array([0.0, 1.0])
    sigma_vec, delta_vec = np.array([sigma]), np.array([delta])
    return np.log(val_I(y_no_insur, sigma_vec, delta_vec, s))[0, 0] / sigma


def value_deductible(deduc, sigma, delta, s):
    y = np.array([deduc, 0.0])
    sigma_vec, delta_vec = np.array([sigma]), np.array([delta])
    return (
        cost_non_insur(sigma, delta, s)
        - np.log(val_I(y, sigma_vec, delta_vec, s))[0, 0] / sigma
    )


for deltai, s, sig10, deduc1000 in product(
    range(-7, -2), [4.0], range(2, 6), [500.0, 1000.0]
):
    delta = float(deltai)
    sigma = sig10 / 10.0
    deduc = deduc1000 / 1_000.0
    print(f"For {delta=}, {s=}, {sigma=}:")
    cost_zero = cost_non_insur(sigma, delta, s)
    val_deduc = value_deductible(deduc, sigma, delta, s)
    print(f"   accident proba = {proba_accident(delta, s): 10.3f}")
    print(f"   expected positive loss = {expected_positive_loss(delta, s): 10.3f}")
    print(f"   cost of non-insurance = {cost_zero: 10.3f}")
    print(f"   value of deductible {deduc}:  {val_deduc: 10.3f}")
