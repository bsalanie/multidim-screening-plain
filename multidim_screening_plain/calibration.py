from itertools import product

import numpy as np

from multidim_screening_plain.insurance_d2_m2_values import (
    cost_non_insur,
    expected_positive_loss,
    proba_claim,
    value_deductible,
)

for deltai, s, sig10, deduc1000 in product(
    range(-7, -2), [4.0], range(2, 6), [500.0, 1000.0]
):
    deltas = np.array([float(deltai)])
    sigmas = np.array([sig10 / 10.0])
    deduc = deduc1000 / 1_000.0
    print(f"For delta={deltas[0]: > 8.3f}, {s=}, sigma={sigmas[0]: > 8.3f}:")
    cost_zero = cost_non_insur(sigmas, deltas, s)
    val_deduc = value_deductible(deduc, sigmas, deltas, s)
    print(f"   accident proba = {proba_claim(deltas, s)[0]: 10.3f}")
    print(f"   expected positive loss = {expected_positive_loss(deltas, s)[0]: 10.3f}")
    print(f"   cost of non-insurance = {cost_zero[0]: 10.3f}")
    print(f"   value of deductible {deduc}:  {val_deduc[0]: 10.3f}")
