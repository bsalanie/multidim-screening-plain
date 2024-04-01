"""direct calculation of the 1st best,
and of derivatives of the insuree's gross utility
"""

import numpy as np
from bs_python_utils.bs_opt import check_gradient_scalar_function, minimize_free
from bs_python_utils.bsutils import print_stars

from multidim_screening_plain.utils import H_fun, bs_norm_cdf, bs_norm_pdf, print_matrix

sigma, s, loading, k = 0.6, 2.0, 0.25, 0.012

dim_grid = 10
deltas = np.linspace(4.0, 8.0, num=dim_grid)
p_1 = k * deltas

vals_A = 1.0 - p_1 * bs_norm_cdf(deltas / s)
grads_A = -p_1 * bs_norm_pdf(deltas / s) / s - k * bs_norm_cdf(deltas / s)


def gross_utility(yv, args):
    p_1v, delta, val_A, grad_A = args
    dy0s = (delta - yv) / s
    val_expB = np.exp(sigma * (delta + sigma * s * s / 2.0))
    argu1 = sigma * s + dy0s
    argu2 = sigma * s + delta / s
    cdf1, pdf1 = bs_norm_cdf(argu1), bs_norm_pdf(argu1)
    cdf2, pdf2 = bs_norm_cdf(argu2), bs_norm_pdf(argu2)
    cdf0, pdf0 = bs_norm_cdf(dy0s), bs_norm_pdf(dy0s)
    val_expC = np.exp(sigma * yv)
    BC_term = val_expB * (cdf2 - cdf1) + val_expC * cdf0
    I_val = p_1v * BC_term + val_A
    argu2 = delta / s
    val = -np.log(I_val) / sigma
    grad_Iy = p_1v * (pdf1 * val_expB / s + (sigma * cdf0 - pdf0 / s) * val_expC)
    grad_y = -grad_Iy / I_val / sigma
    grad_Id = (
        grad_A
        + k * BC_term
        + p_1v
        * (val_expB * (sigma * (cdf2 - cdf1) + (pdf2 - pdf1) / s) + val_expC * pdf0 / s)
    )
    grad_d = -grad_Id / I_val / sigma
    cross_dy = k * (
        val_expB * pdf1 / s + val_expC * (sigma * cdf0 - pdf0 / s)
    ) + p_1v * (
        val_expC * pdf0 * (sigma * s + dy0s) - val_expB * pdf1 * (sigma * s + argu1)
    ) / (
        s * s
    )
    cross_dy = sigma * grad_d * grad_y - cross_dy / I_val / sigma
    return val, grad_y, grad_d, cross_dy


def obj_grad_FB(yv: np.ndarray, args: list, gr: bool = False):
    p_1v, delta, val_A, grad_A = args
    val, grad, *_ = gross_utility(yv, args)
    val, grad = -val, -grad
    dy0s = (delta - yv) / s
    cdf0, H_0 = bs_norm_cdf(dy0s), H_fun(dy0s)
    val += (1.0 + loading) * p_1v * s * H_0
    grad -= (1.0 + loading) * p_1v * cdf0
    if gr:
        return val, grad
    else:
        return val


def obj_FB(y: np.ndarray, args: list):
    return obj_grad_FB(y, args)


def grad_FB(y: np.ndarray, args: list):
    return obj_grad_FB(y, args, gr=True)[1]


x = np.zeros(dim_grid)

gu = []
for i in range(dim_grid):
    args = [p_1[i], deltas[i], vals_A[i], grads_A[i]]
    y_init = np.array([0.2])
    check_gradient_scalar_function(obj_grad_FB, y_init, args=args)
    res_FB = minimize_free(obj_FB, grad_FB, y_init, args=args, bounds=[(0.0, 1.0)])
    print(res_FB)
    x[i] = res_FB.x[0]
    gu.append(gross_utility(res_FB.x, args))


grad_y = np.array([g[1] for g in gu])
grad_d = np.array([g[2] for g in gu])
cross_dy = np.array([g[3] for g in gu])
print_stars("FIRST BEST")
print("     delta      y_1         u_y       u_d        u_dy")
print_matrix(np.column_stack((deltas, x, grad_y, grad_d, cross_dy)))
