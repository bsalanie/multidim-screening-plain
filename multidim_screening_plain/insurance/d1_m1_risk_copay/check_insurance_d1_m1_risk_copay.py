"""direct calculation of the 1st best"""

import numpy as np
from bs_python_utils.bs_opt import check_gradient_scalar_function, minimize_free

from multidim_screening_plain.utils import H_fun, bs_norm_cdf

sigma, s, loading, k = 0.6, 2.0, 0.25, 0.012

dim_grid = 50
deltas = np.linspace(4.0, 8.0, num=dim_grid)
p_1 = k * deltas

vals_A = 1.0 - p_1 * bs_norm_cdf(deltas / s)


def obj_grad_FB(yv: np.ndarray, args: list, gr: bool = False):
    p_1v, delta, val_A = args
    sig_y = sigma * yv
    val_exp = np.exp(sig_y * (delta + sig_y * s * s / 2.0))
    argu1 = sig_y * s + delta / s
    cdf1 = bs_norm_cdf(argu1)
    produ = p_1v * val_exp * cdf1 + val_A
    argu2 = delta / s
    H_2 = H_fun(argu2)
    val = np.log(produ) / sigma + (1.0 + loading) * (1.0 - yv) * p_1v * s * H_2
    if gr:
        grad_produ = p_1v * s * sigma * H_fun(argu1) * val_exp
        grad = grad_produ / produ / sigma - (1.0 + loading) * p_1v * s * H_2
        return val, grad
    else:
        return val


def obj_FB(y: np.ndarray, args: list):
    return obj_grad_FB(y, args)


def grad_FB(y: np.ndarray, args: list):
    return obj_grad_FB(y, args, gr=True)[1]


x = np.zeros(dim_grid)

for i in range(dim_grid):
    args = [p_1[i], deltas[i], vals_A[i]]
    y_init = np.array([0.2])
    check_gradient_scalar_function(obj_grad_FB, y_init, args=args)
    res_FB = minimize_free(obj_FB, grad_FB, y_init, args=args)
    print(res_FB)
    x[i] = res_FB.x


print(x)
