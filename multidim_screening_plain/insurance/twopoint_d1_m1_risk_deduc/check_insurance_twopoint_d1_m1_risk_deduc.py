# """direct calculation of the 1st best,
# and of derivatives of the insuree's gross utility
# """

# import numpy as np
# from bs_python_utils.bs_opt import check_gradient_scalar_function, minimize_free
# from bs_python_utils.bsutils import print_stars

# from multidim_screening_plain.utils import H_fun, bs_norm_cdf, bs_norm_pdf, print_matrix

# sigma, s, loading, loss = 0.6, 2.0, 0.25, 6.0

# dim_grid = 10
# probas = np.linspace(4.0, 8.0, num=dim_grid)

# vals_A = 1.0 - probas


# def gross_CE(proba, deduc):
#     expval = np.exp(sigma * deduc)
#     log_arg = 1.0 - proba + proba * expval
#     val = -np.log(log_arg) / sigma
#     grad_p = -(expval - 1.0) / log_arg / sigma
#     grad_d = -proba * expval / log_arg
#     return val, grad_p, grad_d


# def profit(proba, deduc):

#     return val, grad_p, grad_d, cross_dy


# def obj_grad_FB(yv: np.ndarray, args: list, gr: bool = False):
#     p_1v, delta, val_A, grad_A = args
#     val, grad, *_ = gross_utility(yv, args)
#     val, grad = -val, -grad
#     dy0s = (delta - yv) / s
#     cdf0, H_0 = bs_norm_cdf(dy0s), H_fun(dy0s)
#     val += (1.0 + loading) * p_1v * s * H_0
#     grad -= (1.0 + loading) * p_1v * cdf0
#     if gr:
#         return val, grad
#     else:
#         return val


# def obj_FB(y: np.ndarray, args: list):
#     return obj_grad_FB(y, args)


# def grad_FB(y: np.ndarray, args: list):
#     return obj_grad_FB(y, args, gr=True)[1]


# x = np.zeros(dim_grid)

# gu = []
# for i in range(dim_grid):
#     args = [probas[i]]
#     y_init = np.array([4.0])
#     check_gradient_scalar_function(obj_grad_FB, y_init, args=args)
#     res_FB = minimize_free(obj_FB, grad_FB, y_init, args=args, bounds=[(0.0, 10.0)])
#     print(res_FB)
#     x[i] = res_FB.x[0]
#     gu.append(gross_utility(res_FB.x, args))


# grad_y = np.array([g[1] for g in gu])
# grad_d = np.array([g[2] for g in gu])
# cross_dy = np.array([g[3] for g in gu])
# print_stars("FIRST BEST")
# print("     delta      y_1         u_y       u_d        u_dy")
# print_matrix(np.column_stack((probas, x, grad_y, grad_d, cross_dy)))
