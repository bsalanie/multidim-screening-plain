"""We compute the solution for a 1-dimensionsal model directly as a check.
This has type=risk, contract=deductible.
"""


from pathlib import Path

import numpy as np
from bs_python_utils.bs_opt import minimize_free
from dotenv import dotenv_values
from insurance_d1_m1_risk_deduc_values import val_I

from multidim_screening_plain.classes import ScreeningModel
from multidim_screening_plain.setup import setup_model

# load configuration
config = dotenv_values(
    Path.cwd() / "multidim_screening_plain" / "config_insurance_d1_m1_risk_deduc.env"
)
model = setup_model(config)
module = model.model_module

S_fun = model.S_function
N = model.N
sigma = model.params[0]


def v_fun(y, args, gr=False):
    theta, theta1, i = args
    S_vals = S_fun(model, y, theta=theta, gr=gr)
    I_vals = val_I(model, y, theta=theta, gr=gr)
    I_vals1 = val_I(model, y, theta=theta1, gr=gr)
    if gr:
        S_val, S_grad = S_vals
        I_val, I_grad = I_vals
        I_val1, I_grad1 = I_vals1
        CE_val = -np.log(I_val) / sigma
        CE_val1 = -np.log(I_val1) / sigma
        CE_grad = -I_grad / (I_val * sigma)
        CE_grad1 = -I_grad1 / (I_val1 * sigma)
        return -S_val + (N - 1 - i) * (CE_val - CE_val1), -S_grad + (N - 1 - i) * (
            CE_grad - CE_grad1
        )
    else:
        CE_val = -np.log(I_vals) / sigma
        CE_val1 = -np.log(I_vals1) / sigma
        return -S_vals + (N - 1 - i) * (CE_val - CE_val1)


def v_grad(y, args):
    return v_fun(y, args, gr=True)[1]


thetas = model.theta_mat
y0 = np.array([1.0])
args = [thetas[7], thetas[8], 7]
# check_gradient_scalar_function(v_fun, y0, args)


def compute_second_best(model: ScreeningModel) -> np.ndarray:
    y_second = np.zeros((N, 1))
    args = [thetas[-1, :], thetas[-1, :], N - 1]
    res = minimize_free(v_fun, v_grad, y0, args)
    y_second[-1] = res.x
    for j in range(N - 2, -1, -1):
        args = [thetas[j, :], thetas[j + 1, :], j]
        res = minimize_free(v_fun, v_grad, y0, args)
        if v_fun(res.x, args) > 0.0:
            y_second[j, 0] = res.x
        else:
            y_second[:j, 0] = 10.0
            break
    return y_second


y_second = compute_second_best(model)
print(y_second)
