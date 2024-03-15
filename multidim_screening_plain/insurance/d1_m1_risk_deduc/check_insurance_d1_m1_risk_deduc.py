"""We compute the solution for a 1-dimensional model directly as a check.
This has type=risk, contract=deductible.
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from bs_python_utils.bs_opt import minimize_free
from bs_python_utils.bsnputils import ThreeArrays
from dotenv import dotenv_values
from rich.console import Console
from rich.table import Table

from multidim_screening_plain.classes import ScreeningModel
from multidim_screening_plain.insurance_d1_m1_risk_deduc_values import val_I
from multidim_screening_plain.setup import setup_model

# load configuration
config = dotenv_values(
    Path.cwd() / "multidim_screening_plain" / "config_insurance_d1_m1_risk_deduc.env"
)
model = setup_model(config)
module = model.model_module

S_fun = model.S_function
N = model.N
sigma = cast(np.ndarray, model.params)[0]


def w_fun(y, args, gr=False):
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
        return -S_val - (N - 1 - i) * (CE_val - CE_val1), -S_grad - (N - 1 - i) * (
            CE_grad - CE_grad1
        )
    else:
        CE_val = -np.log(I_vals) / sigma
        CE_val1 = -np.log(I_vals1) / sigma
        return -S_vals - (N - 1 - i) * (CE_val - CE_val1)


def w_grad(y, args):
    return w_fun(y, args, gr=True)[1]


thetas = model.theta_mat
y0 = np.array([1.0])
y_no_insurance = np.array([20.0])


def compute_second_best(model: ScreeningModel) -> ThreeArrays:
    y_second = np.zeros(N)
    w_second = np.zeros(N)
    args = [thetas[-1, :], thetas[-1, :], N - 1]
    res = minimize_free(w_fun, w_grad, y0, args)
    y_second[-1] = res.x[0]
    w_second[-1] = res.fun
    for k in range(N - 2, -1, -1):
        args = [thetas[k, :], thetas[k + 1, :], k]
        res = minimize_free(w_fun, w_grad, y0, args)
        y_second[k] = res.x[0]
        w_second[k] = res.fun

    CE_no_insur = np.zeros(N)
    for j in range(N):
        CE_no_insur[j] = -np.log(val_I(model, y_no_insurance, thetas[j, :])) / sigma

    for j in range(N - 1, 0, -1):
        print(f"{j=}: {y_second[j]=}")
        dw = w_second[j] - CE_no_insur[j]
        print(f"{dw=}")
        if dw < 0:
            y_second[: (j + 1)] = 20.0
            break

    return y_second, w_second, CE_no_insur


y_second, w_second, CE_no_insur = compute_second_best(model)

y_algorithm = (
    pd.read_csv(model.resdir / "second_best_contracts.csv", header=None)
    .values[1:]
    .flatten()
    .astype(float)
)

print(y_algorithm)

console = Console()

console.print("\n" + "-" * 80 + "\n", style="bold blue")

table = Table(title="Checking the second-best for insurance_d1_m1_risk_deduc")
table.add_column("Deductible (algorithm)", justify="center", style="blue", no_wrap=True)
table.add_column(
    "Deductible (direct calculation)", justify="center", style="red", no_wrap=True
)
table.add_column("Gross virtual surplus", justify="center", style="blue", no_wrap=True)
table.add_column("Reservation value", justify="center", style="green", no_wrap=True)

for y_algo, y, w, zero in zip(
    y_algorithm, y_second, w_second, CE_no_insur, strict=True
):
    table.add_row(
        f"{y_algo: > 8.3f}",
        f"{y: > 8.3f}",
        f"{w: > 8.3f}",
        f"{zero: > 8.3f}",
    )

console.print(table)
