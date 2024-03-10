from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from bs_python_utils.bs_opt import minimize_free
from bs_python_utils.bsutils import bs_error_abort

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults
from multidim_screening_plain.general_plots import general_plots
from multidim_screening_plain.jointtax_d2_m2_plots import plot_marginal_tax_rate
from multidim_screening_plain.utils import (
    check_args,
    contracts_vector,
    split_y,
)


def b_fun(
    model: ScreeningModel,
    y: np.ndarray,
    theta: np.ndarray | None = None,
    gr: bool = False,
):
    """evaluates the `b` function, and maybe its gradient

    Args:
        model: the ScreeningModel
        y:  a `2 k`-vector of $k$ contracts
        theta: a 2-vector of characteristics of one type, if provided
        gr: whether we compute the gradient

    Returns:
        if `theta` is provided then `k` should be 1, and we return b(y,theta)
            for this contract for this type
        otherwise we return an (N,k)-matrix with `b_i(y_j)` for all `N` types `i` and
            all `k` contracts `y_j` in `y`
        and if `gr` is `True` we provide the gradient wrt `y`
    """
    check_args("b_fun", y, 2, 2, theta)
    params = cast(np.ndarray, model.params)
    eta = params[2]
    if theta is not None:
        savings, hours = y
        endowment, disutility = theta
        exp_conso = np.exp(-eta * (endowment - savings))
        net_utility = (1.0 - exp_conso) / eta - disutility * hours
        if not gr:
            return net_utility
        else:
            grad = np.zeros(2)
            grad[0] = -exp_conso
            grad[1] = -disutility
            return net_utility, grad
    else:
        theta_mat = model.theta_mat
        savings, hours = split_y(y, 2)
        endowments, disutilities = theta_mat[:, 0], theta_mat[:, 1]
        exp_consos = np.exp(-eta * np.subtract.outer(endowments, savings))
        net_utilities = (1.0 - exp_consos) / eta - np.outer(disutilities, hours)
        if not gr:
            return net_utilities
        else:
            k = y.size // 2
            grad = np.empty((2, endowments.size, k))
            grad[0, :, :] = -exp_consos
            grad[1, :, :] = -np.tile(disutilities, (k, 1)).T
            return net_utilities, grad


def S_fun(model: ScreeningModel, y: np.ndarray, theta: np.ndarray, gr: bool = False):
    """evaluates the joint surplus, and maybe its gradient, for 1 contract for 1 type

    Args:
        model: the ScreeningModel
        y:  a 2-vector of 1 contract `y`
        theta: a 2-vector of characteristics of one type
        gr: whether we compute the gradient

    Returns:
        the value of `S(y,theta)` for this contract and this type,
            and its gradient wrt `y` if `gr` is `True`
    """
    check_args("S_fun", y, 2, 2, theta)
    params = cast(np.ndarray, model.params)
    w, R, eta = params
    savings, hours = y[0], y[1]
    endowment, disutility = theta[0], theta[1]
    exp_conso = np.exp(-eta * (endowment - savings))
    val_S = (1.0 - exp_conso) / eta + R * savings + (w - disutility) * hours
    if not gr:
        return val_S
    else:
        grad_S = np.zeros(2)
        grad_S[0] = -exp_conso + R
        grad_S[1] = w - disutility
        return val_S, grad_S


def create_initial_contracts(
    model: ScreeningModel,
    start_from_first_best: bool,
    y_first_best_mat: np.ndarray | None = None,
) -> tuple[np.ndarray, list]:
    """Initializes the contracts for the second best problem (MODEL-DEPENDENT)

    Args:
        model: the ScreeningModel object
        start_from_first_best: whether to start from the first best
        y_first_best_mat: the `(N, m)` matrix of first best contracts. Defaults to None.

    Returns:
        tuple[np.ndarray, list]: initial contracts (an `(m *N)` vector) and a list of types for whom
            the contracts are to be determined.
    """
    N = model.N
    if start_from_first_best:
        if y_first_best_mat is None:
            bs_error_abort("We start from the first best but y_first_best_mat is None")
        y_init = contracts_vector(cast(np.ndarray, y_first_best_mat))
        set_fixed_y: set[int] = set()
        set_not_insured: set[int] = set()
        free_y = list(range(N))
    else:
        model_resdir = cast(Path, model.resdir)
        y_init = np.loadtxt(model_resdir / "current_y.txt")
        EPS = 0.001
        set_not_insured = {i for i in range(N) if y_init[i, 1] > 1.0 - EPS}
        set_fixed_y = set_not_insured

        set_free_y = set(range(N)).difference(set_fixed_y)
        list(set_fixed_y)
        free_y = list(set_free_y)
        not_insured = list(set_not_insured)
        # only_deductible = list(set_only_deductible)
        rng = np.random.default_rng(645)

        MIN_Y0, MAX_Y0 = -np.inf, np.inf
        MIN_Y1, MAX_Y1 = 0.0, 1.0
        y_init = cast(np.ndarray, y_init)
        perturbation = 0.001
        yinit_0 = np.clip(y_init[:, 0] + rng.normal(0, perturbation, N), MIN_Y0, MAX_Y0)
        yinit_1 = np.clip(y_init[:, 1] + rng.normal(0, perturbation, N), MIN_Y1, MAX_Y1)
        yinit_0[not_insured] = 0.0
        yinit_1[not_insured] = 1.0

        y_init = cast(np.ndarray, np.concatenate((yinit_0, yinit_1)))
        model.v0 = np.loadtxt(model_resdir / "current_v.txt")

    return y_init, free_y


def proximal_operator(
    model: ScreeningModel,
    theta: np.ndarray,
    z: np.ndarray | None = None,
    t: float | None = None,
) -> np.ndarray | None:
    """Proximal operator of `-t S_i` at `z`;
        minimizes `-S_i(y) + 1/(2 t)  ||y-z||^2`

    Args:
        model: the ScreeningModel
        theta: type `i`'s characteristics, a `d`-vector
        z: a `2`-vector for a type, if any
        t: the step; if None, we maximize `S_i(y)`

    Returns:
        the minimizing `y`, a 2-vector
    """

    def prox_obj_and_grad(
        y: np.ndarray, args: list, gr: bool = False
    ) -> float | tuple[float, np.ndarray]:
        S_vals = S_fun(model, y, theta=theta, gr=gr)
        if not gr:
            obj = -S_vals
            if t is not None:
                dyz = y - z
                dist_yz2 = np.sum(dyz * dyz)
                obj += dist_yz2 / (2 * t)
            return cast(float, obj)
        if gr:
            obj, grad = -S_vals[0], -S_vals[1]
            if t is not None:
                dyz = y - z
                dist_yz2 = np.sum(dyz * dyz)
                obj += dist_yz2 / (2 * t)
                grad += dyz / t
            return cast(float, obj), cast(np.ndarray, grad)

    def prox_obj(y: np.ndarray, args: list) -> float:
        return cast(float, prox_obj_and_grad(y, args, gr=False))

    def prox_grad(y: np.ndarray, args: list) -> np.ndarray:
        return cast(tuple[float, np.ndarray], prox_obj_and_grad(y, args, gr=True))[1]

    y_init = np.array([theta[0], 1.0]) if t is None else z

    # check_gradient_scalar_function(prox_obj_and_grad, y_init, args=[])
    # bs_error_abort("done")

    mini = minimize_free(
        prox_obj,
        prox_grad,
        x_init=y_init,
        args=[],
        bounds=[(-np.inf, np.inf), (0.0, 1.0)],
    )

    if mini.success or mini.status == 2:
        y = mini.x
        return cast(np.ndarray, y)
    else:
        print(f"{mini.message}")
        bs_error_abort(f"Minimization did not converge: status {mini.status}")
        return None


def add_results(results: ScreeningResults):
    pass


def plot_results(model: ScreeningModel) -> None:
    model_resdir = cast(Path, model.resdir)
    model_plotdir = cast(Path, model.plotdir)
    df_all_results = (
        pd.read_csv(model_resdir / "all_results.csv")
        .rename(
            columns={
                "FB_y_0": "First-best Savings",
                "FB_y_1": "First-best Hours",
                "y_0": "Second-best Savings",
                "y_1": "Second-best Hours",
                "theta_0": "Endowment",
                "theta_1": "Disutility",
                "FB_surplus": "First-best surplus",
                "SB_surplus": "Second-best surplus",
                "info_rents": "Informational rent",
            }
        )
        .round(3)
    )

    general_plots(model, df_all_results)

    plot_marginal_tax_rate(
        df_all_results,
        title="Marginal tax rate on savings",
        cmap="winter",
        path=str(model_plotdir) + "/marginal_tax_rate.png",
    )
