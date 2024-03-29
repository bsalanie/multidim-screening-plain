from pathlib import Path
from typing import cast

import numpy as np
import scipy.optimize as spopt
from bs_python_utils.bsutils import bs_error_abort

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults
from multidim_screening_plain.jointtax.d2_m2.jointtax_d2_m2_plots import (
    plot_marginal_tax_rate,
)
from multidim_screening_plain.plot_utils import setup_for_plots
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
    else:
        model_resdir = cast(Path, model.resdir)
        y_init = np.loadtxt(model_resdir / "current_y.txt")
        rng = np.random.default_rng(645)

        MIN_Y0, MAX_Y0 = -np.inf, np.inf
        MIN_Y1, MAX_Y1 = 0.0, 1.0
        y_init = cast(np.ndarray, y_init)
        perturbation = 0.001
        yinit_0 = np.clip(y_init[:, 0] + rng.normal(0, perturbation, N), MIN_Y0, MAX_Y0)
        yinit_1 = np.clip(y_init[:, 1] + rng.normal(0, perturbation, N), MIN_Y1, MAX_Y1)

        y_init = cast(np.ndarray, np.concatenate((yinit_0, yinit_1)))
        model.v0 = np.loadtxt(model_resdir / "current_v.txt")

    free_y = list(range(N))

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
    params = cast(np.ndarray, model.params)
    w, R, eta = params
    endowment, disutility = theta
    if t is None:
        savings = endowment + np.log(R) / eta
        hours = 1.0
        y = np.array([savings, hours])
        return y
    else:
        s0, l0 = cast(np.ndarray, z)
        # util_l0 = - l0 * l0 / (2 * t)
        # util_l1 = w - disutility - (1.0 - l0)**2 / (2 * t)
        # hours = 1.0 if (util_l1 >= util_l0) else 0.0
        hours = np.clip(l0 + t * (w - disutility), 0.0, 1.0)

        def foc_savings(s):
            return R - np.exp(-eta * (endowment - s)) - (s - s0) / t

        if foc_savings(0.0) <= 0.0:
            savings = 0.0
        elif foc_savings(endowment) >= 0.0:
            savings = endowment
        else:
            savings = spopt.root_scalar(
                foc_savings, x0=endowment / 2.0, bracket=[0, endowment]
            ).root

        return np.array([savings, hours])


def add_results(results: ScreeningResults):
    pass


def adjust_excluded(results: ScreeningResults):
    pass


def add_plots(model: ScreeningModel) -> None:
    df_all_results, model_plotdir = setup_for_plots(model)

    plot_marginal_tax_rate(
        df_all_results,
        title="Marginal tax rate on savings",
        cmap="winter",
        path=str(model_plotdir) + "/marginal_tax_rate.png",
    )
