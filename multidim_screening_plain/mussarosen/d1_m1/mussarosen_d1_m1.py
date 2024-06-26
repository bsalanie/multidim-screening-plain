from pathlib import Path
from typing import cast

import numpy as np
from bs_python_utils.bsutils import bs_error_abort

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults
from multidim_screening_plain.utils import (
    check_args,
    contracts_vector,
)


def b_fun(
    model: ScreeningModel,
    y: np.ndarray,
    theta: np.ndarray | None = None,
    gr: bool = False,
):
    """evaluates the value of the coverage, and maybe its gradient

    Args:
        model: the ScreeningModel
        y:  a `k`-vector of $k$ contracts
        theta: a 1-vector of characteristic of one type, if provided
        gr: whether we compute the gradient

    Returns:
        if `theta` is provided then `k` should be 1, and we return b(y,theta)
            for this contract for this type
        otherwise we return an (N,k)-matrix with `b_i(y_j)` for all `N` types `i` and
            all `k` contracts `y_j` in `y`
        and if `gr` is `True` we provide the gradient wrt `y`
    """
    check_args("b_fun", y, 1, 1, theta)
    if theta is not None:
        b_val = np.dot(theta, y)
        if not gr:
            return b_val
        else:
            return b_val, theta
    else:
        theta_mat = model.theta_mat
        y_0 = y
        b_vals = np.outer(theta_mat[:, 0], y_0)
        if not gr:
            return b_vals
        else:
            k = y.size
            grad = np.zeros((1, model.N, k))
            grad[0, :, :] = np.tile(theta_mat[:, 0], (k, 1)).T
            return b_vals, grad


def S_fun(model: ScreeningModel, y: np.ndarray, theta: np.ndarray, gr: bool = False):
    """evaluates the joint surplus, and maybe its gradient, for 1 contract for 1 type

    Args:
        model: the ScreeningModel
        y:  a 1-vector of 1 contract `y`
        theta: a 1-vector of characteristics of one type
        gr: whether we compute the gradient

    Returns:
        the value of `S(y,theta)` for this contract and this type,
            and its gradient wrt `y` if `gr` is `True`
    """
    check_args("S_fun", y, 1, 1, theta)
    b_vals = b_fun(model, y, theta=theta, gr=gr)
    cost = np.dot(y, y) / 2.0
    if not gr:
        return b_vals - cost
    else:
        b_values, b_gradient = b_vals
        val_S = b_values - cost
        grad_S = b_gradient - y
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
    free_y = list(range(N))
    if start_from_first_best:
        if y_first_best_mat is None:
            bs_error_abort("We start from the first best but y_first_best_mat is None")
        y_init = contracts_vector(cast(np.ndarray, y_first_best_mat))
    else:
        model_resdir = cast(Path, model.resdir)
        y_init = np.loadtxt(model_resdir / "current_y.txt")
        rng = np.random.default_rng(645)

        MIN_Y0, MAX_Y0 = 0.0, np.inf
        y_init = cast(np.ndarray, y_init)
        perturbation = 0.001
        y_init = np.clip(y_init + rng.normal(0, perturbation, N), MIN_Y0, MAX_Y0)
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
        z: a `1`-vector for a type, if any
        t: the step; if None, we maximize `S_i(y)`

    Returns:
        the minimizing `y`, a 1-vector
    """
    if isinstance(t, float) and isinstance(z, np.ndarray):
        return cast(np.ndarray, (t * theta + z) / (t + 1.0))
    else:
        return theta


def adjust_excluded(results: ScreeningResults) -> None:
    """Adjusts the results for the excluded types, or just `pass`

    Args:
        results: the results
    """
    FBs, SBs = results.FB_surplus, results.SB_surplus
    EPS = 0.001
    excluded_types = np.where(SBs / FBs < EPS, True, False).tolist()
    results.SB_surplus[excluded_types] = results.info_rents[excluded_types] = 0.0
    n_excluded = np.sum(excluded_types)
    results.SB_y[excluded_types] = np.zeros((n_excluded, 1))
    results.excluded_types = excluded_types


def add_results(
    results: ScreeningResults,
) -> None:
    """Adds more results to the `ScreeningResults` object if needed;
    otherwise just `pass`

    Args:
        results: the results
    """
    pass


def add_plots(model: ScreeningModel) -> None:
    """Adds more plots if needed; otherwise just `pass`

    Args:
        model: the ScreeningModel
    """
    pass
