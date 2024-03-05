from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from bs_python_utils.bsutils import bs_error_abort

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults
from multidim_screening_plain.mussarosen_d2_m2_plots import (
    plot_best_contracts,
    plot_contract_models,
    plot_contract_riskavs,
    plot_second_best_contracts,
    plot_utilities,
)
from multidim_screening_plain.utils import (
    contracts_vector,
    display_variable,
    plot_constraints,
)


def split_y(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split y into two halves of equal length (deductibles and copays)"""
    N = y.size // 2
    y_0, y_1 = y[:N], y[N:]
    return y_0, y_1


def check_args(function_name: str, y: np.ndarray, theta: np.ndarray | None) -> None:
    """check the arguments passed"""
    if theta is not None:
        if theta.shape != (2,):
            bs_error_abort(
                f"{function_name}: If theta is given it should be a 2-vector, not shape"
                f" {theta.shape}"
            )
        if y.shape != (2,):
            bs_error_abort(
                f"{function_name}: If theta is given, y should be a 2-vector, not shape"
                f" {y.shape}"
            )
    else:
        if y.ndim != 1:
            bs_error_abort(
                f"{function_name}: y should be a vector, not {y.ndim}-dimensional"
            )
        if y.size % 2 != 0:
            bs_error_abort(
                f"{function_name}: y should have an even number of elements, not"
                f" {y.size}"
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
    check_args("b_fun", y, theta)
    if theta is not None:
        b_val = np.dot(theta, y)
        if not gr:
            return b_val
        else:
            return b_val, theta
    else:
        theta_mat = model.theta_mat
        y_0, y_1 = split_y(y)
        b_vals = np.outer(theta_mat[:, 0], y_0) + np.outer(theta_mat[:, 1], y_1)
        if not gr:
            return b_vals
        else:
            k = y.size // 2
            grad = np.zeros((2, model.N, k))
            grad[0, :, :] = np.tile(theta_mat[:, 0], (k, 1)).T
            grad[1, :, :] = np.tile(theta_mat[:, 1], (k, 1)).T
            return b_vals, grad


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
    check_args("S_fun", y, theta)
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
        MIN_Y1, MAX_Y1 = 0.0, np.inf
        y_init = cast(np.ndarray, y_init)
        perturbation = 0.001
        yinit_0 = np.clip(y_init[:, 0] + rng.normal(0, perturbation, N), MIN_Y0, MAX_Y0)
        yinit_1 = np.clip(y_init[:, 1] + rng.normal(0, perturbation, N), MIN_Y1, MAX_Y1)

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
    if isinstance(t, float) and isinstance(z, np.ndarray):
        return cast(np.ndarray, (t * theta + z) / (t + 1.0))
    else:
        return theta


def add_results(
    results: ScreeningResults,
) -> None:
    """Adds more results to the `ScreeningResults` object

    Args:
        results: the results
    """
    return None


def plot_results(model: ScreeningModel) -> None:
    model_resdir = cast(Path, model.resdir)
    model_plotdir = str(cast(Path, model.plotdir))
    df_all_results = (
        pd.read_csv(model_resdir / "all_results.csv")
        .rename(
            columns={
                "FB_y_0": "First-best y_0",
                "FB_y_1": "First-best y_1",
                "y_0": "Second-best y_0",
                "y_1": "Second-best y_1",
                "FB_surplus": "First-best surplus",
                "SB_surplus": "Second-best surplus",
                "info_rents": "Informational rent",
            }
        )
        .round(3)
    )

    # first plot the first best
    display_variable(
        df_all_results,
        variable="First-best y_0",
        theta_names=model.type_names,
        cmap="viridis",
        path=model_plotdir + "/first_best_y_0",
    )

    df_contracts = df_all_results[
        [
            "theta_0",
            "theta_1",
            "First-best y_0",
            "First-best y_1",
            "Second-best y_0",
            "Second-best y_1",
        ]
    ]

    df_first_and_second = pd.DataFrame(
        {
            "Model": np.concatenate(
                (np.full(model.N, "First-best"), np.full(model.N, "Second-best"))
            ),
            "theta_0": np.tile(df_contracts["theta_0"].values, 2),
            "theta_1": np.tile(df_contracts["theta_1"].values, 2),
            "y_0": np.concatenate(
                (
                    df_contracts["First-best y_0"],
                    df_contracts["Second-best y_0"],
                )
            ),
            "y_1": np.concatenate(
                (
                    df_contracts["First-best y_1"],
                    df_contracts["Second-best y_1"],
                )
            ),
        }
    )

    plot_contract_models(df_first_and_second, "y_0", path=model_plotdir + "/y_0_models")

    plot_contract_models(df_first_and_second, "y_1", path=model_plotdir + "/y_1_models")

    plot_contract_riskavs(
        df_first_and_second,
        "y_0",
        path=model_plotdir + "/y_0_riskavs",
    )

    plot_contract_riskavs(
        df_first_and_second, "y_1", path=model_plotdir + "/y_1_riskavs"
    )

    df_second = df_first_and_second.query('Model == "Second-best"')

    plot_best_contracts(
        df_first_and_second,
        path=model_plotdir + "/optimal_contracts",
    )

    # plot_y_range(df_first_and_second, contract_names=model.contract_varnames,
    #              path=model_plotdir + "/y_range")

    plot_second_best_contracts(
        df_second,
        title="Second-best contracts",
        cmap="viridis",
        path=model_plotdir + "/second_best_contracts",
    )

    IR_binds = np.loadtxt(model_resdir / "IR_binds.txt").astype(int).tolist()

    IC_binds = np.loadtxt(model_resdir / "IC_binds.txt").astype(int).tolist()

    plot_constraints(
        df_all_results,
        model.type_names,
        IR_binds,
        IC_binds,
        path=model_plotdir + "/constraints",
    )

    plot_utilities(
        df_all_results,
        path=model_plotdir + "/utilities",
    )
