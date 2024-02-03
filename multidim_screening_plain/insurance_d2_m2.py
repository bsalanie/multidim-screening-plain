from pathlib import Path
from typing import cast

import numpy as np
from bs_python_utils.bsutils import bs_error_abort, mkdir_if_needed

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults
from multidim_screening_plain.insurance_d2_m2_values import (
    S_penalties,
    d0_S_fun,
    d0_val_B,
    d0_val_C,
    d1_S_fun,
    d1_val_C,
    multiply_each_col,
    split_y,
    val_D,
    val_I,
)
from multidim_screening_plain.utils import plots_dir, results_dir


def create_model(model_name: str) -> ScreeningModel:
    """initializes the ScreeningModel object:
    fills in the dimensions, the numbers in each type, the characteristics of the types,
    the model parameters, and the directories.

    Args:
        model_name: the name of the model

    Returns:
        the ScreeningModel object
    """
    # size of grid for types in each dimension
    n0 = n1 = 2
    N = n0 * n1

    # dimension of contracts
    m = 2

    suffix = ""
    case = f"N{N}{suffix}"
    model_id = f"{model_name}_{case}"
    resdir = mkdir_if_needed(results_dir / model_id)
    plotdir = mkdir_if_needed(plots_dir / model_id)

    sigma_min = 0.2
    sigma_max = 0.5
    delta_min = -7.0
    delta_max = -3.0
    # risk-aversions and risk location parameter (unit=1,000 euros)
    sigmas, deltas = np.linspace(sigma_min, sigma_max, num=n0), np.linspace(
        delta_min, delta_max, num=n1
    )
    theta0, theta1 = np.meshgrid(sigmas, deltas)
    theta_mat = np.column_stack(
        (theta0.flatten(), theta1.flatten())
    )  # is a N x 2 matrix

    f = np.ones(N)  # weights of distribution

    # model parameters setting
    s = 4.0  # dispersion of individual losses
    loading = 0.25  # loading factor
    params = np.array([s, loading])

    return ScreeningModel(
        f=f,
        model_id=model_id,
        theta_mat=theta_mat,
        params=params,
        params_names=["s", "loading"],
        m=m,
        resdir=resdir,
        plotdir=plotdir,
    )


def b_fun(y, theta_mat, params):
    """evaluates the value of the coverage

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(q,k)$-matrix: $b_{ij} = b(y_j, \theta_i)$
    """
    sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
    s = params[0]
    y_no_insur = np.array([0.0, 1.0])
    diff_logs = np.log(val_I(y_no_insur, sigmas, deltas, s)) - np.log(
        val_I(y, sigmas, deltas, s)
    )
    return multiply_each_col(
        diff_logs,
        (1.0 / sigmas),
    )


def db_fun(y, theta_mat, params):
    """calculates both derivatives of the coverage

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(2,q,k)$-array:  $db_{lij} = db(y_j, \theta_i)/dy_{jl}$
    """
    y_0, _ = split_y(y)
    sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
    s = params[0]
    denom_inv = 1.0 / multiply_each_col(val_I(y, sigmas, deltas, s), sigmas)
    derivatives_b = np.empty((2, sigmas.size, y_0.size))
    derivatives_b[0, :, :] = (
        -(d0_val_B(y, sigmas, deltas, s) + d0_val_C(y, sigmas, deltas, s)) * denom_inv
    )
    derivatives_b[1, :, :] = -d1_val_C(y, sigmas, deltas, s) * denom_inv
    return derivatives_b


def S_fun(y, theta_mat, params):
    """evaluates the joint surplus

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(q,k)$-matrix: $S_{ij} = S(y_j, \theta_i)$
    """
    deltas = theta_mat[:, 1]
    s, loading = params[0], params[1]
    return (
        b_fun(y, theta_mat, params)
        - (1.0 + loading) * val_D(y, deltas, s)
        - S_penalties(y)
    )


def dS_fun(y, theta_mat, params):
    """calculates both derivatives of the surplus

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(2,q,k)$-array: $dS_{lij} = dS(y_j, \theta_i)/dy_{jl}$
    """
    sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
    s, loading = params[0], params[1]
    dS = np.empty((2, sigmas.size, y.size // 2))
    # print("dS=", dS)
    # print(
    #     "d0_S_fun(y, sigmas, deltas, s, loading)=",
    #     d0_S_fun(y, sigmas, deltas, s, loading),
    # )
    # print(
    #     "d1_S_fun(y, sigmas, deltas, s, loading)=",
    #     d1_S_fun(y, sigmas, deltas, s, loading),
    # )
    dS[0, :, :] = d0_S_fun(y, sigmas, deltas, s, loading)
    dS[1, :, :] = d1_S_fun(y, sigmas, deltas, s, loading)
    # print("dS=", dS)
    return dS


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
        tuple[np.ndarray, list]: initial contracts (an `(N,m)` matrix) and a list of types for whom
            the contracts are to be determined.
    """
    N = model.N
    if start_from_first_best:
        if y_first_best_mat is None:
            bs_error_abort("We start from the first best but y_first_best_mat is None")
        y_init = y_first_best_mat
        set_fixed_y: set[int] = set()
        set_not_insured: set[int] = set()
    else:
        model_resdir = cast(Path, model.resdir)
        y_init = np.loadtxt(model_resdir / "current_y.txt")
        EPS = 0.001
        set_not_insured = {i for i in range(N) if y_init[i, 1] > 1.0 - EPS}
        # not_insured2 = {i for i in range(N) if theta_mat[i, 1] <= -6.0}
        # not_insured3 = {
        #     i
        #     for i in range(N)
        #     if theta_mat[i, 1] <= -5.0 and theta_mat[i, 0] <= 0.425
        # }
        # not_insured4 = {
        #     i
        #     for i in range(N)
        #     if theta_mat[i, 1] <= -4.0 and theta_mat[i, 0] <= 0.35
        # }
        # set_only_deductible = {i for i in range(N) if (start_from_current and y_init[i, 0] < EPS)}
        # set_fixed_y = set_not_insured.union(
        #         set_only_deductible
        #     )
        set_fixed_y = set_not_insured

    set_free_y = set(range(N)).difference(set_fixed_y)
    fixed_y = list(set_fixed_y)
    free_y = list(set_free_y)
    not_insured = list(set_not_insured)
    # only_deductible = list(set_only_deductible)

    print(f"{free_y=}")
    print(f"{fixed_y=}")

    rng = np.random.default_rng(645)

    MIN_Y0, MAX_Y0 = 0.3, np.inf
    MIN_Y1, MAX_Y1 = 0.0, np.inf
    y_init = cast(np.ndarray, y_init)
    yinit_0 = np.clip(y_init[:, 0] + rng.normal(0, 0.00000, N), MIN_Y0, MAX_Y0)
    yinit_1 = np.clip(y_init[:, 1] + rng.normal(0, 0.00000, N), MIN_Y1, MAX_Y1)
    yinit_0[not_insured] = 0.0
    yinit_1[not_insured] = 1.0

    y_init = cast(np.ndarray, np.concatenate((yinit_0, yinit_1)))

    return y_init, free_y


def plot_results(results: ScreeningResults) -> None:
    pass
    # model = results.model
    # model_resdir = cast(Path, model.resdir)
    # model_plotdir = cast(Path, model.plotdir)
    # df_all_results = pd.read_csv(model_resdir / "all_results.csv").rename(
    #     columns={
    #         "FB_y_0": "First-best deductible",
    #         "y_0": "Second-best deductible",
    #         "y_1": "Second-best copay",
    #         "theta_0": "Risk-aversion",
    #         "theta_1": "Risk location",
    #         "FB_surplus": "First-best surplus",
    #         "SB_surplus": "Second-best surplus",
    #         "info_rents": "Informational rents",
    #     }
    # )

    # # first plot the first best
    # y_first_best = df_all_results["First-best deductible"].values.round(3)
    # theta_mat = df_all_results[["Risk-aversion", "Risk location"]].values.round(2)

    # # plot_calibration(df_first_best, path=model_plotdir / "calibration")

    # y_second_best = results.SB_y.round(3)

    # ## put FB and SB together
    # df_first = df_first_best[["Risk-aversion", "Risk location", "Deductible", "Copay"]]
    # df_first["Model"] = "First-best"
    # df_second = pd.DataFrame(
    #     {
    #         "Deductible": y_second_best[:, 0],
    #         "Copay": y_second_best[:, 1],
    #     }
    # )
    # df_second["Model"] = "Second-best"
    # df_first_and_second = pd.concat((df_first, df_second), axis=1)

    # display_variable(
    #     y_first_best[:, 0],
    #     theta_mat,
    #     cmap="viridis",
    #     cmap_label=r"First-best deductible $y_0$",
    #     path=model_plotdir / "first_best_deduc",
    # )

    # plot_contract_models(
    #     df_first_and_second, "Deductible", path=model_plotdir / "deducs_models"
    # )

    # plot_contract_models(
    #     df_first_and_second, "Copay", path=model_plotdir / "copays_models"
    # )

    # plot_contract_riskavs(
    #     df_first_and_second,
    #     "Deductible",
    #     path=model_plotdir / "deducs_riskavs",
    # )

    # plot_contract_riskavs(
    #     df_first_and_second, "Copay", path=model_plotdir / "copays_riskavs"
    # )

    # plot_copays(df_second, path=model_plotdir / "copays")

    # plot_best_contracts(
    #     df_first_and_second,
    #     path=model_plotdir / "optimal_contracts",
    # )

    # plot_y_range(df_first_and_second, path=model_plotdir / "y_range")

    # plot_second_best_contracts(
    #     theta_mat,
    #     y_second_best,
    #     title="Second-best contracts",
    #     cmap="viridis",
    #     path=model_plotdir / "second_best_contracts",
    # )

    # IR_binds = (
    #     np.loadtxt(model_resdir / "second_best_IR_binds.txt").astype(int).tolist()
    # )

    # IC_binds = (
    #     np.loadtxt(model_resdir / "second_best_IC_binds.txt").astype(int).tolist()
    # )

    # plot_constraints(theta_mat, IR_binds, IC_binds, path=model_plotdir / "constraints")

    # U_second, S_second = results.info_rents, results.SB_surplus
    # S_first = df_first_best["Surplus"].values

    # plot_utilities(
    #     theta_mat,
    #     S_first,
    #     S_second,
    #     U_second,
    #     path=model_plotdir / "utilities",
    # )
