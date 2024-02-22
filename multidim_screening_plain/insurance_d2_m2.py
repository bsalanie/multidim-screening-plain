from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from bs_python_utils.bs_opt import minimize_free
from bs_python_utils.bsutils import bs_error_abort

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults
from multidim_screening_plain.insurance_d2_m2_plots import (
    plot_best_contracts,
    plot_calibration,
    plot_contract_models,
    plot_contract_riskavs,
    plot_copays,
    plot_second_best_contracts,
    plot_utilities,
)
from multidim_screening_plain.insurance_d2_m2_values import (
    S_penalties,
    cost_non_insur,
    expected_positive_loss,
    proba_claim,
    val_A,
    val_D,
    val_I,
)
from multidim_screening_plain.utils import (
    contracts_vector,
    display_variable,
    plot_constraints,
    plot_y_range,
)


def precalculate(model: ScreeningModel) -> dict:
    theta_mat = model.theta_mat
    sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
    s = model.params[0]
    values_A = val_A(deltas, s)
    y_no_insurance = np.array([0.0, 1.0])
    I_no_insurance = val_I(y_no_insurance, sigmas, deltas, s)[:, 0]
    return {"values_A": values_A, "I_no_insurance": I_no_insurance}


def b_fun(y: np.ndarray, theta_mat: np.ndarray, params: np.ndarray, gr: bool = False):
    """evaluates the value of the coverage, and maybe its gradient

    Args:
        y:  a $2 k$-vector of $k$ contracts
        theta_mat: a $(q,2)$-vector of characteristics of types
        params: the parameters of the model
        gr: whether we compute the gradient

    Returns:
        a $(q,k)$-matrix: $b_{ij} = b(y_j, \theta_i)$; and a $(2,q,k)$ array if `gr` is `True`
    """
    sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
    s = params[0]
    y_no_insur = np.array([0.0, 1.0])
    value_non_insured = val_I(y_no_insur, sigmas, deltas, s, gr)
    value_insured = val_I(y, sigmas, deltas, s, gr)
    if not gr:
        diff_logs = np.log(value_non_insured) - np.log(value_insured)
        return diff_logs / sigmas.reshape((-1, 1))
    else:
        val_insured, dval_insured = value_insured
        diff_logs = np.log(value_non_insured[0]) - np.log(val_insured)
        denom_inv = 1.0 / (val_insured * sigmas.reshape((-1, 1)))
        grad = np.empty((2, sigmas.size, y.size // 2))
        grad[0, :, :] = -dval_insured[0, :, :] * denom_inv
        grad[1, :, :] = -dval_insured[1, :, :] * denom_inv
        return diff_logs / sigmas.reshape((-1, 1)), grad


def S_fun(y: np.ndarray, theta: np.ndarray, params: np.ndarray, gr: bool = False):
    """evaluates the joint surplus, and maybe its gradient, for 1 contract for 1 type

    Args:
        y:  a $2$-vector of 1 contract $y$
        theta: a $2$-vector of characteristics of 1 type $\theta$
        params: the parameters of the model
        gr: whether we compute the gradient

    Returns:
        S(y, \theta)$,  and an $m$ array of its derivates wrt $y$ if `gr` is `True`
    """
    theta_mat = theta.reshape((1, 2))
    deltas = theta_mat[:, 1]
    s, loading = params[0], params[1]
    b_vals, D_vals, penalties = (
        b_fun(y, theta_mat, params, gr),
        val_D(y, deltas, s, gr),
        S_penalties(y, gr),
    )
    if not gr:
        return b_vals[0, 0] - (1.0 + loading) * D_vals[0, 0] - penalties
    else:
        b_values, b_gradient = b_vals
        D_values, D_gradient = D_vals
        val_penalties, grad_penalties = penalties
        val_S = b_values[0, 0] - (1.0 + loading) * D_values[0, 0] - val_penalties
        grad_S = (
            b_gradient[:, 0, 0]
            - (1.0 + loading) * D_gradient[:, 0, 0]
            - grad_penalties[:, 0]
        )
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
        list(set_fixed_y)
        free_y = list(set_free_y)
        not_insured = list(set_not_insured)
        # only_deductible = list(set_only_deductible)
        rng = np.random.default_rng(645)

        MIN_Y0, MAX_Y0 = 0.3, np.inf
        MIN_Y1, MAX_Y1 = 0.0, np.inf
        y_init = cast(np.ndarray, y_init)
        perturbation = 0.001
        yinit_0 = np.clip(y_init[:, 0] + rng.normal(0, perturbation, N), MIN_Y0, MAX_Y0)
        yinit_1 = np.clip(y_init[:, 1] + rng.normal(0, perturbation, N), MIN_Y1, MAX_Y1)
        yinit_0[not_insured] = 0.0
        yinit_1[not_insured] = 1.0

        y_init = cast(np.ndarray, np.concatenate((yinit_0, yinit_1)))

    return y_init, free_y


def proximal_operator(
    z: np.ndarray, theta: np.ndarray, params: np.ndarray, t: float | None = None
) -> np.ndarray | None:
    """Proximal operator of -t S_i at z;
        minimizes $-S_i(y) + 1/(2 t) \\lVert y-z \rVert^2$

    Args:
        z: an `m`-vector
        theta: type $i$'s characteristics, a $d$-vector
        params: the parameters of the model
        t: the step; if None, we maximize $S_i(y)$

    Returns:
        the minimizing $y$, an $m$-vector
    """

    def prox_obj_and_grad(
        y: np.ndarray, args: list, gr: bool = False
    ) -> float | tuple[float, np.ndarray]:
        S_vals = S_fun(y, theta, params, gr)
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

    y_init = np.array([1.0, 0.2]) if t is None else z

    # check_gradient_scalar_function(prox_obj_and_grad, y_init, args=[])
    # bs_error_abort("done")

    mini = minimize_free(
        prox_obj,
        prox_grad,
        x_init=y_init,
        args=[],
    )

    if mini.success or mini.status == 2:
        y = mini.x
        return cast(np.ndarray, y)
    else:
        print(f"{mini.message}")
        bs_error_abort(f"Minimization did not converge: status {mini.status}")
        return None


def additional_results(
    results: ScreeningResults,
) -> None:
    """Adds more results to the `ScreeningResults` object

    Args:
        results: the results
    """
    model = results.model
    sigmas, deltas = model.theta_mat[:, 0], model.theta_mat[:, 1]
    s = model.params[0]
    results.additional_results_names = [
        "Second-best actuarial premium",
        "Cost of non-insurance",
        "Expected positive loss",
        "Probability of claim",
    ]
    FB_y_vec = contracts_vector(model.FB_y)
    SB_y_vec = contracts_vector(results.SB_y)
    FB_values_coverage = np.diag(b_fun(FB_y_vec, model.theta_mat, model.params))
    FB_actuarial_premia = np.diag(val_D(FB_y_vec, deltas, s))
    SB_values_coverage = np.diag(b_fun(SB_y_vec, model.theta_mat, model.params))
    SB_actuarial_premia = np.diag(val_D(SB_y_vec, deltas, s))
    results.additional_results = [
        FB_actuarial_premia,
        SB_actuarial_premia,
        cost_non_insur(sigmas, deltas, s),
        expected_positive_loss(deltas, s),
        proba_claim(deltas, s),
        FB_values_coverage,
        SB_values_coverage,
    ]
    results.additional_results_names = [
        "Actuarial premium at first-best",
        "Actuarial premium at second-best",
        "Cost of non-insurance",
        "Expected positive loss",
        "Probability of claim",
        "Value of first-best coverage",
        "Value of second-best coverage",
    ]


def plot_results(model: ScreeningModel) -> None:
    model_resdir = cast(Path, model.resdir)
    model_plotdir = str(cast(Path, model.plotdir))
    df_all_results = (
        pd.read_csv(model_resdir / "all_results.csv")
        .rename(
            columns={
                "FB_y_0": "First-best deductible",
                "y_0": "Second-best deductible",
                "y_1": "Second-best copay",
                "theta_0": "Risk-aversion",
                "theta_1": "Risk location",
                "FB_surplus": "First-best surplus",
                "SB_surplus": "Second-best surplus",
                "info_rents": "Informational rent",
            }
        )
        .round(3)
    )
    df_all_results.loc[:, "Second-best copay"] = np.clip(
        df_all_results["Second-best copay"].values, 0.0, 1.0
    )

    # first plot the first best
    plot_calibration(df_all_results, path=model_plotdir + "/calibration")

    display_variable(
        df_all_results,
        variable="First-best deductible",
        cmap="viridis",
        path=model_plotdir + "/first_best_deduc",
    )

    df_contracts = df_all_results[
        [
            "Risk-aversion",
            "Risk location",
            "First-best deductible",
            "Second-best deductible",
            "Second-best copay",
        ]
    ]

    df_first_and_second = pd.DataFrame(
        {
            "Model": np.concatenate(
                (np.full(model.N, "First-best"), np.full(model.N, "Second-best"))
            ),
            "Risk-aversion": np.tile(df_contracts["Risk-aversion"].values, 2),
            "Risk location": np.tile(df_contracts["Risk location"].values, 2),
            "Deductible": np.concatenate(
                (
                    df_contracts["First-best deductible"],
                    df_contracts["Second-best deductible"],
                )
            ),
            "Copay": np.concatenate(
                (np.zeros(model.N), df_contracts["Second-best copay"].values)
            ),
        }
    )

    plot_contract_models(
        df_first_and_second, "Deductible", path=model_plotdir + "/deducs_models"
    )

    plot_contract_models(
        df_first_and_second, "Copay", path=model_plotdir + "/copays_models"
    )

    plot_contract_riskavs(
        df_first_and_second,
        "Deductible",
        path=model_plotdir + "/deducs_riskavs",
    )

    plot_contract_riskavs(
        df_first_and_second, "Copay", path=model_plotdir + "/copays_riskavs"
    )

    df_second = df_first_and_second.query('Model == "Second-best"')
    plot_copays(df_second, path=model_plotdir + "/copays")

    plot_best_contracts(
        df_first_and_second,
        path=model_plotdir + "/optimal_contracts",
    )

    plot_y_range(df_first_and_second, path=model_plotdir + "/y_range")

    plot_second_best_contracts(
        df_second,
        title="Second-best contracts",
        cmap="viridis",
        path=model_plotdir + "/second_best_contracts",
    )

    IR_binds = np.loadtxt(model_resdir / "IR_binds.txt").astype(int).tolist()

    IC_binds = np.loadtxt(model_resdir / "IC_binds.txt").astype(int).tolist()

    plot_constraints(
        df_all_results, IR_binds, IC_binds, path=model_plotdir + "/constraints"
    )

    plot_utilities(
        df_all_results,
        path=model_plotdir + "/utilities",
    )
