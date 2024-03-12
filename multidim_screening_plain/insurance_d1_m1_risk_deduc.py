from pathlib import Path
from typing import cast

import numpy as np
from bs_python_utils.bs_opt import minimize_free
from bs_python_utils.bsutils import bs_error_abort

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults
from multidim_screening_plain.insurance_d1_m1_risk_deduc_plots import (
    plot_calibration,
)
from multidim_screening_plain.insurance_d1_m1_risk_deduc_values import (
    S_penalties,
    cost_non_insur,
    expected_positive_loss,
    proba_claim,
    val_D,
    val_I,
)
from multidim_screening_plain.plot_utils import setup_for_plots
from multidim_screening_plain.utils import check_args, contracts_vector


def b_fun(
    model: ScreeningModel,
    y: np.ndarray,
    theta: np.ndarray | None = None,
    gr: bool = False,
):
    """evaluates the value of the coverage, and maybe its gradient

    Args:
        model: the ScreeningModel
        y:  a `k`-vector of deductible values
        theta: a 1-vector with the risk of one type, if provided
        gr: whether we compute the gradient

    Returns:
        if `theta` is provided then `k` should be 1, and we return b(y,theta)
            for this contract for this type
        otherwise we return an (N,k)-matrix with `b_i(y_j)` for all `N` types `i` and
            all `k` contracts `y_j` in `y`
        and if `gr` is `True` we provide the gradient wrt `y`
    """
    check_args("b_fun", y, 1, 1, theta)
    sigma = cast(np.ndarray, model.params)[0]
    y_no_insur = np.array([20.0])
    if theta is not None:
        return b_fun_1(model, y, y_no_insur, theta, sigma, gr=gr)
    else:
        return b_fun_all(model, y, y_no_insur, sigma, gr=gr)


def b_fun_1(
    model: ScreeningModel,
    y: np.ndarray,
    y_no_insur: np.ndarray,
    theta: np.ndarray,
    sigma: float,
    gr: bool = False,
):
    """`b_fun` for contract `y` for type `theta"""
    value_non_insured = val_I(model, y_no_insur, theta=theta)
    value_insured = val_I(model, y, theta=theta, gr=gr)
    if not gr:
        diff_logs = np.log(value_non_insured) - np.log(value_insured)
        return diff_logs / sigma
    else:
        val_insured, dval_insured = value_insured
        diff_logs = np.log(value_non_insured) - np.log(val_insured)
        grad = -dval_insured / (val_insured * sigma)
        return diff_logs / sigma, grad


def b_fun_all(
    model: ScreeningModel,
    y: np.ndarray,
    y_no_insur: np.ndarray,
    sigma: float,
    gr: bool = False,
):
    """`b_fun` for all contracts in `y` and for all types"""
    value_non_insured = val_I(model, y_no_insur)
    value_insured = val_I(model, y, gr=gr)
    if not gr:
        diff_logs = np.log(value_non_insured) - np.log(value_insured)
        return diff_logs / sigma
    else:
        val_insured, dval_insured = value_insured
        diff_logs = np.log(value_non_insured) - np.log(val_insured)
        denom_inv = 1.0 / (val_insured * sigma)
        grad = np.empty((1, model.N, y.size))
        grad[0, :, :] = -dval_insured[0, :, :] * denom_inv
        return diff_logs / sigma, grad


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
    delta = theta[0]
    params = cast(np.ndarray, model.params)
    s, loading = params[1], params[2]
    b_vals, D_vals, penalties = (
        b_fun(model, y, theta=theta, gr=gr),
        val_D(y, delta, s, gr=gr),
        S_penalties(y, gr=gr),
    )
    if not gr:
        return b_vals - (1.0 + loading) * D_vals - penalties
    else:
        b_values, b_gradient = b_vals
        D_values, D_gradient = D_vals
        val_penalties, grad_penalties = penalties
        val_S = b_values - (1.0 + loading) * D_values - val_penalties
        grad_S = b_gradient - (1.0 + loading) * D_gradient - grad_penalties
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
    else:  # not insured if deductible >= 10
        model_resdir = cast(Path, model.resdir)
        y_init = np.loadtxt(model_resdir / "current_y.txt")
        EPS = 0.001
        set_not_insured = {i for i in range(N) if y_init[i] > 10.0 - EPS}
        set_fixed_y = set_not_insured
        print(f"{set_fixed_y=}")

        set_free_y = set(range(N)).difference(set_fixed_y)
        fixed_y = list(set_fixed_y)
        free_y = list(set_free_y)

        MIN_Y0, MAX_Y0 = 0.0, np.inf
        y_init = cast(np.ndarray, y_init)
        perturbation = 0.001
        rng = np.random.default_rng(645)
        y_init = np.clip(y_init + rng.normal(0, perturbation, N), MIN_Y0, MAX_Y0)
        y_init[fixed_y] = 10.0

        y_init = cast(np.ndarray, y_init)

    return y_init, free_y


def proximal_operator(
    model: ScreeningModel,
    theta: np.ndarray,
    z: np.ndarray | None = None,
    t: float | None = None,
) -> np.ndarray | None:
    """Proximal operator of `-t S_i` at `z`;
        minimizes `-S_i(y) + 1/(2 t)  ||y-z||^2` if `z` and `t` are given
        otherwise, maximizes `S_i(y)`

    Args:
        model: the ScreeningModel
        z: a `1`-vector, if any
        theta: type `i`'s characteristics, a `1`-vector
        t: the step, if any

    Returns:
        the optimized `y`, a 1-vector
    """
    # if t is not None:
    # print(f"{theta=}, {z=}, {t=}")

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

    y_init = np.array([1.0]) if t is None else z

    #     check_gradient_scalar_function(prox_obj_and_grad, y_init, args=[])
    #     bs_error_abort("done")

    mini = minimize_free(
        prox_obj,
        prox_grad,
        x_init=y_init,
        args=[],
        bounds=[(0.0, 15.0)],
    )

    if mini.success or mini.status == 2:
        y = mini.x
        return cast(np.ndarray, y)
    else:
        print(f"{mini.message}")
        bs_error_abort(f"Minimization did not converge: status {mini.status}")
        return None


def adjust_excluded(results: ScreeningResults) -> None:
    """Adjusts the results for the excluded types, or just `pass`

    Args:
        results: the results
    """
    deduc = results.SB_y[:, 0]
    EPS = 0.001
    MAX_DEDUC = 5.0
    excluded_types = np.where(deduc > MAX_DEDUC - EPS, True, False).tolist()
    results.SB_surplus[excluded_types] = results.info_rents[excluded_types] = 0.0
    n_excluded = np.sum(excluded_types)
    results.SB_y[excluded_types, 0] = np.full(n_excluded, MAX_DEDUC)
    results.excluded_types = excluded_types


def add_results(
    results: ScreeningResults,
) -> None:
    """Adds more results to the `ScreeningResults` object

    Args:
        results: the results
    """
    model = results.model
    N = model.N
    theta_mat = model.theta_mat
    s = cast(np.ndarray, model.params)[1]

    FB_y = model.FB_y
    SB_y = results.SB_y
    FB_values_coverage = np.zeros(N)
    FB_actuarial_premia = np.zeros(N)
    SB_values_coverage = np.zeros(N)
    SB_actuarial_premia = np.zeros(N)

    for i in range(N):
        FB_i = FB_y[i, :]
        theta_i = theta_mat[i, :]
        delta_i = theta_i[0]
        FB_values_coverage[i] = b_fun(model, FB_i, theta=theta_i)
        FB_actuarial_premia[i] = val_D(FB_i, delta_i, s)
        SB_i = SB_y[i]
        SB_values_coverage[i] = b_fun(model, SB_i, theta=theta_i)
        SB_actuarial_premia[i] = val_D(SB_i, delta_i, s)

    deltas = model.theta_mat[:, 0]
    results.additional_results = [
        FB_actuarial_premia,
        SB_actuarial_premia,
        cost_non_insur(model),
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


def additional_plots(model: ScreeningModel) -> None:
    df_all_results, model_plotdir = setup_for_plots(model)

    # plot the first best
    plot_calibration(df_all_results, path=model_plotdir + "/calibration")
