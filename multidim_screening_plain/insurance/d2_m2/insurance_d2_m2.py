from pathlib import Path
from typing import cast

import numpy as np
from bs_python_utils.bs_opt import minimize_free
from bs_python_utils.bsutils import bs_error_abort

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults
from multidim_screening_plain.insurance.d2_m2.insurance_d2_m2_plots import (
    plot_calibration,
)
from multidim_screening_plain.insurance.d2_m2.insurance_d2_m2_values import (
    cost_non_insur,
    expected_positive_loss,
    proba_claim,
    val_D,
    val_I,
    val_I_no_insurance,
)
from multidim_screening_plain.plot_utils import setup_for_plots
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
        y:  a `2 k`-vector of $k$ contracts
        theta: a 2-vector of characteristics of one type, if provided
        gr: whether we compute the gradient

    Returns:
        if `theta` is provided then `k` should be 1, and we return b(y,theta)
            for this contract for this type
        otherwise we return an (N,k)-matrix with `b_i(y_j)` for all `N` types `i` and
            all `k` contracts `y_j` in `y`
        and if `gr` is `True` we provide the gradient wrt `y`, a `2`-vector or a `(2,N,k)` array
    """
    check_args("b_fun", y, 2, 2, theta)
    if theta is not None:
        value_non_insured = val_I_no_insurance(model, theta=theta)
        value_insured = val_I(model, y, theta=theta, gr=gr)
        sigma = theta[0]
        if not gr:
            diff_logs = -np.log(value_insured) + np.log(value_non_insured)
            return diff_logs / sigma
        else:
            val_insured, dval_insured = value_insured
            diff_logs = -np.log(val_insured) + np.log(value_non_insured)
            grad = -dval_insured / (val_insured * sigma)
            return diff_logs / sigma, grad
    else:
        theta_mat = model.theta_mat
        sigmas = theta_mat[:, 0]
        y_no_insur = np.array([0.0, 1.0])
        value_non_insured = val_I(model, y_no_insur)
        value_insured = val_I(model, y, gr=gr)
        if not gr:
            diff_logs = -np.log(value_insured) + np.log(value_non_insured).reshape(
                (-1, 1)
            )
            return diff_logs / sigmas.reshape((-1, 1))
        else:
            val_insured, dval_insured = value_insured
            diff_logs = -np.log(val_insured) + np.log(value_non_insured).reshape(
                (-1, 1)
            )
            denom_inv = 1.0 / (val_insured * sigmas.reshape((-1, 1)))
            grad = np.empty((2, sigmas.size, y.size // 2))
            grad[0, :, :] = -dval_insured[0, :, :] * denom_inv
            grad[1, :, :] = -dval_insured[1, :, :] * denom_inv
            return diff_logs / sigmas.reshape((-1, 1)), grad


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
    delta = theta[1]
    params = cast(np.ndarray, model.params)
    s, loading, k = params
    b_vals, D_vals = (
        b_fun(model, y, theta=theta, gr=gr),
        val_D(y, delta, s, k, gr=gr),
    )
    if not gr:
        return b_vals - (1.0 + loading) * D_vals
    else:
        b_values, b_gradient = b_vals
        D_values, D_gradient = D_vals
        val_S = b_values - (1.0 + loading) * D_values
        grad_S = b_gradient - (1.0 + loading) * D_gradient
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
    else:  # we read current values and we filtered out the not insured (copay=1)
        model_resdir = cast(Path, model.resdir)
        y_init = np.loadtxt(model_resdir / "current_y.txt")
        EPS = 0.001
        set_not_insured = {i for i in range(N) if y_init[i, 1] > 1.0 - EPS}
        set_fixed_y = set_not_insured

        set_free_y = set(range(N)).difference(set_fixed_y)
        list(set_fixed_y)
        free_y = list(set_free_y)
        not_insured = list(set_not_insured)
        rng = np.random.default_rng(645)

        MIN_Y0, MAX_Y0 = model.y_minmax[0, :]
        MIN_Y1, MAX_Y1 = model.y_minmax[1, :]
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

    y_init = np.array([1.0, 0.2]) if t is None else z

    # check_gradient_scalar_function(prox_obj_and_grad, y_init, args=[])
    # bs_error_abort("done")

    y_mins, y_maxs = model.y_minmax[:, 0], model.y_minmax[:, 1]

    mini = minimize_free(
        prox_obj,
        prox_grad,
        x_init=y_init,
        args=[],
        bounds=[(y_mins[0], y_maxs[0]), (y_mins[1], y_maxs[1])],
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
    copay = results.SB_y[:, 1]
    EPS = 0.001
    excluded_types = np.where(copay > 1.0 - EPS, True, False).tolist()
    results.SB_surplus[excluded_types] = results.info_rents[excluded_types] = 0.0
    MAX_DEDUC = results.model.y_minmax[0, 1]
    n_excluded = np.sum(excluded_types)
    results.SB_y[excluded_types, 0] = np.full(n_excluded, MAX_DEDUC)
    results.SB_y[excluded_types, 1] = np.ones(n_excluded)
    results.excluded_types = excluded_types


def add_results(
    results: ScreeningResults,
) -> None:
    """Adds more results to the `ScreeningResults` object if needed;
    otherwise just `pass`

    Args:
        results: the results
    """
    model = results.model
    N = model.N
    theta_mat = model.theta_mat
    params = cast(np.ndarray, model.params)
    s, _, k = params

    FB_y = model.FB_y
    SB_y = results.SB_y
    FB_values_coverage = np.zeros(N)
    FB_actuarial_premia = np.zeros(N)
    SB_values_coverage = np.zeros(N)
    SB_actuarial_premia = np.zeros(N)

    for i in range(N):
        FB_i = FB_y[i, :]
        theta_i = theta_mat[i, :]
        delta_i = theta_i[1]
        FB_values_coverage[i] = b_fun(model, FB_i, theta=theta_i)
        FB_actuarial_premia[i] = val_D(FB_i, delta_i, s, k)
        SB_i = SB_y[i]
        SB_values_coverage[i] = b_fun(model, SB_i, theta=theta_i)
        SB_actuarial_premia[i] = val_D(SB_i, delta_i, s, k)

    # uninsured types
    excluded_types = results.excluded_types
    SB_values_coverage[excluded_types] = SB_actuarial_premia[excluded_types] = 0.0

    deltas = model.theta_mat[:, 1]
    results.additional_results = [
        FB_actuarial_premia,
        SB_actuarial_premia,
        cost_non_insur(model),
        expected_positive_loss(deltas, s),
        proba_claim(deltas, s, k),
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


def add_plots(model: ScreeningModel) -> None:
    """Adds more plots if needed; otherwise just `pass`

    Args:
        model: the ScreeningModel
    """
    df_all_results, model_plotdir = setup_for_plots(model)
    plot_calibration(df_all_results, path=model_plotdir + "/calibration")
