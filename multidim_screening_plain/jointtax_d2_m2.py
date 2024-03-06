# from pathlib import Path
# from typing import cast

# import numpy as np
# import pandas as pd
# from bs_python_utils.bs_opt import minimize_free
# from bs_python_utils.bsutils import bs_error_abort

# from multidim_screening_plain.classes import ScreeningModel, ScreeningResults
# from multidim_screening_plain.jointtax_d2_m2_plots import (
#     plot_best_contracts,
#     plot_calibration,
#     plot_contract_models,
#     plot_contract_riskavs,
#     plot_copays,
#     plot_second_best_contracts,
#     plot_utilities,
# )
# from multidim_screening_plain.utils import (
#     contracts_vector,
#     display_variable,
#     plot_constraints,
#     plot_y_range,
# )

# ######
# ###### this was a failed attempt to save execution time
# ######
# # def precalculate(model: ScreeningModel) -> None:
# #     theta_mat = model.theta_mat
# #     sigmas, deltas = theta_mat[:, 0], theta_mat[:, 1]
# #     s = cast(np.ndarray, model.params)[0]
# #     values_A = val_A(deltas, s)
# #     sigmas_s = s * sigmas
# #     argu1 = deltas / s + sigmas_s
# #     cdf1 = bs_norm_cdf(argu1)
# #     val_expB = np.exp(sigmas * (s * sigmas_s / 2.0 + deltas))
# #     model.precalculated_values = {
# #         "values_A": cast(np.ndarray, values_A),
# #         "argu1": cast(np.ndarray, argu1),
# #         "cdf1": cast(np.ndarray, cdf1),
# #         "val_expB": cast(np.ndarray, val_expB),
# #     }
# #     y_no_insurance = np.array([0.0, 1.0])
# #     I_no_insurance = val_I(model, y_no_insurance)  # [:, 0]
# #     model.precalculated_values["I_no_insurance"] = cast(np.ndarray, I_no_insurance)


# def b_fun(
#     model: ScreeningModel,
#     y: np.ndarray,
#     theta: np.ndarray | None = None,
#     gr: bool = False,
# ):
#     """evaluates the value of the coverage, and maybe its gradient

#     Args:
#         model: the ScreeningModel
#         y:  a `2 k`-vector of $k$ contracts
#         theta: a 2-vector of characteristics of one type, if provided
#         gr: whether we compute the gradient

#     Returns:
#         if `theta` is provided then `k` should be 1, and we return b(y,theta)
#             for this contract for this type
#         otherwise we return an (N,k)-matrix with `b_i(y_j)` for all `N` types `i` and
#             all `k` contracts `y_j` in `y`
#         and if `gr` is `True` we provide the gradient wrt `y`
#     """
#     check_args("b_fun", y, 2, 2, theta)
#     if theta is not None:
#         y_no_insur = np.array([0.0, 1.0])
#         value_non_insured = val_I(model, y_no_insur, theta=theta)
#         value_insured = val_I(model, y, theta=theta, gr=gr)
#         sigma = theta[0]
#         if not gr:
#             diff_logs = np.log(value_non_insured) - np.log(value_insured)
#             return diff_logs / sigma
#         else:
#             val_insured, dval_insured = value_insured
#             diff_logs = np.log(value_non_insured) - np.log(val_insured)
#             grad = -dval_insured / (val_insured * sigma)
#             return diff_logs / sigma, grad
#     else:
#         theta_mat = model.theta_mat
#         sigmas = theta_mat[:, 0]
#         # precalculated_values = model.precalculated_values
#         # value_non_insured = precalculated_values["I_no_insurance"]
#         # print(f"precalc: {value_non_insured}")
#         y_no_insur = np.array([0.0, 1.0])
#         value_non_insured = val_I(model, y_no_insur)
#         # # print(f"calc: {value_non_insured}")
#         # assert np.allclose(value_non_insured, value_non_insured_calc)
#         value_insured = val_I(model, y, gr=gr)
#         if not gr:
#             diff_logs = np.log(value_non_insured) - np.log(value_insured)
#             return diff_logs / sigmas.reshape((-1, 1))
#         else:
#             val_insured, dval_insured = value_insured
#             diff_logs = np.log(value_non_insured) - np.log(val_insured)
#             denom_inv = 1.0 / (val_insured * sigmas.reshape((-1, 1)))
#             grad = np.empty((2, sigmas.size, y.size // 2))
#             grad[0, :, :] = -dval_insured[0, :, :] * denom_inv
#             grad[1, :, :] = -dval_insured[1, :, :] * denom_inv
#             return diff_logs / sigmas.reshape((-1, 1)), grad


# def S_fun(model: ScreeningModel, y: np.ndarray, theta: np.ndarray, gr: bool = False):
#     """evaluates the joint surplus, and maybe its gradient, for 1 contract for 1 type

#     Args:
#         model: the ScreeningModel
#         y:  a 2-vector of 1 contract `y`
#         theta: a 2-vector of characteristics of one type
#         gr: whether we compute the gradient

#     Returns:
#         the value of `S(y,theta)` for this contract and this type,
#             and its gradient wrt `y` if `gr` is `True`
#     """
#     check_args("S_fun", y, theta)
#     delta = theta[1]
#     params = cast(np.ndarray, model.params)
#     s, loading = params[0], params[1]
#     b_vals, D_vals, penalties = (
#         b_fun(model, y, theta=theta, gr=gr),
#         val_D(y, delta, s, gr=gr),
#         S_penalties(y, gr=gr),
#     )
#     if not gr:
#         return b_vals - (1.0 + loading) * D_vals - penalties
#     else:
#         b_values, b_gradient = b_vals
#         D_values, D_gradient = D_vals
#         val_penalties, grad_penalties = penalties
#         val_S = b_values - (1.0 + loading) * D_values - val_penalties
#         grad_S = b_gradient - (1.0 + loading) * D_gradient - grad_penalties
#         return val_S, grad_S


# def create_initial_contracts(
#     model: ScreeningModel,
#     start_from_first_best: bool,
#     y_first_best_mat: np.ndarray | None = None,
# ) -> tuple[np.ndarray, list]:
#     """Initializes the contracts for the second best problem (MODEL-DEPENDENT)

#     Args:
#         model: the ScreeningModel object
#         start_from_first_best: whether to start from the first best
#         y_first_best_mat: the `(N, m)` matrix of first best contracts. Defaults to None.

#     Returns:
#         tuple[np.ndarray, list]: initial contracts (an `(m *N)` vector) and a list of types for whom
#             the contracts are to be determined.
#     """
#     N = model.N
#     if start_from_first_best:
#         if y_first_best_mat is None:
#             bs_error_abort("We start from the first best but y_first_best_mat is None")
#         y_init = contracts_vector(cast(np.ndarray, y_first_best_mat))
#         set_fixed_y: set[int] = set()
#         set_not_insured: set[int] = set()
#         free_y = list(range(N))
#     else:
#         model_resdir = cast(Path, model.resdir)
#         y_init = np.loadtxt(model_resdir / "current_y.txt")
#         EPS = 0.001
#         set_not_insured = {i for i in range(N) if y_init[i, 1] > 1.0 - EPS}
#         set_fixed_y = set_not_insured

#         set_free_y = set(range(N)).difference(set_fixed_y)
#         list(set_fixed_y)
#         free_y = list(set_free_y)
#         not_insured = list(set_not_insured)
#         # only_deductible = list(set_only_deductible)
#         rng = np.random.default_rng(645)

#         MIN_Y0, MAX_Y0 = 0.3, np.inf
#         MIN_Y1, MAX_Y1 = 0.0, np.inf
#         y_init = cast(np.ndarray, y_init)
#         perturbation = 0.001
#         yinit_0 = np.clip(y_init[:, 0] + rng.normal(0, perturbation, N), MIN_Y0, MAX_Y0)
#         yinit_1 = np.clip(y_init[:, 1] + rng.normal(0, perturbation, N), MIN_Y1, MAX_Y1)
#         yinit_0[not_insured] = 0.0
#         yinit_1[not_insured] = 1.0

#         y_init = cast(np.ndarray, np.concatenate((yinit_0, yinit_1)))
#         model.v0 = np.loadtxt(model_resdir / "current_v.txt")

#     return y_init, free_y


# def proximal_operator(
#     model: ScreeningModel,
#     theta: np.ndarray,
#     z: np.ndarray | None = None,
#     t: float | None = None,
# ) -> np.ndarray | None:
#     """Proximal operator of `-t S_i` at `z`;
#         minimizes `-S_i(y) + 1/(2 t)  ||y-z||^2`

#     Args:
#         model: the ScreeningModel
#         theta: type `i`'s characteristics, a `d`-vector
#         z: a `2`-vector for a type, if any
#         t: the step; if None, we maximize `S_i(y)`

#     Returns:
#         the minimizing `y`, a 2-vector
#     """

#     def prox_obj_and_grad(
#         y: np.ndarray, args: list, gr: bool = False
#     ) -> float | tuple[float, np.ndarray]:
#         S_vals = S_fun(model, y, theta=theta, gr=gr)
#         if not gr:
#             obj = -S_vals
#             if t is not None:
#                 dyz = y - z
#                 dist_yz2 = np.sum(dyz * dyz)
#                 obj += dist_yz2 / (2 * t)
#             return cast(float, obj)
#         if gr:
#             obj, grad = -S_vals[0], -S_vals[1]
#             if t is not None:
#                 dyz = y - z
#                 dist_yz2 = np.sum(dyz * dyz)
#                 obj += dist_yz2 / (2 * t)
#                 grad += dyz / t
#             return cast(float, obj), cast(np.ndarray, grad)

#     def prox_obj(y: np.ndarray, args: list) -> float:
#         return cast(float, prox_obj_and_grad(y, args, gr=False))

#     def prox_grad(y: np.ndarray, args: list) -> np.ndarray:
#         return cast(tuple[float, np.ndarray], prox_obj_and_grad(y, args, gr=True))[1]

#     y_init = np.array([1.0, 0.2]) if t is None else z

#     # check_gradient_scalar_function(prox_obj_and_grad, y_init, args=[])
#     # bs_error_abort("done")

#     mini = minimize_free(
#         prox_obj,
#         prox_grad,
#         x_init=y_init,
#         args=[],
#     )

#     if mini.success or mini.status == 2:
#         y = mini.x
#         return cast(np.ndarray, y)
#     else:
#         print(f"{mini.message}")
#         bs_error_abort(f"Minimization did not converge: status {mini.status}")
#         return None


# def add_results(
#     results: ScreeningResults,
# ) -> None:
#     """Adds more results to the `ScreeningResults` object

#     Args:
#         results: the results
#     """
#     model = results.model
#     N = model.N
#     theta_mat = model.theta_mat
#     s = cast(np.ndarray, model.params)[0]

#     FB_y = model.FB_y
#     SB_y = results.SB_y
#     FB_values_coverage = np.zeros(N)
#     FB_actuarial_premia = np.zeros(N)
#     SB_values_coverage = np.zeros(N)
#     SB_actuarial_premia = np.zeros(N)

#     for i in range(N):
#         FB_i = FB_y[i, :]
#         theta_i = theta_mat[i, :]
#         delta_i = theta_i[1]
#         FB_values_coverage[i] = b_fun(model, FB_i, theta=theta_i)
#         FB_actuarial_premia[i] = val_D(FB_i, delta_i, s)
#         SB_i = SB_y[i]
#         SB_values_coverage[i] = b_fun(model, SB_i, theta=theta_i)
#         SB_actuarial_premia[i] = val_D(SB_i, delta_i, s)

#     deltas = model.theta_mat[:, 1]
#     results.additional_results = [
#         FB_actuarial_premia,
#         SB_actuarial_premia,
#         cost_non_insur(model),
#         expected_positive_loss(deltas, s),
#         proba_claim(deltas, s),
#         FB_values_coverage,
#         SB_values_coverage,
#     ]
#     results.additional_results_names = [
#         "Actuarial premium at first-best",
#         "Actuarial premium at second-best",
#         "Cost of non-insurance",
#         "Expected positive loss",
#         "Probability of claim",
#         "Value of first-best coverage",
#         "Value of second-best coverage",
#     ]


# def plot_results(model: ScreeningModel) -> None:
#     model_resdir = cast(Path, model.resdir)
#     model_plotdir = str(cast(Path, model.plotdir))
#     df_all_results = (
#         pd.read_csv(model_resdir / "all_results.csv")
#         .rename(
#             columns={
#                 "FB_y_0": "First-best deductible",
#                 "y_0": "Second-best deductible",
#                 "y_1": "Second-best copay",
#                 "theta_0": "Risk-aversion",
#                 "theta_1": "Risk location",
#                 "FB_surplus": "First-best surplus",
#                 "SB_surplus": "Second-best surplus",
#                 "info_rents": "Informational rent",
#             }
#         )
#         .round(3)
#     )
#     df_all_results.loc[:, "Second-best copay"] = np.clip(
#         df_all_results["Second-best copay"].values, 0.0, 1.0
#     )

#     # first plot the first best
#     plot_calibration(df_all_results, path=model_plotdir + "/calibration")

#     display_variable(
#         df_all_results,
#         variable="First-best deductible",
#         theta_names=model.type_names,
#         cmap="viridis",
#         path=model_plotdir + "/first_best_deduc",
#     )

#     df_contracts = df_all_results[
#         [
#             "Risk-aversion",
#             "Risk location",
#             "First-best deductible",
#             "Second-best deductible",
#             "Second-best copay",
#         ]
#     ]

#     df_first_and_second = pd.DataFrame(
#         {
#             "Model": np.concatenate(
#                 (np.full(model.N, "First-best"), np.full(model.N, "Second-best"))
#             ),
#             "Risk-aversion": np.tile(df_contracts["Risk-aversion"].values, 2),
#             "Risk location": np.tile(df_contracts["Risk location"].values, 2),
#             "Deductible": np.concatenate(
#                 (
#                     df_contracts["First-best deductible"],
#                     df_contracts["Second-best deductible"],
#                 )
#             ),
#             "Copay": np.concatenate(
#                 (np.zeros(model.N), df_contracts["Second-best copay"].values)
#             ),
#         }
#     )

#     plot_contract_models(
#         df_first_and_second, "Deductible", path=model_plotdir + "/deducs_models"
#     )

#     plot_contract_models(
#         df_first_and_second, "Copay", path=model_plotdir + "/copays_models"
#     )

#     plot_contract_riskavs(
#         df_first_and_second,
#         "Deductible",
#         path=model_plotdir + "/deducs_riskavs",
#     )

#     plot_contract_riskavs(
#         df_first_and_second, "Copay", path=model_plotdir + "/copays_riskavs"
#     )

#     df_second = df_first_and_second.query('Model == "Second-best"')
#     plot_copays(df_second, path=model_plotdir + "/copays")

#     plot_best_contracts(
#         df_first_and_second,
#         path=model_plotdir + "/optimal_contracts",
#     )

#     plot_y_range(
#         df_first_and_second, model.contract_varnames, path=model_plotdir + "/y_range"
#     )

#     plot_second_best_contracts(
#         df_second,
#         title="Second-best contracts",
#         cmap="viridis",
#         path=model_plotdir + "/second_best_contracts",
#     )

#     IR_binds = np.loadtxt(model_resdir / "IR_binds.txt").astype(int).tolist()

#     IC_binds = np.loadtxt(model_resdir / "IC_binds.txt").astype(int).tolist()

#     plot_constraints(
#         df_all_results,
#         model.type_names,
#         IR_binds,
#         IC_binds,
#         path=model_plotdir + "/constraints",
#     )

#     plot_utilities(
#         df_all_results,
#         path=model_plotdir + "/utilities",
#     )
