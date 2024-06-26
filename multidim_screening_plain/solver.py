"""Algorithm for multidimensional screening"""

# import multiprocessing as mp
from datetime import timedelta
from math import sqrt
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from bs_python_utils.bsnputils import ThreeArrays, npmaxabs
from bs_python_utils.bsutils import bs_error_abort

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults
from multidim_screening_plain.utils import (
    L2_norm,
    contracts_matrix,
    contracts_vector,
    print_matrix,
    print_row,
)


def construct_D(N: int) -> tuple[sparse.csr_matrix, float]:
    """Constructs the matrix `D` and the step size for the projection

    Args:
        N: number of types

    Returns:
        D: the `(N^2, N)` matrix `D`
        gamma_proj: the step size
    """
    D = sparse.vstack(
        [
            -sparse.eye(N)
            + sparse.coo_matrix(
                (
                    np.ones(N),
                    (
                        np.arange(N),
                        i * np.ones(N, dtype=int),
                    ),
                ),
                shape=(N, N),
            )
            for i in range(N)
        ],
        format="csr",
    )
    D = D[range(N * N)]
    _, s_proj, _ = sparse.linalg.svds(D, k=1, solver="arpack")  # norm_Lambda = s[0]
    gamma_proj = 1 / s_proj[0] ** 2
    return D, gamma_proj


def JLambda(model: ScreeningModel, y: np.ndarray) -> np.ndarray:
    """computes `Lambda'_{ij}(y) = b'_i(y_j)-b'_j(y_j)`

    Args:
        model: the ScreeningModel
        y: the contracts for all types, an array of size `m N`

    Returns:
        an `(m, N, N)` array.
    """
    b_function = model.b_function
    N, m = model.N, model.m
    # we compute the (N, N) matrices db_i/dy_0(y_j) and db_i/dy_1(y_j)
    _, db_vals = b_function(model, y, gr=True)
    J = np.zeros((m, N, N))
    for i in range(m):
        db_vals_i = db_vals[i, :, :]
        J[i, :, :] = db_vals_i - np.diag(db_vals_i)
    return J


def get_first_best(model: ScreeningModel) -> np.ndarray:
    """computes the first-best contracts for all types

    Args:
        model: the screening model

    Returns:
        the `(N, m)` matrix of first-best contracts.
    """
    prox_operator = model.proximal_operator_surplus
    theta_mat = model.theta_mat
    N, m = model.N, model.m

    y_first = np.empty((N, m))
    for i in range(N):
        theta_i = theta_mat[i, :]
        y_first[i, :] = prox_operator(model, theta_i)
        if i % 10 == 0:
            print(f"\n Done {i=} types out of {N}")
            print(f"\ni={i}:  theta is")
            print_row(theta_mat, i)
            print("\t\t first-best contract:")
            print_row(y_first, i)

    df_first_best_contracts = pd.DataFrame(
        y_first.round(6), columns=[f"y_{i}" for i in range(m)]
    )
    df_first_best_contracts.to_csv(
        cast(Path, model.resdir) / "first_best_contracts.csv", index=False
    )
    return cast(np.ndarray, y_first)


def D_mul(v: np.ndarray) -> np.ndarray:
    """computes `D v`

    Args:
        v: an `N`-vector

    Returns:
        an $N^d$-vector
    """
    N = v.size
    return np.add.outer(v, -v).reshape(N * N)


def D_star(v_mat: np.ndarray) -> Any:
    """computes $D^\ast v$

    Args:
        v_mat: an $(N, N)$-matrix

    Returns:
        an `N`-vector
    """
    return np.sum(v_mat, 1) - np.sum(v_mat, 0)


def nlLambda(
    model: ScreeningModel,
    y: np.ndarray,
) -> Any:
    """computes `Lambda_{ij}(y) = b_i(y_j)-b_j(y_j)`

    Args:
        y: the contracts, a vector of size `m N` (`y_0` then `y_1` etc)

    Returns:
        an `(N,N)` matrix.
    """
    b_function = model.b_function
    N = model.N
    b_vals = b_function(model, y)
    db = b_vals
    for j in range(N):
        db[:, j] -= b_vals[j, j]
    return db


############ ############ ############ ############ ############ ######
############  UNSUCCESSFUL ATTEMPT TO USE MULTIPROCESSING  ############
############ ############ ############ ############ ############ ######
# def prox_work_func_mp(model: ScreeningModel, list_working: list) -> list:
#     prox_operator = model.proximal_operator_surplus
#     len(list_working)
#     n_procs = mp.cpu_count() - 2
#     chunksize = 10  # 1 + n_working // n_procs
#     print(f"\n using {n_procs=} and {chunksize=}")

# chunksize, leftover = divmod(n_working, n_procs)
# for i in range(n_procs-1):
#     i0 = i * chunksize
#     i1 = (i + 1) * chunksize
#     list_working_i = list_working[i0:i1]
#     p = mp.Process(target=prox_operator, args=(list_working_i,))
#     res[i0:i1] = p.run()
# list_working_i += list_working[i1:]
# p = mp.Process(target=prox_operator, args=(list_working_i,))
# res[i1:] = p.run()

# with mp.Pool(processes=n_procs) as pool:
#     res = pool.starmap(prox_operator, list_working, chunksize=chunksize)

# # res: list = [None] * n_working
# # for i in range(n_working):
# #     arg_i = list_working[i]
# #     res[i] = prox_operator(*arg_i)
# return res


def prox_work_func(model: ScreeningModel, list_working: list) -> list:
    prox_operator = model.proximal_operator_surplus
    n_working = len(list_working)
    res: list = [None] * n_working
    for i in range(n_working):
        arg_i = list_working[i]
        res[i] = prox_operator(*arg_i)
    return res


def prox_minusS(
    model: ScreeningModel,
    z: np.ndarray,
    tau: float,
    y_current: np.ndarray,
    fix_top: bool = False,
    free_y: list | None = None,
) -> np.ndarray:
    """Proximal operator of `-S(y) = sum_i f_i H(y_i, theta_i)`

    Args:
        model: the model
        z: an `m N`-vector
        y_current: the current value of the `m N` vector of contracts
        tau: scale factor
        fix_top: True if first-best imposed at top
        free_y: a list of types for which we optimize over contracts

    Returns:
        the minimizing `y`, an `m N`-vector
    """
    theta_mat = model.theta_mat
    N = theta_mat.shape[0]
    m = model.m
    f = model.f

    # these are the types we will be working with
    Nmax = N - 1 if fix_top else N
    working_i0 = [i for i in free_y if i < Nmax] if free_y else list(range(Nmax))

    list_working = [
        [
            model,
            theta_mat[i, :],
            np.array([z[k * N + i] for k in range(m)]),
            tau * f[i],
        ]
        for i in working_i0
    ]
    n_working = len(list_working)

    y = y_current.copy()
    if fix_top:
        # we fix the second-best at the first-best at the top
        for k in range(m):
            y[k * N + N - 1] = model.FB_y[-1, k]
    # start_prox = perf_counter()
    res = prox_work_func(model, list_working)
    # end_prox = perf_counter()
    # print(f"\n the proximal operator took {end_prox - start_prox: > 10.5f} seconds")
    for i in range(n_working):
        res_i = res[i]
        i_working = working_i0[i]
        for k in range(m):
            y[i_working + k * N] = res_i[k]
    return y


def proj_K(
    model: ScreeningModel,
    w: np.ndarray,
    lamb: np.ndarray,
    gamma_proj: float,
    warmstart: bool = True,
    atol_proj: float = 1e-6,
    rtol_proj: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Projection of `w` onto `K` by Fast Projected Gradient

    Args:
        model: the model
        w: an `(N * N)` vector
        lamb: an `N` vector, the previous value
        warmstart: whether to start from the previous `lamb`
        gamma_proj: the step size
        atol_proj: absolute tolerance
        rtol_proj: relative tolerance
    Returns:
        the projection, the value of `lamb`,
        the number of iterations, and a status code
    """
    N = model.N
    eta = model.f

    it_max = 100_000
    lamb1 = lamb if warmstart else np.zeros(N)
    lamb_extra = lamb1
    converged: bool = False
    it = 0
    c_tol = rtol_proj * eta + atol_proj
    while it < it_max and not converged:
        lamb_old = lamb1
        Dw = np.clip(w - D_mul(lamb_extra), 0.0, None).reshape((N, N))
        lamb1 = np.clip(lamb_extra - gamma_proj * (eta - D_star(Dw)), 0.0, None)
        v = np.clip(w - D_mul(lamb1), 0.0, None)
        constraints = D_star(v.reshape((N, N))) - eta
        lamb_constraints = lamb1 @ constraints
        dvw = v - w
        dvw2 = np.sum(dvw * dvw)
        obj = 0.5 * dvw2 + lamb_constraints
        o_tol = rtol_proj * obj + atol_proj
        converged = cast(
            bool, np.all(constraints < c_tol) and abs(lamb_constraints) < o_tol
        )
        lamb_extra = lamb1 + (it - 1.0) / (it + 3.0) * (lamb1 - lamb_old)

    if converged:
        return v, lamb1, it, converged
    else:
        bs_error_abort("failed to converge.")
        return None


def solve(
    model: ScreeningModel,
    stepratio: float = 1.0,
    scale: bool = True,
    warmstart: bool = True,
    t_acc: float = 1.0,
    log: bool = True,
    it_max: int = 100_000,
    tol_primal: float = 1e-6,
    tol_dual: float = 1e-6,
    fix_top: bool = True,
    mult_fac: float = 1.0,
):
    # initialization
    N = model.N
    N2 = N * N
    model.rescale_step(mult_fac)

    _, gamma_proj = construct_D(N)

    criteria = np.array([False, False, False])
    rec_primal_residual = []
    rec_dual_residual = []
    rec_it_proj = []

    v = model.v0.reshape(N * N)
    y = model.y_init

    # scaling of the tolerances
    if scale:
        tol_primal = sqrt(len(y)) * tol_primal
        tol_dual = sqrt(len(v)) * tol_dual

    tau = 1.0 / (sqrt(stepratio) * model.norm_Lambda) / mult_fac
    sig = sqrt(stepratio) / model.norm_Lambda / mult_fac

    # loop
    lamb = np.zeros(N)
    JLy = JLambda(model, y)
    LTv = make_LTv(v.reshape((N, N)), JLy)

    t_start = timer()
    it = 0
    while it < it_max and not criteria.all():
        it += 1
        y_old = y.copy()
        v_old = v.copy()
        LTv_old = LTv.copy()
        # print(f"{LTv_old=}")
        lamb_old = lamb
        # primal update
        y = prox_minusS(
            model,
            y_old - tau * LTv_old,
            tau,
            y_old,
            fix_top=fix_top,
            free_y=model.free_y,
        )
        # print(f"{npmaxabs(y - y_old)=}")
        # dual update
        y_bar = y + t_acc * (y - y_old)
        Ly_bar = nlLambda(model, y_bar).reshape(N2)
        proj_res = cast(
            tuple[np.ndarray, np.ndarray, int, int],
            proj_K(
                model,
                v_old + sig * Ly_bar,
                lamb_old,
                gamma_proj,
                warmstart=warmstart,
            ),
        )
        v, lamb, n_it_proj, proj_converged = proj_res
        # print(f"{npmaxabs(v - v_old)=}")
        # bs_error_abort("stop")
        # Ly = nlLambda(model, y).reshape(N2)
        Ly = nlLambda(model, y).reshape(N2)
        JLy = JLambda(model, y)
        LTv = make_LTv(v.reshape((N, N)), JLy)
        # record
        norm1 = L2_norm((y_old - y) * t_acc / tau - (LTv_old - LTv))
        rec_primal_residual.append(norm1)
        norm2 = L2_norm((v_old - v) / sig + (Ly_bar - Ly))
        rec_dual_residual.append(norm2)
        rec_it_proj.append(n_it_proj)
        # stopping criterion
        criteria = np.array(
            [
                rec_primal_residual[-1] < tol_primal,
                rec_dual_residual[-1] < tol_dual,
                proj_converged,
            ]
        )
        y_mat = contracts_matrix(y, N)
        v_mat = v.reshape((N, N))
        if it % 100 == 0 and log:
            print("\n\ty is:")
            print_matrix(y_mat)
            np.savetxt(
                cast(Path, model.resdir) / "current_y.txt",
                np.round(y_mat, 6),
            )
            np.savetxt(
                cast(Path, model.resdir) / "current_v.txt",
                np.round(v_mat, 6),
            )
            print(f"\n\t\t\t{criteria=} at {it=}\n")
            print(f"\t\t\t\tprimal: {rec_primal_residual[-1] / tol_primal: > 10.2f}")
            print(f"\t\t\t\tdual: {rec_dual_residual[-1] / tol_dual: > 10.2f}")
    # end
    elapsed = timer() - t_start

    if log:
        print(
            f"convergence = {criteria.all()}, ",
            f" iterations = {it}, ",
            f"elapsed time = {timedelta(seconds=elapsed)}",
        )
        print(
            f"primal residual = {rec_primal_residual[-1] / tol_primal:.2e} tol, "
            f" dual residual = {rec_dual_residual[-1] / tol_dual:.2e} tol"
        )

    with open(cast(Path, model.resdir) / "log.txt", "a") as f:
        f.write(f"N = {N}, tau = {tau:.2e}, sig = {sig:.2e}, ")
        f.write(f"step ratio = {stepratio}, primal tol = {tol_primal:.2e}, ")
        f.write(
            f"dual tol = {tol_dual:.2e}\n",
        )
        if model.params is not None:
            params = cast(np.ndarray, model.params)
            params_names = cast(list, model.params_names)
            f.writelines(
                ", ".join(
                    [
                        key + f" = {value}"
                        for key, value in zip(params_names, params, strict=True)
                    ]
                )
                + "\n"
            )
        f.write(f"convergence = {criteria.all()}, iterations = {it}, ")
        f.write(
            f"elapsed time = {timedelta(seconds=elapsed)}\n",
        )
        f.write(f"primal residual = {rec_primal_residual[-1] / tol_primal:.2e} tol, ")
        f.write(f" dual residual = {rec_dual_residual[-1] / tol_dual:.2e} tol\n")
        f.write("\n")

    v_mat = v.reshape((N, N))

    # the original version had IC binding where v_mat > 0; I added a tolerance
    v_tol = tol_dual
    IC_binding = np.argwhere(v_mat > v_tol).astype(int)

    model_results = ScreeningResults(
        model=model,
        SB_y=y_mat,
        v_mat=v_mat,
        IC_binds=IC_binding,
        rec_primal_residual=rec_primal_residual,
        rec_dual_residual=rec_dual_residual,
        rec_it_proj=rec_it_proj,
        it=it,
        elapsed=elapsed,
    )

    return model_results


def make_LTv(v_mat: np.ndarray, JLy: np.ndarray) -> np.ndarray:
    """creates `(Lambda'(y))* v`

    Args:
        v_mat: an `(N, N)` matrix
        JLy: an `(m,N,N)` array

    Returns:
        an `m N`-vector
    """
    m, N = JLy.shape[:-1]
    LTv = np.empty(m * N)
    iN = 0
    for i in range(m):
        jN = iN + N
        LTv[iN:jN] = np.sum(JLy[i, :, :] * v_mat, 0)
        iN = jN
    return LTv


def compute_utilities(
    results: ScreeningResults,
    tol_fp: float = 1e-6,
) -> ThreeArrays:
    """Computes the rents for each type using the iterative algorithm `T_{Lambda}`
      of Prop 2

    Args:
        results: ScreeningResults
        tol_fp: tolerance for fixed point

    Returns:
        S_first: an `N`-vector of the values of the joint surplus in the first-best
        U_second: an `N`-vector of informational rents in the second-best
        S_second: an `N`-vector of the values of the joint surplus in the second-best
    """
    model = results.model
    N = model.N
    S_function = model.S_function
    y_first_best = model.FB_y
    theta_mat = model.theta_mat
    S_first = np.zeros(N)
    S_second = np.zeros(N)
    y_second_best = results.SB_y
    for i in range(N):
        theta = theta_mat[i, :]
        S_first[i] = S_function(model, y_first_best[i, :], theta=theta)
        S_second[i] = S_function(model, y_second_best[i, :], theta=theta)
    y_second = contracts_vector(y_second_best)
    Lambda_vals = nlLambda(model, y_second).reshape((N, N))

    U_second = U_old = np.zeros(N)
    dU = np.inf
    it_max = N
    it = 0
    while dU > tol_fp and it < it_max:
        for i in range(N):
            U_second[i] = np.max(Lambda_vals[i, :] + U_old)
        dU = npmaxabs(U_second - U_old)
        U_old = U_second
        it += 1

    tol = 1e-4
    results.IR_binds = np.argwhere(U_second < tol).flatten()

    return S_first, U_second, S_second
