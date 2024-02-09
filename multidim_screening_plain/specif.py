"""This is an interface module. The user should specify the `model_name`; 
and, in a module `model_name.py`, the model-dependent functions and parameters:
- `create_model`: creates the `ScreeningModel`
- `create_initial_contracts`: initial values for the contracts, and chooses the 
types for whom we optimize contracts
- `b_fun`: computes $b_i(y_j)$ for types $i$ and  contracts $y_j$
- `db_fun`: its derivatives wrt to all dimensions of the contracts
- `S_fun`: computes $S_i(y_j)$ for types $i$ and  contracts $y_j$
- `dS_fun`: its derivatives wrt to all dimensions of the contracts
- `proximal_operator`: the proximal operator of the surplus function
- `additional_results`: additional results to be added to the `ScreeningResults` object
- `plot_results`: plots the results.

We use `importlib` to import the model-dependent module.
"""
import importlib
from typing import Any, cast

import numpy as np

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults

# ONLY CHANGE THIS LINE: PROVIDE THE NAME OF THE MODEL
model_name = "insurance_d2_m2"

# DO NOT CHANGE BELOW THIS LINE
model_module = importlib.import_module(
    f".{model_name}", package="multidim_screening_plain"
)


def setup_model(model_name: str) -> ScreeningModel:
    screening_model = model_module.create_model(model_name)
    return cast(ScreeningModel, screening_model)


def initialize_contracts(
    model: ScreeningModel,
    start_from_first_best: bool,
    y_first_best_mat: np.ndarray | None = None,
) -> tuple[np.ndarray, list]:
    """Initializes the contracts for the second best problem

    Args:
        model: the screening model
        start_from_first_best: whether to start from the first best
        y_first_best_mat: the `(N, m)` matrix of first best contracts. Defaults to None.

    Returns:
        tuple[np.ndarray, list]: initial contracts (an `(N,m)` matrix) and a list of types for whom
         we optimize contracts
    """
    return cast(
        tuple[np.ndarray, list],
        model_module.create_initial_contracts(
            model, start_from_first_best, y_first_best_mat
        ),
    )


def b_function(
    y: np.ndarray, theta_mat: np.ndarray, params: np.ndarray, gr: bool = False
) -> Any:
    """The b function

    Args:
        y: an $m k$-vector of $k$ contracts
        theta_mat: a $(q,d)$-vector of characteristics of types
        params: the parameters of the model
        gr: whether we compute the gradient

    Returns:
        a $(q,k)$-matrix, and a $(m,q,k)$ array if `gr` is `True`
    """
    return model_module.b_fun(y, theta_mat, params, gr)


# def b_deriv(y: np.ndarray, theta_mat: np.ndarray, params: np.ndarray) -> Any:
#     """The derivatives of the b function

#     Args:
#         y: an $m k$-vector of $k$ contracts
#         theta_mat: a $(q,d)$-vector of characteristics of types
#         params: the parameters of the model

#     Returns:
#         an $(m,q,k)$-array
#     """
#     return model_module.db_fun(y, theta_mat, params)


def S_function(
    y: np.ndarray, theta_mat: np.ndarray, params: np.ndarray, gr: bool = False
) -> Any:
    """The S function and maybe its gradient

    Args:
        y: an $m k$-vector of $k$ contracts
        theta_mat: a $(q,d)$-vector of characteristics of types
        params: the parameters of the model
        gr: whether we compute the gradient

    Returns:
        a $(q,k)$-matrix, and a $(m,q,k)$ array if `gr` is `True`
    """
    return model_module.S_fun(y, theta_mat, params, gr)


# def S_deriv(y: np.ndarray, theta_mat: np.ndarray, params: np.ndarray) -> Any:
#     """The derivatives of the S function

#     Args:
#         y:  an $m k$-vector of $k$ contracts
#         theta_mat: a $(q,d)$-vector of characteristics of types
#         params: the parameters of the model

#     Returns:
#         an $(m,q,k)$-array
#     """
#     return model_module.dS_fun(y, theta_mat, params)


def proximal_operator_surplus(
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
    return model_module.proximal_operator(z, theta, params, t)


def add_results(
    results: ScreeningResults,
) -> None:
    """Adds more results to the `ScreeningResults` object

    Args:
        results: the results
    """
    return cast(None, model_module.additional_results(results))


def plot(model: ScreeningModel) -> None:
    """Plots the results

    Args:
        results: the results
    """
    model_module.plot_results(model)
