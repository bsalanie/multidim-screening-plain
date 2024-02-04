"""the model-dependent functions and parameters are defined in  `model_name`.py
we use `importlib` to import what is needed
"""

import importlib
from typing import Any, cast

import numpy as np

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults

# we provide the name of the model
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


def b_function(y: np.ndarray, theta_mat: np.ndarray, params: np.ndarray) -> Any:
    """The b function

    Args:
        y: an $m k$-vector of $k$ contracts
        theta_mat: a $(q,d)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(q,k)$-matrix
    """
    return model_module.b_fun(y, theta_mat, params)


def b_deriv(y: np.ndarray, theta_mat: np.ndarray, params: np.ndarray) -> Any:
    """The derivatives of the b function

    Args:
        y: an $m k$-vector of $k$ contracts
        theta_mat: a $(q,d)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        an $(m,q,k)$-array
    """
    return model_module.db_fun(y, theta_mat, params)


def S_function(y: np.ndarray, theta_mat: np.ndarray, params: np.ndarray) -> Any:
    """The S function

    Args:
        y: an $m k$-vector of $k$ contracts
        theta_mat: a $(q,d)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        a $(q,k)$-matrix
    """
    return model_module.S_fun(y, theta_mat, params)


def S_deriv(y: np.ndarray, theta_mat: np.ndarray, params: np.ndarray) -> Any:
    """The derivatives of the S function

    Args:
        y:  an $m k$-vector of $k$ contracts
        theta_mat: a $(q,d)$-vector of characteristics of types
        params: the parameters of the model

    Returns:
        an $(m,q,k)$-array
    """
    return model_module.dS_fun(y, theta_mat, params)


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
