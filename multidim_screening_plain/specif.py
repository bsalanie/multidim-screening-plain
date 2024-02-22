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
from typing import cast

import numpy as np
from bs_python_utils.bsutils import bs_error_abort, mkdir_if_needed, print_stars

from multidim_screening_plain.classes import ScreeningModel, ScreeningResults
from multidim_screening_plain.utils import (
    make_grid,
    parse_string,
    plots_dir,
    results_dir,
)


def setup_model(model_config: dict) -> ScreeningModel:
    """initializes the `ScreeningModel` object:
    fills in the dimensions, the numbers in each type, the characteristics of the types,
    the model parameters, and the directories.

    Args:
        model_config: the dictionary read from 'config.env'

    Returns:
        the `ScreeningModel` object
    """
    model_name = cast(str, model_config["MODEL_NAME"])
    print_stars(f"Running model {model_name}")

    # first we deal with the types
    d = int(cast(str, model_config["DIMENSION_TYPES"]))
    type_names = parse_string(
        cast(str, model_config["TYPES_NAMES"]), d, ",", "type names", "str"
    )
    str_dims_grid = cast(str, model_config["TYPES_GRID_SIZE"])
    dims_grid = parse_string(str_dims_grid, d, "x", "types", "int")
    N = np.prod(dims_grid)  # number of types
    # the grid of types
    str_grid_mins = cast(str, model_config["TYPES_MINIMA"])
    grid_mins = parse_string(str_grid_mins, d, ",", "minima of types", "float")
    str_grid_maxs = cast(str, model_config["TYPES_MAXIMA"])
    grid_maxs = parse_string(str_grid_maxs, d, ",", "maxima of types", "float")
    theta: list[np.ndarray] = [np.zeros(1)] * d
    type_distrib = cast(str, model_config["TYPES_DISTRIBUTION"])
    if type_distrib == "uniform":
        f = np.ones(N)  # weights of distribution
        for j in range(d):
            if grid_mins[j] >= grid_maxs[j]:
                bs_error_abort(
                    f"Wrong bounds for the types: {grid_mins[j]} >= {grid_maxs[j]}"
                )
            else:
                print(
                    f"Dimension {j+1} of the types: from {grid_mins[j]} to"
                    f" {grid_maxs[j]}"
                )
                theta[j] = np.linspace(grid_mins[j], grid_maxs[j], num=dims_grid[j])
    else:
        bs_error_abort(
            f"Unknown distribution of types: {type_distrib}; only uniform is"
            " implemented."
        )
    theta_mat = make_grid(theta)  # an (N,d) matrix

    # dimension of contracts
    m = int(cast(str, model_config["NUMBER_CONTRACT_VARIABLES"]))
    contract_varnames = parse_string(
        cast(str, model_config["CONTRACT_VARIABLE_NAMES"]),
        m,
        ",",
        "contract variable names",
        "str",
    )

    suffix = ""
    case = f"N{N}{suffix}"
    model_id = f"{model_name}_{case}"
    resdir = mkdir_if_needed(results_dir / model_id)
    plotdir = mkdir_if_needed(plots_dir / model_id)

    # model parameters
    n_params = int(cast(str, model_config["NUMBER_PARAMETERS"]))
    params = parse_string(
        cast(str, model_config["PARAMETERS"]),
        n_params,
        ",",
        "parameters",
        "float",
    )
    params_names = parse_string(
        cast(str, model_config["PARAMETER_NAMES"]),
        n_params,
        ",",
        "parameter names",
        "str",
    )
    model_module = importlib.import_module(
        f".{model_name}", package="multidim_screening_plain"
    )
    return ScreeningModel(
        f=f,
        model_id=model_id,
        theta_mat=theta_mat,
        type_names=type_names.tolist(),
        contract_varnames=contract_varnames.tolist(),
        params=params,
        params_names=params_names.tolist(),
        m=m,
        resdir=resdir,
        plotdir=plotdir,
        model_module=model_module,
        proximal_operator_surplus=model_module.proximal_operator,
        b_function=model_module.b_fun,
        S_function=model_module.S_fun,
    )


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
        tuple[np.ndarray, list]: initial contracts (an `(N,m)` matrix)
        and a list of types for whom we optimize contracts.
    """
    return cast(
        tuple[np.ndarray, list],
        model.model_module.create_initial_contracts(
            model, start_from_first_best, y_first_best_mat
        ),
    )


def add_results(
    results: ScreeningResults,
) -> None:
    """Adds more results to the `ScreeningResults` object

    Args:
        results: the results
    """
    model = results.model
    model.model_module.additional_results(results)


def plot(model: ScreeningModel) -> None:
    """Plots the results

    Args:
        model: The ScreeningModel
    """
    model.model_module.plot_results(model)
