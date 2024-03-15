"""This sets up the model based in the configuration read from `config.env`.

A module `model_name.py` should provide the model-dependent functions and parameters:

- `create_initial_contracts`: initial values for the contracts, and chooses the
types for whom we optimize contracts

- `b_fun`: computes `b_i(y_j)` for all pairs of types `i` and  contracts `y_j`, and
its derivatives wrt to all dimensions of the contracts

- `S_fun`: computes `S_i(y_i)` for a type `i` and their contract `y_i`, and
its derivatives wrt to all dimensions of the contract

- `proximal_operator`: the proximal operator of the surplus function

- `additional_results`: additional results to be added to the `ScreeningResults` object

- `plot_results`: plots the results.

We use `importlib` to import the model-dependent module.
"""

import importlib
from typing import cast

import numpy as np
from bs_python_utils.bsutils import bs_error_abort, mkdir_if_needed, print_stars

from multidim_screening_plain.classes import ScreeningModel
from multidim_screening_plain.utils import (
    make_grid,
    parse_string,
    results_dir,
)


def setup_model(
    model_config: dict, model_type: str, model_instance: str
) -> ScreeningModel:
    """initializes the `ScreeningModel` object:
    fills in the dimensions, the numbers in each type, the characteristics of the types,
    the model parameters, and the directories.

    Args:
        model_config: the dictionary read from 'config.env'
        model_type: the first command line argument, e.g. "insurance"
        model_instance: the second command line argument, e.g. "d2_m2"

    Returns:
        the `ScreeningModel` object
    """
    model_name = f"{model_type}_{model_instance}"
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
    model_id = f"{model_name}_{str_dims_grid}{suffix}"
    resdir = mkdir_if_needed(results_dir / model_type / model_instance / model_id)
    plotdir = mkdir_if_needed(resdir / "plots")
    print(f"\nResults will be saved in {resdir}")
    print(f"Plots will be saved in {plotdir}\n")

    # model parameters
    n_params = int(cast(str, model_config["NUMBER_PARAMETERS"]))
    if n_params > 0:
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
        ).tolist()
    else:
        params = None
        params_names = None

    model_module = importlib.import_module(
        f".{model_type}.{model_instance}.{model_name}",
        package="multidim_screening_plain",
    )
    return ScreeningModel(
        f=f,
        model_id=model_id,
        theta_mat=theta_mat,
        type_names=type_names.tolist(),
        contract_varnames=contract_varnames.tolist(),
        params=params,
        params_names=params_names,
        m=m,
        resdir=resdir,
        plotdir=plotdir,
        model_module=model_module,
        proximal_operator_surplus=model_module.proximal_operator,
        b_function=model_module.b_fun,
        S_function=model_module.S_fun,
    )
