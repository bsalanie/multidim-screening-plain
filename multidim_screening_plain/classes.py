from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import cast

import numpy as np
import pandas as pd
import scipy.linalg as spla
from bs_python_utils.bsnputils import check_matrix, check_vector
from bs_python_utils.bsutils import bs_error_abort
from rich.console import Console
from rich.table import Table

from multidim_screening_plain.utils import contracts_matrix, print_matrix


@dataclass
class ScreeningModel:
    """Create a model.


    Attributes:
        f: an $N$-vector with the numbers of individuals in each type
        theta_mat: an $N \\times d$ matrix with the types
        type_names: a list of $d$ strings with the names of the types
        contract_varnames: a list of $m$ strings with the names of the contract variables
        params: a $p$-vector with the parameters of the model
        params_names: a list of $p$ strings with the names of the parameters
        m: the dimension of the contracts
        model_id: a string with the name of the model
        resdir: a Path to the directory where the results are stored
        plotdir: a Path to the directory where the plots are stored
        model_module: the module with the functions specific to the model
        b_function: the function  that computes $b_i(y_j)$ for all $i,j=1,\\ldots,n$
        S_function: the function that computes $S_i(y_i)$ for one type $i$
        proximal_operator_surplus: the proximal operator of the surplus
        N: the number of types
        d: their dimension
        v0: the initial values of the dual variables
        y_init: the initial values of the contracts
        free_y: the indices of the  contracts over which we optimize
        norm_lLambda: the norm of the $\\Lambda$ function
        M: the value of the $M$ parameter (for the step size)
        FB_y: thefirst-best contracts for the $N$ types
        precalculated_values: a dictionary with values that do not depend on the
            contracts and therefore can be calculated before the optimization.
    """

    f: np.ndarray
    theta_mat: np.ndarray
    type_names: list[str]
    contract_varnames: list[str]
    m: int  # the dimension of the contracts
    model_id: str
    resdir: Path
    plotdir: Path
    model_module: ModuleType
    b_function: Callable
    S_function: Callable
    proximal_operator_surplus: Callable

    N: int = field(init=False)
    d: int = field(init=False)  # their dimension
    v0: np.ndarray = field(init=False)
    y_init: np.ndarray = field(init=False)
    free_y: list = field(init=False)
    norm_Lambda: float = field(init=False)
    M: float = field(init=False)
    FB_y: np.ndarray = field(init=False)
    precalculated_values: dict = field(init=False)

    params: np.ndarray | None = None  # the parameters of the model
    params_names: list | None = None

    def __post_init__(self):
        self.N, self.d = check_matrix(self.theta_mat)
        N_f = check_vector(self.f)
        if N_f != self.N:
            bs_error_abort(
                f"Wrong number of rows of f: {N_f} but we have {self.N} types"
            )
        self.v0 = np.zeros((self.N, self.N))

    def add_first_best(self, y_first_best: np.ndarray):
        self.FB_y = y_first_best.reshape((self.N, self.m))

    def initialize(self, y_init: np.ndarray, free_y: list, JLy: np.ndarray):
        self.y_init = y_init
        self.free_y = free_y
        self.norm_Lambda = max([spla.svdvals(JLy[i, :, :])[0] for i in range(self.m)])
        print("\n Free contracts in y_init:")
        print(free_y)
        print("\n Initial contracts:")
        print_matrix(contracts_matrix(y_init, self.N))

    def rescale_step(self, mult_fac: float) -> None:
        self.M = 2.0 * (self.N - 1) * mult_fac

    def precalculate(self) -> dict:
        self.precalculated_values = self.model_module.precalculate_values()
        return self.precalculated_values

    def __repr__(self) -> str:
        clstr = f"\nModel {self.model_id}: {self.N} {self.d}-dimensional types:\n"
        for j in range(self.d - 1):
            clstr += f"    {self.type_names[j]} and "
        clstr += f"    {self.type_names[-1]}\n"
        clstr += f" and {self.m}-dimensional contracts:\n"
        for j in range(self.m - 1):
            clstr += f"    {self.contract_varnames[j]} and "
        clstr += f"    {self.contract_varnames[-1]}\n"
        if self.params is not None:
            params_names = cast(list, self.params_names)
            clstr += "    the model parameters are:\n"
            for name, par in zip(params_names, self.params, strict=True):
                clstr += f"    {name}: {par: > 10.3f}\n"
        return clstr + "\n"


@dataclass
class ScreeningResults:
    """Simulation results."""

    model: ScreeningModel
    SB_y: np.ndarray
    v_mat: np.ndarray
    IR_binds: np.ndarray
    IC_binds: np.ndarray
    rec_primal_residual: list
    rec_dual_residual: list
    rec_it_proj: list
    it: int
    elapsed: float
    info_rents: np.ndarray = field(init=False)
    FB_surplus: np.ndarray = field(init=False)
    SB_surplus: np.ndarray = field(init=False)
    additional_results: list | None = None
    additional_results_names: list | None = None

    def add_utilities(
        self, S_first: np.ndarray, U_second: np.ndarray, S_second: np.ndarray
    ) -> None:
        """Add the utilities to the results.

        Args:
            S_first: the first best surplus
            U_second: the second best informational rents
            S_second: the second best surplus
        """
        self.FB_surplus = S_first
        self.SB_surplus = S_second
        self.info_rents = U_second

    def make_table(self, df_output) -> Table:
        model = self.model
        d, m = model.d, model.m
        table = Table(title=f"Optimal contracts for {model.model_id}")
        theta_names = model.type_names
        for i in range(d):
            table.add_column(
                theta_names[i], justify="center", style="red", no_wrap=True
            )
        contract_varnames = model.contract_varnames
        for j in range(m):
            table.add_column(
                f"FB {contract_varnames[j]}",
                justify="center",
                style="blue",
                no_wrap=True,
            )
        for j in range(m):
            table.add_column(
                f"SB {contract_varnames[j]}",
                justify="center",
                style="green",
                no_wrap=True,
            )
        table.add_column("1B surplus", justify="center", style="red", no_wrap=True)
        table.add_column("2B surplus", justify="center", style="blue", no_wrap=True)
        table.add_column("Info. rent", justify="center", style="black", no_wrap=True)

        elements_list = [df_output["theta_0"]]
        for i in range(1, d):
            elements_list.append(df_output[f"theta_{i}"])
        for j in range(m):
            elements_list.append(df_output[f"FB_y_{j}"])
        for j in range(m):
            elements_list.append(df_output[f"y_{j}"])
        elements_list.extend(
            [df_output["FB_surplus"], df_output["SB_surplus"], df_output["info_rents"]]
        )

        for elements_row in zip(*elements_list, strict=True):
            gen_row = (f"{x: > 8.3f}" for x in elements_row)
            table.add_row(*gen_row)

        return table

    def output_results(self) -> None:
        """prints the optimal contracts, and saves the results in a dataframe."""
        model = self.model
        y_mat = self.SB_y
        df_output = pd.DataFrame(
            {
                "theta_0": model.theta_mat[:, 0],
                "y_0": y_mat[:, 0],
                "FB_y_0": model.FB_y[:, 0],
                "FB_surplus": self.FB_surplus,
                "SB_surplus": self.SB_surplus,
                "info_rents": self.info_rents,
            }
        )
        d, m = model.d, model.m
        for i in range(1, d):
            df_output[f"theta_{i}"] = model.theta_mat[:, i]
        for i in range(1, m):
            df_output[f"y_{i}"] = y_mat[:, i]
            df_output[f"FB_y_{i}"] = model.FB_y[:, i]

        FB_y_columns = [f"FB_y_{i}" for i in range(m)]
        y_columns = [f"y_{i}" for i in range(m)]
        theta_columns = [f"theta_{i}" for i in range(d)]
        df_output = df_output[
            theta_columns
            + FB_y_columns
            + y_columns
            + ["FB_surplus", "SB_surplus", "info_rents"]
        ]

        if self.additional_results and self.additional_results_names:
            additional_results_names = cast(list, self.additional_results_names)
            additional_results = cast(list, self.additional_results)
            for name, res in zip(
                additional_results_names, additional_results, strict=True
            ):
                df_output[name] = res.round(3)

        console = Console()
        console.print("\n" + "-" * 80 + "\n", style="bold blue")
        table = self.make_table(df_output)
        console.print(table)

        model_resdir = cast(Path, model.resdir)
        df_output[y_columns].to_csv(
            model_resdir / "second_best_contracts.csv", index=False
        )
        np.savetxt(model_resdir / "IC_binds.txt", self.IC_binds)
        np.savetxt(model_resdir / "IR_binds.txt", self.IR_binds)
        np.savetxt(model_resdir / "v_mat.txt", self.v_mat)

        df_output.to_csv(model_resdir / "all_results.csv", index=False)

        # save the value of the parameters of the model
        if model.params and model.params_names:
            params_names = cast(list, model.params_names)
            params = cast(np.ndarray, model.params)
            df_params = pd.DataFrame()
            for k, v in zip(params_names, params, strict=True):
                df_params[k] = [v]
            df_params.to_csv(model_resdir / "params.csv", index=False)

    def __repr__(self) -> str:
        model = self.model
        clstr = model.__repr__()
        clstr += "\n    the optimal contracts are:\n"
        clstr += (
            "         type:                         first-best                  "
            " second-best:\n"
        )
        for i in range(model.N):
            for j in range(model.d):
                clstr += f"{model.theta_mat[i, j]: >8.3f} "
            clstr += ":\t"
            for k in range(model.m):
                clstr += f"{model.FB_y[i, k]: >8.3f} "
            clstr += ";\t"
            for k in range(model.m):
                clstr += f"{self.SB_y[i, k]: >8.3f} "
            clstr += "\n"

        return clstr + "\n"
