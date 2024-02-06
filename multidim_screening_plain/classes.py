from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import scipy.linalg as spla
from bs_python_utils.bsnputils import check_matrix, check_vector
from bs_python_utils.bsutils import bs_error_abort

from multidim_screening_plain.utils import contracts_matrix, print_matrix


@dataclass
class ScreeningModel:
    """Create a model."""

    f: np.ndarray  # the numbers of individuals in each type
    theta_mat: np.ndarray  # the characteristics of the types
    params: np.ndarray  # the parameters of the model
    params_names: list
    m: int  # the dimension of the contracts
    model_id: str
    resdir: Path
    plotdir: Path

    N: int = field(init=False)  # the number of types
    d: int = field(init=False)  # their dimension
    v0: np.ndarray = field(init=False)
    y_init: np.ndarray = field(init=False)
    free_y: list = field(init=False)
    norm_Lambda: float = field(init=False)
    M: float = field(init=False)
    FB_y: np.ndarray = field(init=False)

    def __post_init__(self):
        self.N, self.d = check_matrix(self.theta_mat)
        N_f = check_vector(self.f)
        if N_f != self.N:
            bs_error_abort(
                f"Wrong number of rows of f: {N_f} but we have {self.N} types"
            )
        self.v0 = np.zeros((self.N, self.N))

    def add_first_best(self, y_first_best: np.ndarray):
        self.FB_y = y_first_best

    def initialize(self, y_init: np.ndarray, free_y: list, JLy: np.ndarray):
        self.y_init = y_init
        self.free_y = free_y
        self.norm_Lambda = max([spla.svdvals(JLy[i, :, :])[0] for i in range(self.d)])
        print("\n Free contracts in y_init:")
        print(free_y)
        print("\n Initial contracts:")
        print_matrix(contracts_matrix(y_init, self.N))

    def rescale_step(self, mult_fac: float) -> None:
        self.M = 2.0 * (self.N - 1) * mult_fac

    def __repr__(self) -> str:
        clstr = f"\nModel {self.model_id}: {self.N} {self.d}-dimensional types\n"
        clstr += f" and {self.m}-dimensional contracts\n"
        clstr += "    the model parameters are:\n"
        for name, par in zip(self.params_names, self.params, strict=True):
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

    def add_utilities(self, S_first, U_second, S_second) -> None:
        """Add the utilities to the results.

        Args:
            S_first: the first best surplus
            U_second: the second best informational rents
            S_second: the second best surplus
        """
        self.FB_surplus = S_first
        self.SB_surplus = S_second
        self.info_rents = U_second

    def output_results(self) -> None:
        """prints the optimal contracts, and saves a dataframe

        Args:
            self: the `Results`.
        """
        model = self.model
        theta_mat, y_mat = model.theta_mat, self.SB_y
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
        d, m = theta_mat.shape[1], y_mat.shape[1]
        for i in range(d):
            df_output[f"theta_{i}"] = model.theta_mat[:, i]
        for i in range(m):
            df_output[f"y_{i}"] = y_mat[:, i]
            df_output[f"FB_y_{i}"] = model.FB_y[:, i]

        FB_y_columns = [f"FB_y_{i}" for i in range(m)]
        y_columns = [f"y_{i}" for i in range(m)]
        theta_columns = [f"theta_{i}" for i in range(m)]
        df_output = df_output[
            theta_columns
            + FB_y_columns
            + y_columns
            + ["FB_surplus", "SB_surplus", "info_rents"]
        ]
        if self.additional_results and self.additional_results_names:
            for name, res in zip(
                self.additional_results_names, self.additional_results, strict=True
            ):
                df_output[name] = res.round(3)

        with pd.option_context(  # 'display.max_rows', None,
            "display.max_columns",
            None,
            "display.precision",
            3,
        ):
            print(df_output)

        model_resdir = cast(Path, model.resdir)
        df_output[y_columns].to_csv(
            model_resdir / "second_best_contracts.csv", index=False
        )
        np.savetxt(model_resdir / "IC_binds.txt", self.IC_binds)
        np.savetxt(model_resdir / "IR_binds.txt", self.IR_binds)
        np.savetxt(model_resdir / "v_mat.txt", self.v_mat)

        df_output.to_csv(model_resdir / "all_results.csv", index=False)

        # save the value of the parameters of the model
        df_params = pd.DataFrame()
        for k, v in zip(model.params_names, model.params, strict=True):
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
