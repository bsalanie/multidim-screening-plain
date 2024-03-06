from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from multidim_screening_plain.classes import ScreeningModel
from multidim_screening_plain.plot_utils import (
    display_variable_d2,
    melt_for_plots,
    plot_best_contracts_d2_m2,
    plot_constraints_d2,
    plot_contract_by_type_d2,
    plot_contract_models_d2,
    plot_second_best_contracts_d2_m2,
    plot_utilities_d2,
    plot_y_range_m2,
)


def general_plots(model: ScreeningModel, df_all_results: pd.DataFrame) -> None:
    theta_names = model.type_names
    contract_names = model.contract_varnames
    model_plotdir = str(cast(Path, model.plotdir))
    model_resdir = str(cast(Path, model.resdir))
    m = model.m

    # first plot the first best
    for j in range(m):
        display_variable_d2(
            df_all_results,
            f"First-best y_{j}",
            theta_names,
            cmap="viridis",
            path=model_plotdir + f"/first_best_y_{j}",
        )

    # now plot both together
    df_first_and_second = melt_for_plots(df_all_results, model)
    for j in range(m):
        plot_contract_models_d2(
            df_first_and_second,
            f"y_{j}",
            theta_names,
            title="y_0 in first and second best",
            path=model_plotdir + f"/y_{j}_models",
        )
        plot_contract_by_type_d2(
            df_first_and_second,
            f"y_{j}",
            theta_names,
            title=f"y_{j} by type",
            path=model_plotdir + f"/y_{j}_by_type",
        )

    plot_best_contracts_d2_m2(
        df_first_and_second,
        theta_names,
        contract_names,
        title="First-best and second-best contracts",
        path=model_plotdir + "/optimal_contracts",
    )

    plot_y_range_m2(
        df_first_and_second,
        contract_names=model.contract_varnames,
        title="Range of contracts",
        path=model_plotdir + "/y_range",
    )

    plot_second_best_contracts_d2_m2(
        df_first_and_second,
        theta_names,
        contract_names,
        title="Second-best contracts",
        cmap="viridis",
        path=model_plotdir + "/second_best_contracts",
    )

    IR_binds = np.loadtxt(model_resdir + "/IR_binds.txt").astype(int).tolist()

    IC_binds = np.loadtxt(model_resdir + "/IC_binds.txt").astype(int).tolist()

    plot_constraints_d2(
        df_all_results,
        theta_names,
        IR_binds,
        IC_binds,
        title="Binding IR and IC constraints",
        path=model_plotdir + "/constraints",
    )

    plot_utilities_d2(
        df_all_results,
        theta_names,
        title="Utilities",
        path=model_plotdir + "/utilities",
    )
