from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from bs_python_utils.bsutils import bs_error_abort

from multidim_screening_plain.classes import ScreeningModel
from multidim_screening_plain.plot_utils import (
    display_variable_d2,
    melt_for_plots,
    plot_best_contracts_d2_m2,
    plot_constraints_d1,
    plot_constraints_d2,
    plot_contract_by_type_d2,
    plot_contract_models_d1,
    plot_contract_models_d2,
    plot_second_best_contracts_d2_m2,
    plot_utilities_d1,
    plot_utilities_d2,
    plot_y_range_m1,
    plot_y_range_m2,
    setup_for_plots,
)


def general_plots(model: ScreeningModel) -> tuple[pd.DataFrame, str]:
    model_resdir = str(cast(Path, model.resdir))
    df_all_results, model_plotdir = setup_for_plots(model)
    d, m = model.d, model.m
    df_first_and_second = melt_for_plots(df_all_results, model)
    theta_names = model.type_names
    contract_names = model.contract_varnames

    if d == 2:
        for contract_var in contract_names:
            # first plot the first best
            display_variable_d2(
                df_all_results,
                f"First-best {contract_var}",
                theta_names,
                cmap="viridis",
                path=model_plotdir + f"/first_best_{contract_var}",
            )
            # now plot the second best
            display_variable_d2(
                df_all_results,
                f"Second-best {contract_var}",
                theta_names,
                cmap="viridis",
                path=model_plotdir + f"/second_best_{contract_var}",
            )
            # now plot both together
            plot_contract_models_d2(
                df_first_and_second,
                f"{contract_var}",
                theta_names,
                title=f"{contract_var} in first and second best",
                path=model_plotdir + f"/{contract_var}_models",
            )
            plot_contract_by_type_d2(
                df_first_and_second,
                f"{contract_var}",
                theta_names,
                title=f"{contract_var} by type",
                path=model_plotdir + f"/{contract_var}_by_type",
            )
    elif d == 1:
        for contract_var in contract_names:
            plot_contract_models_d1(
                df_first_and_second,
                f"{contract_var}",
                theta_names[0],
                title=f"{contract_var} in first and second best",
                path=model_plotdir + f"/{contract_var}_models",
            )
    else:
        bs_error_abort(f"plots for types of dimension {d} are not implemented yet.")

    if m == 2:
        plot_y_range_m2(
            df_first_and_second,
            contract_names,
            title="Range of contracts",
            path=model_plotdir + "/y_range",
        )
        if d == 2:
            plot_best_contracts_d2_m2(
                df_first_and_second,
                theta_names,
                contract_names,
                title="First-best and second-best contracts",
                path=model_plotdir + "/optimal_contracts",
            )
            plot_second_best_contracts_d2_m2(
                df_first_and_second,
                theta_names,
                contract_names,
                title="Second-best contracts",
                path=model_plotdir + "/second_best_contracts",
            )
    elif m == 1:
        plot_y_range_m1(
            df_first_and_second,
            contract_names[0],
            title="Range of contracts",
            path=model_plotdir + "/y_range",
        )

    IR_binds = np.loadtxt(model_resdir + "/IR_binds.txt").astype(int).tolist()
    IC_binds = np.loadtxt(model_resdir + "/IC_binds.txt").astype(int).tolist()

    if d == 2:
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
    elif d == 1:
        plot_constraints_d1(
            df_all_results,
            theta_names[0],
            IR_binds,
            IC_binds,
            title="Binding IR and IC constraints",
            path=model_plotdir + "/constraints",
        )
        plot_utilities_d1(
            df_all_results,
            theta_names[0],
            title="Utilities",
            path=model_plotdir + "/utilities",
        )
    else:
        bs_error_abort(f"plots for types of dimension {d} are not implemented yet.")

    return df_all_results, model_plotdir
