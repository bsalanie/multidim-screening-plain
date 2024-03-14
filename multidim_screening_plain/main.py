"""Driver for the multidimensional screening program;
use as `python3 main.py config_file` where `config_file` is the name of the configuration file,
e.g. `insurance_d2_m2` for the insurance example with 2 dimensional types and contracts.

The driver program expects to find the model-specific code in `modelname.py`.
"""

from pathlib import Path
from typing import cast

import click
import pandas as pd
from dotenv import dotenv_values

from multidim_screening_plain.general_plots import general_plots
from multidim_screening_plain.setup import setup_model
from multidim_screening_plain.solver import (
    JLambda,
    compute_utilities,
    get_first_best,
    solve,
)


@click.command()
@click.argument("config_file")
def main(config_file):
    # load configuration
    config = dotenv_values(
        Path.cwd() / "multidim_screening_plain" / f"config_{config_file}.env"
    )

    model = setup_model(config)
    module = model.model_module

    print(model)

    do_first_best = config["DO_FIRST_BEST"] == "True"
    do_solve = config["DO_SOLVE"] == "True"
    start_from_first_best = config["START_FROM_FIRST_BEST"] == "True"
    do_plots = config["DO_PLOTS"] == "True"

    if do_first_best:
        # First let us look at the first best: we choose $y_i$ to maximize $S_i$ for each $i$.
        FB_y = get_first_best(model)
    else:
        FB_y = pd.read_csv(
            cast(Path, model.resdir) / "first_best_contracts.csv"
        ).values.reshape((model.N, model.m))

    model.add_first_best(FB_y)

    print(model)

    if do_solve:  # we solve for the second best
        # initial values
        y_init, free_y = module.create_initial_contracts(
            model, start_from_first_best, FB_y
        )
        JLy = JLambda(model, y_init)
        model.initialize(y_init, free_y, JLy)

        it_max = int(cast(str, config["MAX_ITERATIONS"]))
        tol = float(cast(str, config["TOLERANCE"]))

        results = solve(
            model,
            warmstart=True,
            scale=True,
            it_max=it_max,
            stepratio=1.0,
            tol_primal=tol,
            tol_dual=tol,
            fix_top=config["FIX_TOP"] == "True",
        )

        S_first, U_second, S_second = compute_utilities(results)
        results.add_utilities(S_first, U_second, S_second)

        # you may not want to do anything for excluded types;
        #   then just return `pass` from `module.adjust_excluded`
        module.adjust_excluded(results)

        # you may not want any additional results;
        #   then just return `pass` from `module.add_results`
        module.add_results(results)

        results.output_results()

        # print(model)

        # print(results)

    if do_plots:
        general_plots(model)

        # you may not want any additional plots;
        #   then just return `pass` from `module.add_itional_plots`
        module.add_plots(model)


if __name__ == "__main__":
    main()
