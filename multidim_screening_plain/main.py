"""Example with two-dimensional types (sigma=risk-aversion, delta=risk)
and two-dimensional contracts  (y0=deductible, y1=proportional copay)
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from dotenv import dotenv_values

from multidim_screening_plain.solver import (
    JLambda,
    compute_utilities,
    get_first_best,
    solve,
)
from multidim_screening_plain.specif import (
    add_results,
    initialize_contracts,
    plot,
    setup_model,
)

if __name__ == "__main__":
    # load configuration
    config = dotenv_values(Path.cwd() / "multidim_screening_plain" / "config.env")
    model = setup_model(config)
    precalculated = model.precalculate()

    print(model)

    do_first_best = config["DO_FIRST_BEST"] == "True"
    do_solve = config["DO_SOLVE"] == "True"
    start_from_first_best = config["START_FROM_FIRST_BEST"] == "True"
    start_from_current = not start_from_first_best
    do_plots = config["DO_PLOTS"] == "True"

    FB_y = np.empty((model.N, model.m))
    if do_first_best:
        # First let us look at the first best: we choose $y_i$ to maximize $S_i$ for each $i$.
        FB_y = get_first_best(model)
    else:
        FB_y = pd.read_csv(cast(Path, model.resdir) / "first_best_contracts.csv").values

    model.add_first_best(FB_y)

    print(model)

    if do_solve:  # we solve for the second best
        # initial values
        y_init, free_y = initialize_contracts(model, start_from_first_best, FB_y)
        JLy = JLambda(model, y_init, model.theta_mat, model.params)
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
        add_results(results)

        results.output_results()

        # print(model)

        print(results)

    if do_plots:
        plot(model)
