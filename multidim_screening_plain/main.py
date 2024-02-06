"""Example with two-dimensional types (sigma=risk-aversion, delta=risk)
and two-dimensional contracts  (y0=deductible, y1=proportional copay)
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from bs_python_utils.bsutils import print_stars

from multidim_screening_plain.solver import (
    JLambda,
    compute_utilities,
    get_first_best,
    solve,
)
from multidim_screening_plain.specif import (
    add_results,
    initialize_contracts,
    model_name,
    plot,
    setup_model,
)

if __name__ == "__main__":
    print_stars(f"Running model {model_name}")

    model = setup_model(model_name)

    do_first_best = False
    do_solve = True
    do_plots = True

    start_from_first_best = False
    start_from_current = not start_from_first_best

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
        JLy = JLambda(y_init, model.theta_mat, model.params)
        model.initialize(y_init, free_y, JLy)

        results = solve(
            model,
            warmstart=True,
            scale=True,
            it_max=100_000,
            stepratio=1.0,
            tol_primal=1e-5,
            tol_dual=1e-5,
            fix_top=True,
        )

        S_first, U_second, S_second = compute_utilities(results)
        results.add_utilities(S_first, U_second, S_second)
        add_results(results)

        results.output_results()

        # print(model)

        print(results)

    if do_plots:
        plot(model)
