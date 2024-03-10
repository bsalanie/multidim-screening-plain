import numpy as np
import pandas as pd

from multidim_screening_plain.plot_utils import display_variable_d2


def plot_marginal_tax_rate(
    df_all_results: pd.DataFrame,
    title: str | None = None,
    path: str | None = None,
    cmap: str | None = None,
    **kwargs: dict | None,
) -> None:
    R, eta = 1.0, 1.0
    df2 = df_all_results.copy()
    endowments = df_all_results["Endowment"].values
    savings = df_all_results["Second-best Savings"].values
    exp_consos = np.exp(-eta * (endowments - savings))
    df2["Marginal tax rate"] = R - exp_consos
    theta_names = ["Endowment", "Disutility"]
    display_variable_d2(
        df2, "Marginal tax rate", theta_names, title=title, cmap=cmap, path=path
    )
