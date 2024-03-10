"""Plotting utilities"""

from typing import Any, cast

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs_python_utils.bs_altair import _maybe_save

from multidim_screening_plain.classes import ScreeningModel


def drawArrow_2dim(ax, xA, xB, yA, yB, c="k", ls="-"):
    n = 50
    x = np.linspace(xA, xB, 2 * n + 1)
    y = np.linspace(yA, yB, 2 * n + 1)
    ax.plot(x, y, color=c, linestyle=ls)
    ax.annotate(
        "",
        xy=(x[n], y[n]),
        xytext=(x[n - 1], y[n - 1]),
        arrowprops={"arrowstyle": "-|>", "color": c},
        size=15,
        # zorder=2,
    )


def set_colors(list_vals: list[Any], interpolate: bool = False) -> alt.Scale:
    """sets the colors at values of a variable.

    Args:
        list_vals: the values of the variable
        interpolate: whether we interpolate

    Returns:
        the color scheme.
    """
    n_vals = len(list_vals)
    list_colors = [
        "red",
        "lightred",
        "orange",
        "yellow",
        "lightgreen",
        "green",
        "lightblue",
        "darkblue",
        "violet",
        "black",
    ]
    if n_vals == 3:
        list_colors = [list_colors[i] for i in [0, 5, 7]]
    elif n_vals == 4:
        list_colors = [list_colors[i] for i in [0, 3, 6, 9]]
    elif n_vals == 5:
        list_colors = [list_colors[i] for i in [0, 2, 4, 6, 8]]
    elif n_vals == 7:
        list_colors = [list_colors[i] for i in [0, 1, 2, 4, 6, 8, 9]]
    if interpolate:
        our_colors = alt.Scale(domain=list_vals, range=list_colors, interpolate="rgb")
    else:
        our_colors = alt.Scale(domain=list_vals, range=list_colors)

    return our_colors


def set_axis(variable: np.ndarray, margin: float = 0.05) -> tuple[float, float]:
    """sets the axis for a plot with a margin

    Args:
        variable: the values of the variable
        margin: the margin to add, a fraction of the range of the variable

    Returns:
        the min and max for the axis.
    """
    x_min, x_max = variable.min(), variable.max()
    scaled_diff = margin * (x_max - x_min)
    x_min -= scaled_diff
    x_max += scaled_diff
    return x_min, x_max


def display_variable_d2(
    df_all_results: pd.DataFrame,
    variable: str,
    theta_names: list[str],
    title: str | None = None,
    cmap=None,
    path: str | None = None,
    figsize: tuple[int, int] = (5, 5),
    **kwargs: dict | None,
) -> None:
    theta_mat = df_all_results[theta_names].values
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw=kwargs
    )  # subplot_kw=dict(aspect='equal',)
    _ = ax.set_xlabel(theta_names[0])
    _ = ax.set_ylabel(theta_names[1])
    _ = ax.set_title(title)
    scatter = ax.scatter(
        theta_mat[:, 0],
        theta_mat[:, 1],
        c=df_all_results[variable].values,
        cmap=cmap,
    )
    _ = fig.colorbar(scatter, label=variable)
    if title is not None:
        _ = ax.set_title(title)

    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


def melt_for_plots(df_all_results: pd.DataFrame, model: ScreeningModel) -> pd.DataFrame:
    """melt the dataframe for plotting purposes

    Args:
        df_all_results: the dataframe of results
        model: the model

    Returns:
        a new dataframe with first and second best contracts.
    """
    theta_names, contract_names = model.type_names, model.contract_varnames
    df_first_and_second = pd.DataFrame(
        {
            "Model": np.concatenate(
                (np.full(model.N, "First-best"), np.full(model.N, "Second-best"))
            ),
        }
    )
    for theta_var in theta_names:
        df_first_and_second[theta_var] = np.tile(df_all_results[theta_var].values, 2)
    for contract_var in contract_names:
        df_first_and_second[contract_var] = np.concatenate(
            (
                df_all_results[f"First-best {contract_var}"],
                df_all_results[f"Second-best {contract_var}"],
            )
        )
    return df_first_and_second


def plot_y_range_m2(
    df_first_and_second: pd.DataFrame,
    contract_names: list[str],
    figsize: tuple[int, int] = (5, 5),
    s: int = 20,
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> None:
    """the supposed stingray: the optimal contracts for both first and second best in contract space"""
    print(df_first_and_second.columns)
    print(contract_names)
    first = df_first_and_second.query('Model == "First-best"')[contract_names]
    second = df_first_and_second.query('Model == "Second-best"')[contract_names]

    # # discard y1 = 1
    # second = second.query("Copay < 0.99")
    y_0_name, y_1_name = contract_names[0], contract_names[1]
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw=kwargs
    )  # subplot_kw=dict(aspect='equal',)
    _ = ax.scatter(
        second[y_0_name].values,
        second[y_1_name].values,
        color="tab:blue",
        alpha=0.5,
        s=s,
        label="Second-best",
    )
    _ = ax.scatter(
        first[y_0_name].values,
        first[y_1_name].values,
        color="tab:pink",
        s=s,
        label="First-best",
    )
    _ = ax.set_xlabel(y_0_name)
    _ = ax.set_ylabel(y_1_name)
    _ = ax.set_title(title)
    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


def plot_constraints_d2(
    df_all_results: pd.DataFrame,
    theta_names: list[str],
    IR_binds: list,
    IC_binds: list,
    figsize: tuple = (5, 5),
    s: float = 20,
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> None:
    """the original scatterplot of binding constraints.

    Args:
        df_all_results: the dataframe of results
        IR_binds: the list of types for which  IR binds
        IC_binds: the list of pairs (i, j) for which i is indifferent between his contract and j's

    Returns:
        nothing. Just plots the constraints.
    """
    theta_mat = df_all_results[theta_names].values.round(2)
    IC = "IC" if title else "IC binding"
    IR = "IR" if title else "IR binding"
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw=kwargs
    )  # subplot_kw=dict(aspect='equal',)
    _ = ax.scatter(
        theta_mat[:, 0],
        theta_mat[:, 1],
        facecolors="w",
        edgecolors="k",
        s=s,
        zorder=2.5,
    )
    _ = ax.scatter([], [], marker=">", c="k", label=IC)
    _ = ax.scatter(
        theta_mat[:, 0][IR_binds],
        theta_mat[:, 1][IR_binds],
        label=IR,
        c="tab:green",
        s=s,
        zorder=2.5,
    )
    for i, j in IC_binds:
        # if not (i in IR_binds or j in IR_binds):
        _ = drawArrow_2dim(
            ax,
            theta_mat[i, 0],
            theta_mat[j, 0],
            theta_mat[i, 1],
            theta_mat[j, 1],
        )
    _ = ax.set_xlabel(theta_names[0])
    _ = ax.set_ylabel(theta_names[1])

    if title is None:
        _ = ax.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncols=2,
            mode="expand",
            borderaxespad=0.0,
        )
    else:
        _ = ax.set_title(title)
        _ = ax.legend(bbox_to_anchor=(1.02, 1.0), loc="lower right")

    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


def plot_utilities_d1(
    df_all_results: pd.DataFrame,
    theta_name: str,
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    df2 = df_all_results.copy()
    df2["Profit"] = df2["Second-best surplus"] - df2["Informational rent"]
    df2["Lost surplus"] = df2["First-best surplus"] - df2["Second-best surplus"]
    df2m = pd.melt(
        df2,
        id_vars=[theta_name],
        value_vars=["Informational rent", "Profit", "Lost surplus"],
    )

    our_colors = alt.Scale(
        domain=["Informational rent", "Profit", "Lost surplus"],
        range=["blue", "green", "red"],
    )
    ch = (
        alt.Chart(df2m)
        .mark_bar()
        .encode(
            x=alt.X(
                "Risk location:Q",
                scale=alt.Scale(domain=set_axis(df2[theta_name].values)),
            ),
            y=alt.Y("value"),
            color=alt.Color("variable", scale=our_colors),
        )
        .facet(facet="variable:N", columns=3)
    )
    if title:
        ch.properties(title=title)
    _maybe_save(ch, path)


def plot_utilities_d2(
    df_all_results: pd.DataFrame,
    theta_names: list[str],
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    df2 = df_all_results.copy()
    theta_var1, theta_var2 = theta_names
    theta_vals1 = df2[theta_var1].values
    df2["Profit"] = df2["Second-best surplus"] - df2["Informational rent"]
    df2["Lost surplus"] = df2["First-best surplus"] - df2["Second-best surplus"]
    df2m = pd.melt(
        df2,
        id_vars=theta_names,
        value_vars=["Informational rent", "Profit", "Lost surplus"],
    )
    our_colors = alt.Scale(
        domain=["Informational rent", "Profit", "Lost surplus"],
        range=["blue", "green", "red"],
    )
    ch = (
        alt.Chart(df2m)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{theta_var1}:Q",
                scale=alt.Scale(domain=set_axis(theta_vals1)),
            ),
            y=alt.Y("sum(value)"),
            color=alt.Color("variable", scale=our_colors),
            xOffset="variable:N",
        )
        .properties(width=150, height=120)
        .facet(facet=f"{theta_var2}:N", columns=5)
    )
    if title:
        ch = ch.properties(title=title)
    _maybe_save(ch, path)


def plot_best_contracts_d2_m2(
    df_first_and_second: pd.DataFrame,
    theta_names: list[str],
    contract_names: list[str],
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    """plots the optimal contracts for both first and second best  in the type space
    the color of a point indicates the first contract variable
    the size of a point increases with the second contract variable

    Args:
        df_first_and_second: a dataframe
        theta_names: the names of the type characteristics in the dataframe
        contract_names: the names of the contract variables in the dataframe
        title: the title of the plot
        path: the path to save the plot
        kwargs: additional arguments to pass to the plot

    Returns:
          the two interactive scatterplots.
    """
    contract_var1, contract_var2 = contract_names
    contract_vals1 = df_first_and_second[contract_var1].values
    our_colors = set_colors(
        np.quantile(contract_vals1, np.arange(10) / 10.0).tolist(), interpolate=True
    )
    theta_var1, theta_var2 = theta_names
    theta_vals1 = df_first_and_second[theta_var1].values
    theta_vals2 = df_first_and_second[theta_var2].values
    ch = (
        alt.Chart(df_first_and_second)
        .mark_point(filled=True)
        .encode(
            x=alt.X(
                f"{theta_var1}:Q",
                title=theta_var1,
                scale=alt.Scale(domain=set_axis(theta_vals1)),
            ),
            y=alt.Y(
                f"{theta_var2}:Q",
                title=theta_var2,
                scale=alt.Scale(domain=set_axis(theta_vals2)),
            ),
            color=alt.Color(f"{contract_var1}:Q", scale=our_colors),
            size=alt.Size(f"{contract_var2}:Q", scale=alt.Scale(range=[50, 500])),
            tooltip=[theta_var1, theta_var2, contract_var1, contract_var2],
            facet=alt.Facet("Model:N", columns=2),
        )
        .interactive()
    )
    if title:
        ch = ch.properties(title=title)
    _maybe_save(ch, path)
    return cast(alt.Chart, ch)


def plot_second_best_contracts_d2_m2(
    df_first_and_second: pd.DataFrame,
    theta_names: list[str],
    contract_names: list[str],
    title: str | None = None,
    cmap_label: str | None = None,
    path: str | None = None,
    figsize: tuple[int, int] = (5, 5),
    **kwargs: dict | None,
) -> None:
    """the original scatterplot  of only the second-best contracts in the type space"""
    theta_var1, theta_var2 = theta_names
    df_second = df_first_and_second.query('Model == "Second-best"')
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=kwargs)
    _ = ax.set_xlabel(theta_var1)
    _ = ax.set_ylabel(theta_var2)
    if title:
        _ = ax.set_title(title)
    theta_mat = df_second[theta_names].values
    y_mat = df_second[contract_names].values
    min1, max1 = np.min(y_mat[:, 1]), np.max(y_mat[:, 1])
    scatter = ax.scatter(
        theta_mat[:, 0],
        theta_mat[:, 1],
        s=(y_mat[:, 1] - min1) / (max1 - min1) * 200 + 10,
        c=y_mat[:, 0],
    )
    _ = fig.colorbar(scatter, label=cmap_label)
    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


def plot_contract_models_d1(
    df_first_and_second: pd.DataFrame,
    varname: str,
    theta_name: str,
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    """plots a contract variable for both first and second best
    as a function of the type.

    Args:
        df_first_and_second: a dataframe
        varname: the contract variable
        theta_name: the name of the type in the dataframe
        title: a title for the plot
        path: the path to save the plot
        kwargs: additional arguments to pass to the plot

    Returns:
        the two interactive scatterplots.
    """
    df = df_first_and_second
    base = alt.Chart().encode(
        x=alt.X(
            f"{theta_name}:Q",
            title=theta_name,
            scale=alt.Scale(domain=set_axis(df[theta_name].values)),
        ),
        y=alt.Y(f"{varname}:Q"),
        tooltip=[theta_name, varname],
    )
    ch_points = base.mark_point(filled=True, size=50)
    ch_lines = base.mark_line(strokeWidth=0.5)
    ch = alt.layer(ch_points, ch_lines, data=df).interactive().facet(column="Model:N")
    if title:
        ch = ch.properties(title=title)
    _maybe_save(ch, path)


def plot_contract_models_d2(
    df_first_and_second: pd.DataFrame,
    varname: str,
    theta_names: list[str],
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    """plots a contract variable for both first and second best
    as a function of the first type characteristic,
    with different colors by the second type characteristic.

    Args:
        df_first_and_second: a dataframe
        varname: the contract variable
        theta_names: the names of the type characteristics in the dataframe
        title: a title for the plot
        path: the path to save the plot
        kwargs: additional arguments to pass to the plot

    Returns:
        the two interactive scatterplots.
    """
    df = df_first_and_second.copy()
    theta_var1, theta_var2 = theta_names
    theta_vals1 = df[theta_var1].values
    base = alt.Chart().encode(
        x=alt.X(
            f"{theta_var1}:Q",
            title=theta_var1,
            scale=alt.Scale(domain=set_axis(theta_vals1)),
        ),
        y=alt.Y(f"{varname}:Q"),
        color=alt.Color(f"{theta_var2}:Q"),
        tooltip=[theta_var1, theta_var2, varname],
    )
    ch_points = base.mark_point(filled=True, size=50)
    ch_lines = base.mark_line(strokeWidth=0.5)
    ch = alt.layer(ch_points, ch_lines, data=df).interactive().facet(column="Model:N")
    if title:
        ch = ch.properties(title=title)
    _maybe_save(ch, path)


def plot_contract_by_type_d2(
    df_first_and_second: pd.DataFrame,
    varname: str,
    theta_names: list[str],
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    """plots the optimal value of a contract variable for both first and second best
    as a function of the first type characteristic,
    with different colors by the second type characteristic.

    Args:
        df_first_and_second: a dataframe
        varname: the contract variable
        theta_names: the names of the type characteristics in the dataframe
        title: a title for the plot
        path: the path to save the plot
        kwargs: additional arguments to pass to the plot

    Returns:
        as many interactive scatterplots as values of the first type characteristic.
    """
    df = df_first_and_second.copy()
    theta_var1, theta_var2 = theta_names
    theta_vals1 = df[theta_var1].values
    base = alt.Chart().encode(
        x=alt.X(
            f"{theta_var1}:Q",
            scale=alt.Scale(domain=set_axis(theta_vals1)),
        ),
        y=alt.Y(f"{varname}:Q"),
        color=alt.Color("Model:N"),
        tooltip=[theta_var1, theta_var2, varname],
    )
    ch_points = base.mark_point(filled=True, size=50)
    ch_lines = base.mark_line(strokeWidth=0.5)
    ch = (
        alt.layer(ch_points, ch_lines, data=df)
        .properties(width=150, height=120)
        .interactive()
        .facet(facet=f"{theta_var2}:N", columns=5)
    ).properties(title=f"Optimal {varname}")
    if title:
        ch = ch.properties(title=title)
    _maybe_save(ch, path)
