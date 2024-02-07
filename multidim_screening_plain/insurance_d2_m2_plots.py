from typing import cast

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs_python_utils.bs_altair import _maybe_save

from multidim_screening_plain.utils import drawArrow, set_axis, set_colors


def plot_calibration(
    df_all_results: pd.DataFrame,
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    dfm = pd.melt(
        df_all_results,
        id_vars=["Risk-aversion", "Risk location"],
        value_vars=[
            "Actuarial premium at first-best",
            "Cost of non-insurance",
            "Expected positive loss",
            "Probability of claim",
            "Value of first-best coverage",
        ],
    )
    for var in ["Risk-aversion", "Risk location", "value"]:
        dfm[var] = dfm[var].round(2)
        risk_aversion_vals = sorted(dfm["Risk-aversion"].unique())

    our_colors = set_colors(risk_aversion_vals, interpolate=True)
    base = alt.Chart().encode(
        x=alt.X(
            "Risk location:Q",
            title="Risk location",
            scale=alt.Scale(domain=set_axis(dfm["Risk location"].values)),
        ),
        y=alt.Y("value:Q"),
        color=alt.Color("Risk-aversion:N", title="Risk-aversion", scale=our_colors),
        tooltip=["Risk-aversion", "Risk location", "value"],
    )
    ch_points = base.mark_point(filled=True, size=50)
    ch_lines = base.mark_line(strokeWidth=0.5)
    ch = (
        alt.layer(ch_points, ch_lines, data=dfm)
        .interactive()
        .facet(facet="variable:N", columns=2)
        .resolve_scale(y="independent")
    )
    if title:
        ch.properties(title=title)
    _maybe_save(ch, path)
    return cast(alt.Chart, ch)


def plot_utilities(
    df_all_results: pd.DataFrame,
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    df2 = df_all_results.copy()
    df2["Profit"] = df2["Second-best surplus"] - df2["Informational rent"]
    df2["Lost surplus"] = df2["First-best surplus"] - df2["Second-best surplus"]
    df2m = pd.melt(
        df2,
        id_vars=["Risk-aversion", "Risk location"],
        value_vars=["Informational rent", "Profit", "Lost surplus"],
    )
    # print(df2m)
    our_colors = alt.Scale(
        domain=["Informational rent", "Profit", "Lost surplus"],
        range=["blue", "green", "red"],
    )
    ch = (
        alt.Chart(df2m)
        .mark_bar()
        .encode(
            x=alt.X(
                "Risk-aversion:Q",
                scale=alt.Scale(domain=set_axis(df2["Risk-aversion"].values)),
            ),
            y=alt.Y("sum(value)"),
            color=alt.Color("variable", scale=our_colors),
            xOffset="variable:N",
        )
        .properties(width=150, height=120)
        .facet(facet="Risk location:N", columns=5)
    )
    _maybe_save(ch, path)


def plot_best_contracts(
    df_first_and_second: pd.DataFrame,
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    """plots the optimal contracts for both first and second best  in the type space
    the size of a point is proportional to (1 minus the copay)
    the color of a point indicates the deductible

    Args:
        df_first_and_second: a dataframe
        title: the title of the plot
        path: the path to save the plot
        **kwargs: additional arguments to pass to the plot

    Returns:
          the two interactive scatterplots.
    """
    deduc = df_first_and_second["Deductible"].values
    our_colors = set_colors(
        np.quantile(deduc, np.arange(10) / 10.0).tolist(), interpolate=True
    )
    ch = (
        alt.Chart(df_first_and_second)
        .mark_point(filled=True)
        .encode(
            x=alt.X(
                "Risk-aversion:Q",
                title="Risk-aversion",
                scale=alt.Scale(
                    domain=set_axis(df_first_and_second["Risk-aversion"].values)
                ),
            ),
            y=alt.Y(
                "Risk location:Q",
                title="Risk location",
                scale=alt.Scale(
                    domain=set_axis(df_first_and_second["Risk location"].values)
                ),
            ),
            color=alt.Color("Deductible:Q", scale=our_colors),
            size=alt.Size("Copay:Q", scale=alt.Scale(range=[500, 10])),
            tooltip=["Risk-aversion", "Risk location", "Deductible", "Copay"],
            facet=alt.Facet("Model:N", columns=2),
        )
        .interactive()
    )
    if title:
        ch = ch.properties(title=title)
    _maybe_save(ch, path)
    return cast(alt.Chart, ch)


def plot_second_best_contracts(
    df_second: pd.DataFrame,
    title: str | None = None,
    cmap=None,
    cmap_label: str | None = None,
    path: str | None = None,
    figsize: tuple[int, int] = (5, 5),
    **kwargs: dict | None,
) -> None:
    """the original scatterplot  of only the second-best contracts in the type space"""
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=kwargs)
    _ = ax.set_xlabel(r"Risk-aversion $\sigma$")
    _ = ax.set_ylabel(r"Risk location $\delta$")
    _ = ax.set_title(title)
    theta_mat = df_second[["Risk-aversion", "Risk location"]].values
    y_mat = df_second[["Deductible", "Copay"]].values
    scatter = ax.scatter(
        theta_mat[:, 0], theta_mat[:, 1], s=200 * (1.0 - y_mat[:, 1]), c=y_mat[:, 0]
    )
    _ = fig.colorbar(scatter, label=cmap_label)
    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


def display_variable(
    df_all_results: pd.DataFrame,
    variable: str,
    title: str | None = None,
    cmap=None,
    path: str | None = None,
    figsize: tuple[int, int] = (5, 5),
    **kwargs: dict | None,
) -> None:
    theta_mat = df_all_results[["Risk-aversion", "Risk location"]].values
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw=kwargs
    )  # subplot_kw=dict(aspect='equal',)
    _ = ax.set_xlabel(r"Risk-aversion $\sigma$")
    _ = ax.set_ylabel(r"Risk location $\delta$")
    _ = ax.set_title(title)
    scatter = ax.scatter(
        theta_mat[:, 0], theta_mat[:, 1], c=df_all_results[variable].values, cmap=cmap
    )
    _ = fig.colorbar(scatter, label=variable)
    if title is not None:
        _ = ax.set_title(title)

    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


def plot_y_range(
    df_first_and_second: pd.DataFrame,
    figsize: tuple[int, int] = (5, 5),
    s: int = 20,
    title: str | None = None,
    path: str | None = None,
    **kwargs,
) -> None:
    """the supposed stingray: the optimal contracts for both first and second best in contract space
    """
    first = df_first_and_second.query('Model == "First-best"')[["Deductible", "Copay"]]
    second = df_first_and_second.query('Model == "Second-best"')[
        ["Deductible", "Copay"]
    ]

    # discard y1 = 1
    second = second.query("Copay < 0.99")
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw=kwargs
    )  # subplot_kw=dict(aspect='equal',)
    _ = ax.scatter(
        second.Deductible.values,
        second.Copay.values,
        color="tab:blue",
        alpha=0.5,
        s=s,
        label="Second-best",
    )
    _ = ax.scatter(
        first.Deductible.values,
        first.Copay.values,
        color="tab:pink",
        s=s,
        label="First-best",
    )
    _ = ax.set_xlabel("Deductible")
    _ = ax.set_ylabel("Copay")
    _ = ax.set_title(title)
    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


def plot_contract_models(
    df_first_and_second: pd.DataFrame, varname: str, path: str | None = None, **kwargs
) -> alt.Chart:
    """plots a contract variable for both first and second best
    as a function of risk, with different colors by risk-aversion.

    Args:
        df_first_and_second: a dataframe
        varname: the contract variable
        path: the path to save the plot
        **kwargs: additional arguments to pass to the plot

    Returns:
        the two interactive scatterplots.
    """
    # discard y1 = 1
    df = df_first_and_second.query("Copay < 0.99")
    base = alt.Chart().encode(
        x=alt.X(
            "Risk location:Q",
            title="Risk location",
            scale=alt.Scale(domain=set_axis(df["Risk location"].values)),
        ),
        y=alt.Y(f"{varname}:Q"),
        color=alt.Color("Risk-aversion:N"),
        tooltip=["Risk-aversion", "Risk location", "Deductible", "Copay"],
    )
    ch_points = base.mark_point(filled=True, size=50)
    ch_lines = base.mark_line(strokeWidth=0.5)
    ch = alt.layer(ch_points, ch_lines, data=df).interactive().facet(column="Model:N")
    _maybe_save(ch, path)


def plot_contract_riskavs(
    df_first_and_second: pd.DataFrame, varname: str, path: str | None = None, **kwargs
) -> alt.Chart:
    """plots the optimal value of a contract variable for both first and second best
    as a function of risk, with different colors by risk-aversion.

    Args:
        df_first_and_second: a dataframe
        varname: the contract variable
        path: the path to save the plot
        **kwargs: additional arguments to pass to the plot

    Returns:
        as many interactive scatterplots as values of the risk-aversion parameter.
    """
    # discard y1 = 1
    df = df_first_and_second.query("Copay < 0.99")
    base = alt.Chart().encode(
        x=alt.X(
            "Risk location:Q",
            scale=alt.Scale(domain=set_axis(df["Risk location"].values)),
        ),
        y=alt.Y(f"{varname}:Q"),
        color=alt.Color("Model:N"),
        tooltip=["Risk-aversion", "Risk location", "Deductible", "Copay"],
    )
    ch_points = base.mark_point(filled=True, size=50)
    ch_lines = base.mark_line(strokeWidth=0.5)
    ch = (
        alt.layer(ch_points, ch_lines, data=df)
        .properties(width=150, height=120)
        .interactive()
        .facet(facet="Risk-aversion:N", columns=5)
    ).properties(title=f"Optimal {varname}")
    _maybe_save(ch, path)


def plot_copays(
    df_second: pd.DataFrame, path: str | None = None, **kwargs
) -> alt.Chart:
    """plots the optimal copay for the second best.

    Args:
        df_second: a dataframe
        path: the path to save the plot
        **kwargs: additional arguments to pass to the plot

    Returns:
        the interactive scatterplot.
    """
    # discard y1 = 1
    Copay = df_second.Copay.values
    df = df_second[Copay < 0.99]
    rng = np.random.default_rng()
    # jiggle the points a bit
    df.loc[:, "Copay"] += rng.uniform(0.0, 0.01, df.shape[0])
    base = alt.Chart(df).encode(
        x=alt.X(
            "Risk location:Q",
            title="Risk location",
            scale=alt.Scale(domain=set_axis(df["Risk location"].values)),
        ),
        y=alt.Y(
            "Copay:Q", title="Copay", scale=alt.Scale(domain=set_axis(df.Copay.values))
        ),
        color=alt.Color("Risk-aversion:N"),
        tooltip=["Risk-aversion", "Risk location", "Deductible", "Copay"],
    )
    ch_points = base.mark_point(filled=True, size=150)
    ch_lines = base.mark_line(strokeWidth=1)
    ch = (ch_points + ch_lines).interactive()
    _maybe_save(ch, path)


def plot_constraints(
    df_all_results: pd.DataFrame,
    IR_binds: list,
    IC_binds: list,
    figsize: tuple = (5, 5),
    s: float = 20,
    title: str | None = None,
    path: str | None = None,
    **kwargs,
) -> None:
    """the original scatterplot  of binding constraints.

    Args:
        df_all_results: the dataframe of results
        IR_binds: the list of types for which  IR binds
        IC_binds: the list of pairs (i, j) for which i is indifferent between his contract and j's

    Returns:
        nothing. Just plots the constraints.
    """
    theta_mat = df_all_results[["Risk-aversion", "Risk location"]].values.round(2)
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
        _ = drawArrow(
            ax,
            theta_mat[i, 0],
            theta_mat[j, 0],
            theta_mat[i, 1],
            theta_mat[j, 1],
        )
    _ = ax.set_xlabel(r"$\sigma$")
    _ = ax.set_ylabel(r"$\delta$")

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
