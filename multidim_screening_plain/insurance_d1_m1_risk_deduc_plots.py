from typing import cast

import altair as alt
import pandas as pd
from bs_python_utils.bs_altair import _maybe_save

from multidim_screening_plain.utils import set_axis


def plot_calibration(
    df_all_results: pd.DataFrame,
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    dfm = pd.melt(
        df_all_results,
        id_vars=["Risk location"],
        value_vars=[
            "Actuarial premium at first-best",
            "Cost of non-insurance",
            "Expected positive loss",
            "Probability of claim",
            "Value of first-best coverage",
        ],
    )
    for var in ["Risk location", "value"]:
        dfm[var] = dfm[var].round(3)

    base = alt.Chart().encode(
        x=alt.X(
            "Risk location:Q",
            title="Risk location",
            scale=alt.Scale(domain=set_axis(dfm["Risk location"].values)),
        ),
        y=alt.Y("value:Q"),
        tooltip=["Risk location", "value"],
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
        id_vars=["Risk location"],
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
                scale=alt.Scale(domain=set_axis(df2["Risk location"].values)),
            ),
            y=alt.Y("sum(value)"),
            color=alt.Color("variable", scale=our_colors),
            xOffset="variable:N",
        )
    )
    _maybe_save(ch, path)


def plot_deductible_models(
    df_first_and_second: pd.DataFrame, path: str | None = None, **kwargs
) -> alt.Chart:
    """plots the deuctible for both first and second best
    as a function of risk.

    Args:
        df_first_and_second: a dataframe
        varname: the contract variable
        path: the path to save the plot
        **kwargs: additional arguments to pass to the plot

    Returns:
        the two interactive scatterplots.
    """
    # discard y1 = 1
    df = df_first_and_second
    base = alt.Chart().encode(
        x=alt.X(
            "Risk location:Q",
            title="Risk location",
            scale=alt.Scale(domain=set_axis(df["Risk location"].values)),
        ),
        y=alt.Y("Deductible:Q"),
        tooltip=["Risk location", "Deductible"],
    )
    ch_points = base.mark_point(filled=True, size=50)
    ch_lines = base.mark_line(strokeWidth=0.5)
    ch = alt.layer(ch_points, ch_lines, data=df).interactive().facet(column="Model:N")
    _maybe_save(ch, path)
