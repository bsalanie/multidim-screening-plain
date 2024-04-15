from typing import cast

import altair as alt
import pandas as pd
from bs_python_utils.bs_altair import _maybe_save

from multidim_screening_plain.plot_utils import set_axis


def plot_calibration(
    df_all_results: pd.DataFrame,
    title: str | None = None,
    path: str | None = None,
    **kwargs: dict | None,
) -> alt.Chart:
    dfm = pd.melt(
        df_all_results,
        id_vars=["Loss proba"],
        value_vars=[
            "Actuarial premium at first-best",
            "Cost of non-insurance",
            "Expected positive loss",
            "Probability of claim",
            "Value of first-best coverage",
        ],
    )
    for var in ["Loss proba", "value"]:
        dfm[var] = dfm[var].round(3)

    base = alt.Chart().encode(
        x=alt.X(
            "Loss proba:Q",
            title="Loss probability",
            scale=alt.Scale(domain=set_axis(dfm["Loss proba"].values)),
        ),
        y=alt.Y("value:Q"),
        tooltip=["Loss proba", "value"],
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
