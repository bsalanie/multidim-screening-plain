"""plots the distribution of losses"""

import altair as alt
import altair_saver as alt_save
import numpy as np
import pandas as pd
import scipy.stats as sts

n_delta_points = 5
n_loss_points = 100
delta_min, delta_max = 4.0, 8.0
s, k = 2.0, 0.012

deltas = np.linspace(delta_min, delta_max, num=n_delta_points)
losses = np.linspace(0.0, 15.0, num=n_loss_points)

pdf_losses = np.zeros((n_loss_points, n_delta_points))
for i, delta in enumerate(deltas):
    pdf_losses[:, i] = sts.norm.pdf(losses, loc=delta, scale=s)
p_0 = k * deltas
proba_positive_losses = p_0 * sts.norm.cdf(deltas / s)

print(proba_positive_losses)

dfl = pd.DataFrame(
    {
        "Losses": np.repeat(losses, n_delta_points),
        "Density": pdf_losses.reshape(pdf_losses.size),
        "Risk location": np.tile(deltas, n_loss_points),
        "Probability of losses": np.tile(proba_positive_losses, n_loss_points).round(3),
    }
)

print(dfl)

# ch = alt.Chart(dfl).mark_line().encode(
#     x=alt.X('Losses:Q', scale=alt.Scale()),
#     y=alt.Y('Density:Q'),
#     color='Risk location:N'
# ).facet('Risk location:N', columns=3)

width, height = 300, 300
ch = (
    alt.Chart(dfl)
    .mark_line()
    .encode(
        x=alt.X("Losses:Q", scale=alt.Scale(domain=[0.0, 15.0])),
        y=alt.Y("Density:Q"),
        color="Risk location:N",
    )
)
colors = ["blue", "orange", "red", "aquamarine", "green"]
ch += alt.Chart(dfl).mark_text(
    x=0.8 * width,
    align="center",
    y=0.05 * height,
    fontSize=12,
    color="black",
    text="Probability of losses:",
)
for i, _ in enumerate(deltas):
    ch += alt.Chart(dfl).mark_text(
        x=0.8 * width,
        align="center",
        y=0.1 * height + i * 0.06 * height,
        fontSize=12,
        color=colors[i],
        text=str(proba_positive_losses[i].round(3)),
    )


alt_save.save(ch, "distrib_losses.html")
