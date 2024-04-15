"""direct calculations"""

import altair as alt
import numpy as np
import pandas as pd
from altair_saver import save as alt_save
from bs_python_utils.bsnputils import print_quantiles

sigma, s, loading, loss = 0.6, 2.0, 0.25, 6.0

dim_grid = 50
plow, phigh = 0.04, 0.2
probas = np.linspace(plow, phigh, num=dim_grid)
dlow, dhigh = 0.0, loss
deducs = np.linspace(dlow, dhigh, num=dim_grid)


def log_arg(p, d):
    return 1.0 - p + p * np.exp(sigma * d)


def u(p, d):
    return -np.log(log_arg(p, d)) / sigma


def v(p, d):
    return u(p, d) - u(p, loss)


def u_p(p, d):
    return -(np.exp(sigma * d) - 1.0) / log_arg(p, d) / sigma


def v_p(p, d):
    return u_p(p, d) - u_p(p, loss)


def v_d(p, d):
    return -p * np.exp(sigma * d) / log_arg(p, d)


def v_pd(p, d):
    log_arg_val = log_arg(p, d)
    return -np.exp(sigma * d) / (log_arg_val * log_arg_val)


def v_ppd(p, d):
    exp_val = np.exp(sigma * d)
    log_arg_val = log_arg(p, d)
    return 2.0 * exp_val * (exp_val - 1.0) / (log_arg_val**3)


def v_pdd(p, d):
    exp_val = np.exp(sigma * d)
    log_arg_val = log_arg(p, d)
    return sigma * exp_val * (p * exp_val - 1.0 + p) / (log_arg_val**3)


def G(k, d):
    p = plow + (phigh - plow) / (dim_grid - 1) * k
    return (v(p, d) - (1.0 + loading) * p * (loss - d)) / dim_grid - (
        1.0 - (k + 1) / dim_grid
    ) * v_p(p, d)


def G_d(k, d):
    p = plow + (phigh - plow) / (dim_grid - 1) * k
    return (v_d(p, d) + (1.0 + loading) * p) / dim_grid - (
        1.0 - (k + 1) / dim_grid
    ) * v_pd(p, d)


def G_pd(k, d):
    p = plow + (phigh - plow) / (dim_grid - 1) * k
    return (2.0 * v_pd(p, d) + (1.0 + loading)) / dim_grid - (
        1.0 - (k + 1) / dim_grid
    ) * v_ppd(p, d)


def G_dd(k, d):
    p = plow + (phigh - plow) / (dim_grid - 1) * k
    return v_pd(p, d) / dim_grid - (1.0 - (k + 1) / dim_grid) * v_pdd(p, d)


G_vals = np.zeros((dim_grid, dim_grid))
Gp_vals = np.zeros((dim_grid, dim_grid))
Gd_vals = np.zeros((dim_grid, dim_grid))
Gpd_vals = np.zeros((dim_grid, dim_grid))
Gdd_vals = np.zeros((dim_grid, dim_grid))


p, d = np.meshgrid(probas, deducs)
for k in range(dim_grid):
    for ll in range(dim_grid):
        ded = deducs[ll]
        G_vals[k, ll] = G(k, ded)
        Gd_vals[k, ll] = G_d(k, ded)
        Gpd_vals[k, ll] = G_pd(k, ded)
        Gdd_vals[k, ll] = G_dd(k, ded)


qs = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]) / 10
print("G_vals:")
print_quantiles(G_vals, qs)
print("Gd_vals:")
print_quantiles(Gd_vals, qs)
print("Gpd_vals:")
print_quantiles(Gpd_vals, qs)
print("Gdd_vals:")
print_quantiles(Gdd_vals, qs)


def my_heatmap(variable, name):
    pl = pd.DataFrame(
        {"p": p.ravel(), "D": d.ravel(), name: np.clip(variable.ravel(), -0.2, 0.2)}
    )
    ch = (
        alt.Chart(pl)
        .mark_rect()
        .encode(x=alt.X("p:Q"), y=alt.Y("D:Q"), color=alt.Color(f"{name}:Q"))
    )
    alt_save(ch, f"{name}_vals.html")


my_heatmap(G_vals, "G")
my_heatmap(Gd_vals, "Gd")
my_heatmap(Gdd_vals, "Gdd")
my_heatmap(Gpd_vals, "Gpd")

# Gpl = sns.heatmap(G_vals)
# Gpl.set_title("Values of G")
# plt.xticks(probas, rotation=90)
# plt.savefig("G_vals.png")
# plt.clf()

# Gdpl = sns.heatmap(Gd_vals)
# Gdpl.set_title("Values of G_d")
# plt.savefig("Gd_vals.png")
# plt.clf()

# Gddpl = sns.heatmap(Gdd_vals)
# Gddpl.set_title("Values of G_dd")
# plt.savefig("Gdd_vals.png")
# plt.clf()

# Gpdpl = sns.heatmap(Gpd_vals)
# Gpdpl.set_title("Values of G_pd")
# plt.savefig("Gpd_vals.png")
# plt.clf()
