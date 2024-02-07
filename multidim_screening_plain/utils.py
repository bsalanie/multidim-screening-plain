from math import sqrt
from pathlib import Path
from typing import Any, cast

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results_dir = Path.cwd() / "results"
plots_dir = Path.cwd() / "plots"


INV_SQRT_2 = np.sqrt(0.5)
INV_SQRT_2PI = 1.0 / np.sqrt(2 * np.pi)


def add_to_each_col(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """adds a vector to each column of a matrix

    Args:
        mat: a $(q, k)$-matrix
        vec: a $q$-vector

    Returns:
        a $(q, k)$-matrix, the result of adding `vec` to each column of `mat`
    """
    q, k = mat.shape
    c = np.empty((q, k))
    for i in range(k):
        c[:, i] = mat[:, i] + vec
    return c


def multiply_each_col(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """multiplies each column of a matrix by a vector

    Args:
        mat: a $(q, k)$-matrix
        vec: a $q$-vector

    Returns:
        a $(q, k)$-matrix, the result of multiplying each column of `mat` by  `vec`
    """
    q, k = mat.shape
    c = np.empty((q, k))
    for i in range(k):
        c[:, i] = mat[:, i] * vec
    return c


def my_outer_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """outer sum of two vectors

    Args:
        a: a $q$-vector
        b: a $k$-vector

    Returns:
        a $(q,k)$-matrix, the outer sum of `a` and `b`
    """
    q, k = a.size, b.size
    c = np.empty((q, k))
    for i in range(q):
        c[i, :] = a[i] + b
    return c


def print_row(matrix: np.ndarray, row: int) -> None:
    """prints a row of a matrix."""
    print(" ".join(f"{matrix[row, j]: > 10.4f}" for j in range(matrix.shape[1])))


def print_matrix(matrix: np.ndarray) -> None:
    """prints a matrix."""
    for i in range(matrix.shape[0]):
        print_row(matrix, i)
    print("\n")


def bs_norm_pdf(x: np.ndarray) -> np.ndarray:
    """normal PDF."""
    return cast(np.ndarray, INV_SQRT_2PI * np.exp(-0.5 * x * x))


def bs_norm_cdf(x: np.ndarray) -> np.ndarray:
    """Fast standard normal CDF based on Numerical Recipes.

    This function is accurate to 6 decimal places.
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    z = x * INV_SQRT_2
    sign_z = np.sign(z)
    abs_z = np.abs(z)
    t = 1.0 / (1.0 + p * abs_z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)

    return cast(np.ndarray, (1.0 + sign_z * y) / 2.0)


def contracts_vector(y_mat: np.ndarray) -> np.ndarray:
    """converts a matrix of contracts to a vector

    Args:
        y_mat: an `(N,m)` matrix

    Returns:
        y_vec: an `m*N`-vector (y[:,0] first, then y[:,1] etc)
    """
    m = y_mat.shape[1]
    y_vec = np.concatenate([y_mat[:, i] for i in range(m)])
    return cast(np.ndarray, y_vec)


def contracts_matrix(y_vec: np.ndarray, N: int) -> np.ndarray:
    """converts a vector of contracts to a matrix

    Args:
        y_vec: an `m*N`vector (y[:,0] first, then y[:,1] etc)

    Returns:
        y_mat: an `(N,m)`-matrix
    """
    m = y_vec.size // N
    y_mat = np.empty((N, m))
    for i in range(m):
        y_mat[:, i] = y_vec[i * N : (i + 1) * N]
    return cast(np.ndarray, y_mat)


def L2_norm(x: np.ndarray) -> float:
    """Computes the L2 norm of a vector for numba"""
    return sqrt(np.sum(x * x))


def drawArrow(ax, xA, xB, yA, yB, c="k", ls="-"):
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


def display_variable(
    variable: np.ndarray,
    theta_mat: np.ndarray,
    title: str | None = None,
    cmap=None,
    cmap_label: str | None = None,
    path: Path | None = None,
    figsize: tuple[int, int] = (5, 5),
    **kwargs: dict | None,
) -> None:
    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw=kwargs
    )  # subplot_kw=dict(aspect='equal',)
    _ = ax.set_xlabel(r"Risk-aversion $\sigma$")
    _ = ax.set_ylabel(r"Risk location $\delta$")
    _ = ax.set_title(title)
    scatter = ax.scatter(theta_mat[:, 0], theta_mat[:, 1], c=variable, cmap=cmap)
    _ = fig.colorbar(scatter, label=cmap_label)

    if path is not None:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)


def plot_y_range(
    df_first_and_second: pd.DataFrame,
    figsize=(5, 5),
    s=20,
    title=None,
    path=None,
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


def plot_constraints(
    theta_mat: np.ndarray,
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
        theta_mat: the `(N,2)` matrix of type values
        IR_binds: the list of types for which  IR binds
        IC_binds: the list of pairs (i, j) for which i is indifferent between his contract and j's

    Returns:
        nothing. Just plots the constraints.
    """
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
