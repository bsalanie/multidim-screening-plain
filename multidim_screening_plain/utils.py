from math import exp, sqrt
from pathlib import Path
from typing import cast

import numpy as np
from bs_python_utils.bsutils import bs_error_abort

results_dir = Path.cwd() / "results"
plots_dir = Path.cwd() / "plots"

INV_SQRT_2 = np.sqrt(0.5)
INV_SQRT_2PI = 1.0 / np.sqrt(2 * np.pi)


def parse_string(s: str, d: int, ch: str, msg: str, component_type: str) -> np.ndarray:
    bits = s.split(ch)
    if component_type == "int":
        components = np.array([int(b) for b in bits])
    elif component_type == "float":
        components = np.array([float(b) for b in bits])
    elif component_type == "str":
        components = np.array(bits)
    else:
        bs_error_abort(f"Unknown component type {component_type}")
    if components.size != d:
        bs_error_abort(
            f"Wrong number of dimensions for {msg}: {components.size} but we want"
            f" {d} dimensions."
        )
    return cast(np.ndarray, components)


def make_grid(theta: list[np.ndarray]) -> np.ndarray:
    """
    creates the $d$-dimensional grid of types

    Args:
        theta: a list of $d$ arrays

    Returns:
        an $(N,d)$-matrix, the grid of types
    """
    ltheta = np.meshgrid(*theta)
    d = len(ltheta)
    for j in range(d):
        ltheta[j] = ltheta[j].flatten()
    N = ltheta[0].size
    theta_mat = np.zeros((N, d))
    for j in range(d):
        theta_mat[:, j] = ltheta[j]
    return theta_mat


def split_y(y: np.ndarray, m: int) -> list[np.ndarray]:
    """Split y into `m` chunks of equal length"""
    N = y.size // m
    y_chunks = [] * m
    for j in range(m):
        y_chunks.append(y[j * N : (j + 1) * N])
    return y_chunks


def check_args(
    function_name: str, y: np.ndarray, d: int, m: int, theta: np.ndarray | None
) -> None:
    """check the arguments passed"""
    if theta is not None:
        if theta.shape != (d,):
            bs_error_abort(
                f"{function_name}: If theta is given it should be a {d}-vector, not shape"
                f" {theta.shape}"
            )
        if y.shape != (m,):
            bs_error_abort(
                f"{function_name}: If theta is given, y should be a {m}-vector, not shape"
                f" {y.shape}"
            )
    else:
        if y.ndim != 1:
            bs_error_abort(
                f"{function_name}: y should be a vector, not {y.ndim}-dimensional"
            )
        if y.size % m != 0:
            bs_error_abort(
                f"{function_name}: y should have a number of elements multiple of {m}, not"
                f" {y.size}"
            )


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


def bs_norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Fast standard normal CDF based on Numerical Recipes.

    This function is accurate to 6 decimal places.
    """
    is_array = isinstance(x, np.ndarray)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    z = x * INV_SQRT_2
    if is_array:
        sign_z = np.sign(z)
        abs_z = np.abs(z)
        ez2 = np.exp(-z * z)
    else:
        sign_z = 1 if z > 0 else -1
        abs_z = abs(z)
        ez2 = exp(-z * z)
    t = 1.0 / (1.0 + p * abs_z)
    poly_t = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1
    y = 1.0 - poly_t * t * ez2
    res = (1.0 + sign_z * y) / 2.0

    return cast(np.ndarray, res) if is_array else cast(float, res)


def bs_norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    """Fast standard normal PDF, the derivative of `bs_norm_cdf`.

    This function is accurate to 6 decimal places.
    """
    is_array = isinstance(x, np.ndarray)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    z = x * INV_SQRT_2
    if is_array:
        sign_z = np.sign(z)
        abs_z = np.abs(z)
        ez2 = np.exp(-z * z)
    else:
        sign_z = 1 if z > 0 else -1
        abs_z = abs(z)
        ez2 = exp(-z * z)

    t = 1.0 / (1.0 + p * abs_z)
    dt_dz = -t * t * p * sign_z
    poly_t = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1
    1.0 - poly_t * t * np.exp(-z * z)
    dpoly_t = ((4.0 * a5 * t + 3.0 * a4) * t + 2.0 * a3) * t + a2
    dy_dz = -((poly_t + t * dpoly_t) * dt_dz - 2.0 * z * poly_t * t) * ez2

    res = sign_z * dy_dz * INV_SQRT_2 / 2.0

    return cast(np.ndarray, res) if is_array else cast(float, res)


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
