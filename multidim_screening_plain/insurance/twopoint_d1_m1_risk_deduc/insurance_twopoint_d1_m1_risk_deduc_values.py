"""computes the components of the utilities for the two-point insurance model
with one-dimensional types (risk) and straight-deductible contracts
"""

from typing import Any, cast

import numpy as np

from multidim_screening_plain.classes import ScreeningModel
from multidim_screening_plain.utils import (
    check_args,
)


def val_I(
    model: ScreeningModel,
    y: np.ndarray,
    theta: np.ndarray | None = None,
    gr: bool = False,
) -> Any:
    """computes the integral `I`, and its gradient wrt `y` if `gr` is `True`

    Args:
        model: the ScreeningModel
        y:  a `k`-vector of deductible values
        theta: if provided, should be a 1-vector with the risk location of one type;
            then `k` should equal 1
        gr: if `True`, we also return the gradient

    Returns:
        if `theta` is provided, the value of `I(y,theta,s)` for this type and contract;
        otherwise, the values of `I(y,t,s)` for all types `t`
            and for all contracts in `y`,  as an `(N, k)` matrix
        if `gr` is `True` we also return the gradient
    """
    check_args("val_I", y, 1, 1, theta)
    if theta is not None:
        return val_I_1(model, y, theta, gr=gr)
    else:
        return val_I_all(model, y, gr=gr)


def val_I_1(
    model: ScreeningModel,
    y: np.ndarray,
    theta: np.ndarray,
    gr: bool = False,
) -> Any:
    """`val_I` for one type and one contract"""
    delta = theta[0]
    value_A = cast(float, val_A(delta))
    value_BC = val_BC_1(model, y, theta, gr=gr)
    if not gr:
        return value_BC + value_A
    else:
        val, grad = value_BC
        return val + value_A, grad


def val_I_all(model: ScreeningModel, y: np.ndarray, gr: bool = False) -> Any:
    """`val_I` for all types and all contracts in `y`"""
    deltas = model.theta_mat[:, 0]
    values_A = cast(np.ndarray, val_A(deltas))
    values_BC = val_BC_all(model, y, gr=gr)
    if not gr:
        obj = values_BC + values_A.reshape((-1, 1))
        return obj
    else:
        vals, grad = values_BC
        return vals + values_A.reshape((-1, 1)), grad


def val_A(deltas: np.ndarray | float) -> np.ndarray | float:
    """evaluates `A(delta,s,k)`, the probability of no loss
    for all values of `delta` in `deltas`

    Args:
        deltas: a `q`-vector of risk parameters, or a single value

    Returns:
        the values of `A(deltas)` as a `q`-vector, or a single value
    """
    return 1.0 - deltas


def val_BC(
    model: ScreeningModel,
    y: np.ndarray,
    theta: np.ndarray | None = None,
    gr: bool = False,
) -> Any:
    """evaluates the values of `B+C`

    Args:
        model: the ScreeningModel
        y:  a `k`-vector of `k` contracts
        theta: if provided, a 1-vector for one type
        gr: if `True`, we also return the derivatives wrt `y`

    Returns:
        if `theta` is provided, `y` should be  a 1-vector;
            then we return the value of (B + C)(y,theta,s)
            for this contract and this type
        otherwise we return the values of (B + C)(y,theta,s)
            for all types and the contracts in `y` as an `(N, k)` matrix
        If `gr` is `True`, we also return the derivatives wrt `y`.
    """
    check_args("val_BC", y, 1, 1, theta)
    if theta is not None:
        return val_BC_1(model, y, theta, gr=gr)
    else:
        return val_BC_all(model, y, gr=gr)


def val_BC_1(
    model: ScreeningModel, y: np.ndarray, theta: np.ndarray, gr: bool = False
) -> Any:
    """`val_BC` for one type and one contract"""
    params = cast(np.ndarray, model.params)
    sigma = params[0]
    delta, y_0 = theta[0], y[0]
    value_BC = delta * np.exp(sigma * y_0)
    if not gr:
        return value_BC
    else:
        grad = np.zeros(1)
        grad[0] = sigma * val_BC
        return value_BC, grad


def val_BC_all(model: ScreeningModel, y: np.ndarray, gr: bool = False) -> Any:
    params = cast(np.ndarray, model.params)
    sigma = params[0]
    deltas = model.theta_mat[:, 0]
    value_BC = np.outer(deltas, np.exp(sigma * y))
    if not gr:
        return value_BC
    else:
        grad = np.zeros((1, deltas.size, y.size))
        grad[0, :, :] = sigma * value_BC
    return value_BC, grad


def val_I_no_insurance(model: ScreeningModel, theta: np.ndarray | None = None) -> Any:
    params = cast(np.ndarray, model.params)
    sigma, *_, loss = params
    if theta is not None:
        delta = theta[0]
        value_A = val_A(delta)
        value_BC = delta * np.exp(sigma * loss)
        return value_BC + value_A
    else:
        deltas = model.theta_mat[:, 0]
        values_A = cast(np.ndarray, val_A(deltas))
        values_BC = deltas * np.exp(sigma * loss)
        return values_BC + values_A


def val_D(y: np.ndarray, delta: float, loss: float, gr: bool = False) -> Any:
    """evaluates `D`, the actuarial premium

    Args:
        y: a  1-vector with a deductible value
        delta: a risk location parameter
        k: `p_1/delta`
        if `gr` we also return its gradient  wrt `y`

    Returns:
        the value of `D`
            and its gradient  wrt `y` if `gr` is `True`
    """
    val_comp = delta * (loss - y[0])
    if not gr:
        return val_comp
    else:
        grad = np.zeros(1)
        grad[0] = -delta
        return val_comp, grad


def cost_non_insur(model):
    sigma = cast(np.ndarray, model.params)[0]
    return np.log(val_I_no_insurance(model)) / sigma
