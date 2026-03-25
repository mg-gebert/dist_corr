from importlib import import_module

from numpy.typing import NDArray
import numpy as np

_rust = import_module("dist_corr._rust")


def distance_correlation(
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    v1_binary: bool = False,
    v2_binary: bool = False,
) -> float:
    """Compute the distance correlation between two vectors.

    Parameters
    ----------
    v1 : NDArray[np.float64]
        First data vector.
    v2 : NDArray[np.float64]
        Second data vector.
    v1_binary : bool, default=False
        Flag indicating whether `v1` is binary (contains only 0.0 or 1.0).
    v2_binary : bool, default=False
        Flag indicating whether `v2` is binary (contains only 0.0 or 1.0).

    Returns
    -------
    float
        Distance correlation in the range [0.0, 1.0].

    Raises
    ------
    TypeError
        If input is not a 1D contiguous `numpy.ndarray` of `float64`.
    ValueError
        If lengths differ, inputs are empty, or a binary-flagged vector contains
        values other than 0.0 or 1.0.
    """
    return _rust.distance_correlation(v1, v2, v1_binary, v2_binary)


def distance_covariance(
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    v1_binary: bool = False,
    v2_binary: bool = False,
) -> float:
    """Compute the distance covariance between two vectors.

    Parameters
    ----------
    v1 : NDArray[np.float64]
        First data vector.
    v2 : NDArray[np.float64]
        Second data vector.
    v1_binary : bool, default=False
        Flag indicating whether `v1` is binary (contains only 0.0 or 1.0).
    v2_binary : bool, default=False
        Flag indicating whether `v2` is binary (contains only 0.0 or 1.0).

    Returns
    -------
    float
        Distance covariance.

    Raises
    ------
    TypeError
        If input is not a 1D contiguous `numpy.ndarray` of `float64`.
    ValueError
        If lengths differ, inputs are empty, or a binary-flagged vector contains
        values other than 0.0 or 1.0.
    """
    return _rust.distance_covariance(v1, v2, v1_binary, v2_binary)


def distance_variance(v: NDArray[np.float64]) -> float:
    """Compute the distance variance of a single vector.

    Parameters
    ----------
    v : NDArray[np.float64]
        Input data vector.

    Returns
    -------
    float
        Distance variance (always non-negative).

    Raises
    ------
    TypeError
        If input is not a 1D contiguous `numpy.ndarray` of `float64`.
    ValueError
        If input is empty.
    """
    return _rust.distance_variance(v)


__all__ = ["distance_correlation", "distance_covariance", "distance_variance"]
