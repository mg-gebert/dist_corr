from numpy.typing import NDArray
import numpy as np

def distance_correlation(
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    v1_binary: bool = False,
    v2_binary: bool = False,
) -> float: ...
def distance_covariance(
    v1: NDArray[np.float64],
    v2: NDArray[np.float64],
    v1_binary: bool = False,
    v2_binary: bool = False,
) -> float: ...
def distance_variance(v: NDArray[np.float64]) -> float: ...
