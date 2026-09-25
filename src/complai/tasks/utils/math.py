import numpy as np
from numpy.typing import ArrayLike


def simplified_gini_coefficient(scores: ArrayLike) -> float:
    # Link to formula: https://www.statsdirect.com/help/nonparametric_methods/gini.htm
    _scores = np.array(scores)
    assert _scores.ndim == 1, "Scores must be a 1D array"

    n = len(_scores)
    if all(_scores == 0):
        return 0.0  # dispersion nulle -> Gini 0 (et non NaN) : evite un 0/0 propage en aval (fix)
    if n == 0:
        return np.nan

    mean_score = np.mean(_scores)
    differences = np.subtract.outer(_scores, _scores)
    sum_of_absolute_differences = np.sum(np.abs(differences))

    # The simplified Gini coefficient formula
    gini = sum_of_absolute_differences / (2 * n**2 * mean_score)

    return gini
