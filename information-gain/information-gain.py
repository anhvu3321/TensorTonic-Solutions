import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    y = np.asarray(y, dtype=float)
    mask = np.asarray(split_mask, dtype=bool)

    def calc_entropy(values):
        if values.size == 0:
            return 0.0
        counts = np.unique(values, return_counts=True)[1]
        probas = counts / values.size
        return float(-np.sum(probas * np.log2(probas)))

    left = y[mask]
    right = y[~mask]

    if left.size == 0 or right.size == 0:
        return 0.0

    return float(calc_entropy(y) - (left.size * calc_entropy(left) + right.size * calc_entropy(right)) / y.size)