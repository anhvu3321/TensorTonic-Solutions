import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    left = np.asarray(y_left, dtype=float)
    right = np.asarray(y_right, dtype=float)
    
    def node_impurity(labels):
        if labels.size == 0:
            return 0.0

        counts = np.unique(labels, return_counts=True)[1]
        probas = counts / labels.size

        return float(1.0 - np.sum(probas**2))

    total = left.size + right.size
    if total == 0:
        return 0.0

    return float((left.size * node_impurity(left) + right.size * node_impurity(right)) / total)
    