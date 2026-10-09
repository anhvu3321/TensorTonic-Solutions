import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    proba = counts / len(y)

    return float(-np.sum(proba * np.log2(proba)))