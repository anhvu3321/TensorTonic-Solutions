import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    X = np.asarray(X, dtype=float)
    centered = X - np.mean(X, axis=0)

    covariance = centered.T @ centered / (X.shape[0] - 1)
    deviation = np.sqrt(np.diag(covariance))
    matrix = np.outer(deviation, deviation)

    return covariance / matrix