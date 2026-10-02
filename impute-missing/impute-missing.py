import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    X = np.asarray(X, dtype=float)
    
    if X.ndim == 1:
        missing = np.isnan(X)
        existed = X[~missing]
        filled = 0.0 if existed.size == 0 else (np.mean(existed) if strategy == "mean" else np.median(existed))
        X[missing] = filled
        return X

    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        missing = np.isnan(col)
        existed = col[~missing]
        filled = 0.0 if existed.size == 0 else (np.mean(existed) if strategy == "mean" else np.median(existed))
        X[missing, col_idx] = filled

    return X