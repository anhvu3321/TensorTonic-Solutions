import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X = np.asarray(X, dtype=float)
    c = X - np.mean(X, axis=0)
    cov = (c.T @ c)/(X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    W = eigenvectors[:, :k]
    for j in range(k):
        max_idx = np.argmax(np.abs(W[:, j]))
        if W[max_idx, j] < 0:
            W[:, j] *= -1
    X_proj = c @ W
    return X_proj.tolist()