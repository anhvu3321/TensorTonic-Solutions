import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    y = np.asarray(y)
    if not num_classes:
        num_classes = np.max(y) + 1

    onehot = np.zeros((y.size, num_classes), dtype=float)
    onehot[np.arange(y.size), y] = 1.0

    return onehot