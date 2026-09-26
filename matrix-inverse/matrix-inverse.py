import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    A = np.asarray(A, dtype=float)
    size = A.shape[0]
    concatenated = np.concatenate((A.copy(), np.eye(size)), axis=1)

    for col in range(size):
        pivot = col + np.argmax(np.abs(concatenated[col:, col]))
        if abs(concatenated[pivot, col]) < 1e-12:
            return None

        concatenated[[col, pivot]] = concatenated[[pivot, col]]
        concatenated[col] /= concatenated[col, col]

        for row in range(size):
            if row != col:
                concatenated[row] -= concatenated[row, col] * concatenated[col]

    return concatenated[:, size:]