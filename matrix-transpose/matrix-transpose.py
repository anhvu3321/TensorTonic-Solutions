import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    A = np.asarray(A)
    row, col = A.shape

    result = np.ndarray((col, row))

    for i in range(row):
        for j in range(col):
            result[j, i] = A[i, j]

    return result