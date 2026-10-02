import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    indices = np.arange(N)
    if shuffle:
        indices = np.random.default_rng(seed).permutation(indices)
    fold = np.array_split(indices, k)
    result = []
    for idx, val in enumerate(fold):
        train = np.concatenate(fold[:idx] + fold[idx + 1:])
        result.append({"train_idx": train, "val_idx": val})
    return result