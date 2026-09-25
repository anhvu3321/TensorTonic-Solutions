import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    x = np.asarray(x, dtype=float)
    x_mean = np.mean(x)
    centered = x - x_mean
    variance = centered.T@centered
    standard_deviation = np.sqrt(variance / (x.shape[0] - 1))
    t = (x_mean - mu0) / (standard_deviation / np.sqrt(x.shape[0]))
    return float(t)