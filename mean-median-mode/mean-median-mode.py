from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    x = np.asarray(x, dtype=float)

    x_mean = float(np.mean(x))
    x_median = float(np.median(x))

    counter = Counter(x)

    x_mode = counter.most_common(1)
    x_mode = float(x_mode[0][0])

    return {"mean": x_mean, "median": x_median, "mode": x_mode}