import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    probabilities = []
    for i in range(k + 1):
        x = math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        probabilities.append(x)

    return {"pmf": float(probabilities[k]), "cdf": float(sum(probabilities))}