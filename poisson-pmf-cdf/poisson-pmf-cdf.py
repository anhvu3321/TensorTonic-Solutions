import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    exp_lamda = math.exp(-lam)
    probabilities = []

    for i in range(k + 1):
        probabilities.append(float(exp_lamda * (lam ** i) / math.factorial(i)))

    return {"pmf": probabilities[k], "cdf": sum(probabilities)}