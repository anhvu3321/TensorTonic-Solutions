def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    result = []
    for point in points:
        ans = 0
        best = sum((a - b) ** 2 for a, b in zip(point, centroids[0]))

        for i in range(1, len(centroids)):
            dist = sum((a - b) ** 2 for a, b in zip(point, centroids[i]))
            if dist < best:
                best = dist
                ans = i

        result.append(ans)

    return result