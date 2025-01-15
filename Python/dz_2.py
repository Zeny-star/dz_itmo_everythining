import math
def normal_cdf(x, mu, sigma):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2
def angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    l_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    l_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    cos = (ba[0] * bc[0] + ba[1] * bc[1])/(l_ba * l_bc)
    return math.degrees(math.acos(cos))
def triangle_area(a, b, c):
    return 0.5 * abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
def highest_point(alpha, v):
    return (v**2 * math.sin(math.radians(alpha))**2) / (2 * 9.81)
def travel_distance(alpha, v):
    return (v**2 * math.sin(2 * math.radians(alpha))) / 9.81
triangle_area((0, 0), (3, 0), (3, 2))
