import math

from typing import Dict


def deg_to_rads(deg: float) -> float:
    return math.pi * deg / 180


def rad_to_deg(rad: float) -> float:
    return rad * 180 / math.pi


def is_null_or_undefined(value: float) -> bool:
    return bool(value is None)


def rotate_point(origin_x: float, origin_y: float, x: float, y: float, radians_x: float, radians_y: float) -> Dict[str, float]:

    v = {"x": x - origin_x, "y": y - origin_y}
    vx = (v["x"] * math.cos(radians_x)) - (v["y"] * math.sin(radians_x))
    vy = (v["x"] * math.sin(radians_y)) + (v["y"] * math.cos(radians_y))
    return {"x": vx + origin_x,
            "y": vy + origin_y}

##


def calculate_linear(t: float, p1: float, p2: float) -> float:
    return p1 + t * (p2 - p1)


def calculate_point_quadratic(t: float, p1: float, p2: float, p3: float) -> float:
    one_minus_t = 1 - t
    return (one_minus_t * one_minus_t) * p1 + 2 * one_minus_t * t * p2 + (t * t) * p3


def calculate_point_cubic(t: float, p1: float, p2: float, p3: float, p4: float) -> float:
    t2 = t * t
    t3 = t2 * t
    one_minus_t = 1 - t
    return p1 * pow(one_minus_t, 3) + p2 * 3 * pow(one_minus_t, 2) * t + p3 * 3 * one_minus_t * t2 + p4 * t3


def calculate_coordinates_linear(points, min_distance, round_to_nearest, sample_frequency):
    pts = []
    start_x, start_y = points[0], points[1]
    points = points[2:]

    for i in range(0, len(points), 2):
        end_x, end_y = points[i], points[i + 1]
        t = 0
        last_x = start_x
        last_y = start_y

        while t <= 1.0000000000000007:
            x = calculate_linear(t, start_x, end_x)
            y = calculate_linear(t, start_y, end_y)

            delta_x = x - last_x
            delta_y = y - last_y
            dist = math.sqrt((delta_x * delta_x) + (delta_y * delta_y))

            if abs(dist) > min_distance:
                pts.append(x - (x % round_to_nearest))
                pts.append(y - (y % round_to_nearest))
                last_x = x
                last_y = y

            t += sample_frequency

        start_x = end_x
        start_y = end_y

    return pts
