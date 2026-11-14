"""USD float64-compatible fare operations for Eq. 18."""


def round_fare(x: float) -> float:
    """
    Apply the shared Python two-decimal currency convention (model §10, reproducibility
    conventions).
    """
    return round(float(x), 2)


def clip_fare(x: float, lo: float, hi: float) -> float:
    """
    Apply physical fare bounds before cent rounding (model §10, reproducibility
    conventions).
    """
    return min(float(hi), max(float(lo), float(x)))
