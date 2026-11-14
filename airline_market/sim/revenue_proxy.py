"""Explicit all-in minus observed-base fare proxy; no statutory tax assertion."""

import numpy as np


def fit_fee_moments(n, sx, sx2, sy, sxy, sy2):
    """Fit nonnegative affine fee moments with rate in[0,1], using exact SSE."""
    if n <= 0 or sx2 <= 0:
        raise ValueError("fee proxy requires valid observed positive fares")
    matrix = np.array([[sx2, sx], [sx, n]], dtype=float)
    vector = np.array([sxy, sy], dtype=float)
    candidates = [(0.0, 0.0), (0.0, max(0.0, sy / n)),
                  (1.0, max(0.0, (sy - sx) / n)),
                  (float(np.clip(sxy / sx2, 0, 1)), 0.0)]
    if np.linalg.det(matrix) > 1e-10:
        rate, flat = np.linalg.solve(matrix, vector)
        if 0 <= rate <= 1 and flat >= 0:
            candidates.append((float(rate), float(flat)))
    def sse(pair):
        """Evaluate the sufficient-statistic squared-error objective."""
        b = np.asarray(pair)
        return float(sy2 - 2 * b @ vector + b @ matrix @ b)
    rate, flat = min(candidates, key=sse)
    return {"rate": rate, "flat": flat, "n": int(n),
            "rmse": float(np.sqrt(max(0.0, sse((rate, flat))) / n)),
            "interpretation": "Q2 observed totalFare-minus-baseFare affine proxy; not verified tax/pass-through or carrier retained revenue."}


def fee_proxy(all_in_fare, fitted):
    """Bound the declared proxy to the nonnegative charged fare, including refunds."""
    if all_in_fare < 0:
        raise ValueError("charged fare must be nonnegative")
    return min(all_in_fare, max(0.0, fitted["rate"] * all_in_fare + fitted["flat"]))


def net_proxy_metrics(world, compiled, fits):
    """Report active-booking contribution under full original-fare refunds only."""
    if compiled.spec.resolution.refund != "full_original_fare":
        raise ValueError("proxy accounting requires full original-fare refunds")
    result = {}
    for i, carrier in enumerate(compiled.spec.carriers):
        if carrier.kind == "outside_network_proxy":
            continue
        state = world.carriers[i]
        gross, fees, service = 0.0, 0.0, 0.0
        for booking in state.bookings:
            product = compiled.spec.products[booking.product]
            fitted = fits[f"{carrier.code}|{len(product.legs)}"]
            gross += booking.fare
            fees += fee_proxy(booking.fare, fitted)
            service += float(compiled.product_cost[booking.product])
        result[carrier.code] = {"active_all_in_fares": gross, "fee_proxy": fees,
            "net_base_fare_proxy": gross - fees, "assumed_service_cost": service,
            "contribution_proxy": gross - fees - service - state.penalties,
            "cash_identity_error": state.cash - (gross - state.penalties),
            "native_objective_uses_this_proxy": False}
    return result
