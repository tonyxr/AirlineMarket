"""Controller registry; optional learners are imported only by their commands."""

from .fixed import FixedCoefficientController
from .rule import ThresholdLinearRuleController
from .external import ExternalController


def make_controller(assignment):
    """
    Construct a declared controller; execution receives scoped observations (Eqs.
    18–19).
    """
    key = assignment.controller
    p = assignment.params
    if key == "external":
        return ExternalController()
    if key == "fixed":
        return FixedCoefficientController(p.get("theta", ()))
    if key == "threshold_linear_rule":
        return ThresholdLinearRuleController(p.get("rules", {}))
    if key == "schedule":
        from .schedule import TimeScheduleController

        return TimeScheduleController(p["thetas_by_period"])
    if key == "matching":
        from .matching import CompetitorMatchingController

        return CompetitorMatchingController(
            **{k: v for k, v in p.items() if k != "arm"}
        )
    if key == "bid_price_dlp":
        from .bidprice import DLPBidPriceController

        return DLPBidPriceController(**{k: v for k, v in p.items() if k != "arm"})
    if key == "emsr_b":
        from .emsr import EMSRbController

        return EMSRbController(**{k: v for k, v in p.items() if k != "arm"})
    if key == "myopic_br":
        from .adaptive import MyopicBestResponseController

        return MyopicBestResponseController(p.get("theta_grid"), p.get("rival_counts"))
    raise ValueError(f"unknown controller {key}")
