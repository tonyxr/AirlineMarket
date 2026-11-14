"""Feature metadata; formulas are evaluated from observations only (Eq. 18)."""

from dataclasses import dataclass
from ..core.errors import InformationLeakError


@dataclass(frozen=True)
class FeatureDefinition:
    """Feature units and the minimum required actor information (Eqs. 1–19)."""

    units: str
    needs: str = "own"


FEATURES = {
    "intercept": FeatureDefinition("1"),
    "product_indicator": FeatureDefinition("1"),
    "connection_indicator": FeatureDefinition("1"),
    "scarcity_sum": FeatureDefinition("1/seats"),
    "scarcity_max": FeatureDefinition("1"),
    "time_to_departure": FeatureDefinition("1"),
    "booking_pace": FeatureDefinition("1"),
    "direct_rival_gap": FeatureDefinition("USD", "direct"),
    "min_rival_gap": FeatureDefinition("USD", "direct"),
    "rival_availability": FeatureDefinition("1", "direct"),
    "linked_scarcity_signal": FeatureDefinition("1", "linked"),
    "belief_joint_availability": FeatureDefinition("probability", "belief"),
    "public_disruption": FeatureDefinition("1", "public"),
}


def assert_arm_invariant_basis(pricing_spec, arms) -> None:
    """
    Reject belief/linked-only features in a matched information comparison (Eqs. 1–19).
    """
    for f in (
        *[pair[0] for pair in pricing_spec.fixed_features],
        *pricing_spec.controllable_features,
    ):
        if FEATURES[f.name].needs in ("linked", "belief"):
            raise InformationLeakError(
                f"{f.name} changes the basis across information arms"
            )
