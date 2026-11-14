"""Stable string IDs and action value types (Eq. 18)."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class TimingMode(str, Enum):
    """
    Allowed simultaneous or leader/follower posting conventions (model §10,
    reproducibility conventions).
    """

    SIMULTANEOUS = "simultaneous"
    LEADER_FOLLOWER = "leader_follower"


class ArrivalMode(str, Enum):
    """
    Supported single-request Bernoulli and marked-Poisson demand processes (model §10,
    reproducibility conventions).
    """

    SINGLE_REQUEST_BERNOULLI = "single_request_bernoulli"
    POISSON = "poisson"


class ChoiceModel(str, Enum):
    """
    Supported latent-class and sampling-only mixed-logit utility families (model §10,
    reproducibility conventions).
    """

    LATENT_CLASS_MNL = "latent_class_mnl"
    MIXED_LOGIT = "mixed_logit"


class PeriodPhase(str, Enum):
    """
    Explicit position within the cohort pricing/arrival event sequence (model §10,
    reproducibility conventions).
    """

    PRE_DECISION = "pre_decision"
    LEADERS_POSTED = "leaders_posted"
    POSTED = "posted"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class CoefficientAction:
    """
    Immutable vector of additive pricing coefficients (model §10, reproducibility
    conventions).
    """

    theta: tuple[float, ...]


@dataclass(frozen=True)
class PriceAction:
    """
    Explicit product-to-fare action, bypassing coefficient feature selection (model §10,
    reproducibility conventions).
    """

    prices: Mapping[int, float]


@dataclass(frozen=True)
class HoldAction:
    """
    Reuse the carrier's previously committed coefficient vector (model §10,
    reproducibility conventions).
    """

    pass


Action = CoefficientAction | PriceAction | HoldAction
