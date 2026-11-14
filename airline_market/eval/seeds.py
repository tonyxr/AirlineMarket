from dataclasses import dataclass


@dataclass(frozen=True)
class SeedPlan:
    """
    Unique disjoint train, validation and test cohorts plus a planning master seed (Eqs.
    20–22, 29).
    """

    train: tuple
    validation: tuple
    test: tuple
    planning_master: int = 773

    def __post_init__(self):
        """
        Canonicalize declared fields and check the object's construction invariants
        (Eqs. 20–22, 29).
        """
        splits = [tuple(self.train), tuple(self.validation), tuple(self.test)]
        if not splits[0] or len(splits[2]) < 2:
            raise ValueError("nonempty training and at least two test seeds required")
        if any(len(x) != len(set(x)) for x in splits) or len(
            set(sum(splits, ()))
        ) != sum(map(len, splits)):
            raise ValueError("seed splits must be unique and disjoint")

    def episode_index(self, seed):
        """Map a cohort seed to its split-local planning index (Eqs. 20–22, 29)."""
        for split in (self.train, self.validation, self.test):
            if seed in split:
                return tuple(split).index(seed)
        raise ValueError("seed is outside the seed plan")

    @classmethod
    def from_config(cls, data):
        """
        Parse and validate declared train/validation/test seed ranges (Eqs. 20–22, 29).
        """

        def expand(x):
            """
            Enumerate chronological outcomes up to the declared period limit (Eqs.
            20–22, 29).
            """
            if not x:
                return ()
            if isinstance(x, dict):
                return tuple(x["values"])
            return tuple(range(x[0], x[0] + x[1])) if isinstance(x, list) else tuple(x)

        return cls(
            *(expand(data.get(k, [])) for k in ("train", "validation", "test")),
            data.get("planning_master", 773),
        )
