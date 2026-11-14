"""Arm-matched experiments; inherited optimization is not a novelty claim."""


def toy_config(*, scenarios=256, workers=1, legacy=False):
    """
    Build the matched direct/linked/placebo/oracle toy experiment configuration (Eqs.
    28–32).
    """
    methods = [{"name": "fixed", "solver": "fixed_grid", "arm": "direct"}]
    for arm in ("direct", "linked", "unrelated", "oracle"):
        methods.append(
            {
                "name": arm,
                "solver": "legacy_rollout" if legacy else "mpc",
                "arm": arm,
                "params": {
                    "n_scenarios": scenarios,
                    **({} if legacy else {"selection": "split"}),
                },
            }
        )
    return {
        "schema": "airline-experiment-config/1",
        "name": "toy_information_value",
        "spec": "airline_market/spec/library/toy_hub.json",
        "focal": "A",
        "seeds": {
            "train": [100, 64],
            "validation": [5000, 32],
            "test": [10000, 128],
            "planning_master": 773,
        },
        "methods": methods,
        "contrasts": [
            ["linked", "direct"],
            ["unrelated", "direct"],
            ["oracle", "direct"],
            ["direct", "fixed"],
        ],
        "expect_identical": [["unrelated", "direct"]],
        "workers": workers,
        "legacy_tapes": legacy,
    }
