from dataclasses import dataclass
import numpy as np


@dataclass
class Backtest:
    """
    Pre/post-event predictor evaluation with passenger-weighted scores (Eqs. 20–22, 29).
    """

    event: str
    pre_period: object
    post_period: object
    spec_builder: object
    predictor: object

    def run(self, pre_data, post_data):
        """
        Execute the declared paired experiment or backtest and return its outcomes (Eqs.
        20–22, 29).
        """
        spec = self.spec_builder(pre_data)
        predictions = self.predictor(spec, self.post_period)
        observed = np.asarray(post_data["value"])
        point = np.asarray(predictions["mean"])
        weights = np.asarray(post_data.get("passengers", np.ones(len(observed))))
        return {
            "event": self.event,
            "weighted_mse": float(np.average((point - observed) ** 2, weights=weights)),
            "coverage": float(
                np.average(
                    (observed >= predictions["low"])
                    & (observed <= predictions["high"]),
                    weights=weights,
                )
            ),
            "predictions": predictions,
        }
