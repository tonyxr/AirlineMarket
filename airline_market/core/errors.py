"""Domain errors for specification, inference, and bounded enumeration."""


class SpecError(ValueError):
    """
    Invalid declared model input, with its offending specification path (model §10,
    reproducibility conventions).
    """

    def __init__(self, path: str, message: str):
        """
        Bind declared inputs and initialize SpecError bookkeeping (model §10,
        reproducibility conventions).
        """
        self.path = path
        super().__init__(f"{path}: {message}")


class ZeroLikelihoodError(ValueError):
    """
    Observation impossible under the maintained model; never substitute truth (model
    §10, reproducibility conventions).
    """

    pass


class InformationLeakError(ValueError):
    """
    Attempted use of information outside the declared actor arm (model §10,
    reproducibility conventions).
    """

    pass


class KernelLimitError(RuntimeError):
    """
    Unfinished exact enumeration at an explicit computational resource limit (model §10,
    reproducibility conventions).
    """

    pass


class IdentityError(AssertionError):
    """
    Mismatch between a new implementation and its frozen reference (model §10,
    reproducibility conventions).
    """

    pass


class DataValidationError(ValueError):
    """
    Missing or invalid source data required for the declared market scope (model §10,
    reproducibility conventions).
    """

    pass
