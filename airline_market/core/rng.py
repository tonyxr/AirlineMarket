"""Named, stateless child streams independent of execution order."""

import hashlib
import numpy as np


class StreamRegistry:
    """
    Deterministic named random streams keyed without global RNG mutation (model §10,
    reproducibility conventions).
    """

    def __init__(self, root_seed: int, namespace: str):
        """
        Bind declared inputs and initialize StreamRegistry bookkeeping (model §10,
        reproducibility conventions).
        """
        self.root_seed = int(root_seed)
        self.namespace = str(namespace)
        self.ns_hash = int.from_bytes(
            hashlib.sha256(namespace.encode()).digest()[:4], "little"
        )

    def sequence(self, *keys: int) -> np.random.SeedSequence:
        """
        Create a SeedSequence from the named namespace and integer keys (model §10,
        reproducibility conventions).
        """
        return np.random.SeedSequence([self.root_seed, self.ns_hash, *map(int, keys)])

    def generator(self, *keys: int) -> np.random.Generator:
        """
        Create an independent named NumPy random stream (model §10, reproducibility
        conventions).
        """
        return np.random.default_rng(self.sequence(*keys))

    def child_seed(self, *keys: int) -> int:
        """
        Produce a deterministic child seed without consuming another stream (model §10,
        reproducibility conventions).
        """
        return int(self.sequence(*keys).generate_state(1, dtype=np.uint64)[0])
