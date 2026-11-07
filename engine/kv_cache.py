import numpy as np

import config


class KVCache:
    """Stores every layer's K and V so past tokens are never recomputed."""

    def __init__(self):
        self.keys = [None] * config.N_LAYERS    # each (N_HEADS, tokens, HEAD_DIM)
        self.values = [None] * config.N_LAYERS

    def __len__(self):
        """Number of tokens stored (the same for every layer)."""
        return 0 if self.keys[0] is None else self.keys[0].shape[1]

    def append(self, layer, k, v):
        """Add new K/V for one layer; return that layer's full K and V."""
        if self.keys[layer] is None:
            self.keys[layer], self.values[layer] = k, v
        else:
            # axis=1 is the "tokens" axis; heads stay separate.
            self.keys[layer] = np.concatenate([self.keys[layer], k], axis=1)
            self.values[layer] = np.concatenate([self.values[layer], v], axis=1)
        return self.keys[layer], self.values[layer]

    def nbytes(self):
        """Memory used by the cache (the dashboard will show this)."""
        if self.keys[0] is None:
            return 0
        return sum(k.nbytes + v.nbytes for k, v in zip(self.keys, self.values))