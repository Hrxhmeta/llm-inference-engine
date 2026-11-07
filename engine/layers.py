import numpy as np

import config


def layer_norm(x, weight, bias):
    """Normalize each row (one token's vector) to mean 0, std 1, then scale and shift."""
    # axis=-1 means "across the 768 numbers of each token", not across tokens.
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + config.LAYER_NORM_EPS)
    return normed * weight + bias