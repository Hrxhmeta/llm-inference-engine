import numpy as np

import config


def layer_norm(x, weight, bias):
    """Normalize each row (one token's vector) to mean 0, std 1, then scale and shift."""
    # axis=-1 means "across the 768 numbers of each token", not across tokens.
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normed = (x - mean) / np.sqrt(var + config.LAYER_NORM_EPS)
    return normed * weight + bias



def gelu(x):
    """GPT-2's tanh-approximate GELU, applied to every number separately."""
    inner = np.sqrt(2 / np.pi) * (x + config.GELU_COEFF * x**3)
    return 0.5 * x * (1 + np.tanh(inner))


def mlp(x, weights, layer):
    """Feed-forward block: expand to 4x width, GELU, shrink back."""
    p = f"h.{layer}.mlp."
    hidden = x @ weights[p + "c_fc.weight"] + weights[p + "c_fc.bias"]
    hidden = gelu(hidden)
    return hidden @ weights[p + "c_proj.weight"] + weights[p + "c_proj.bias"]