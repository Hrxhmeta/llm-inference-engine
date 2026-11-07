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

def softmax(x):
    """Turn each row of scores into percentages that sum to 1."""
    # Subtracting the row max doesn't change the result, but stops
    # exp() from overflowing on big scores.
    shifted = x - x.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(n):
    """(n, n) grid, True where a token must NOT look (later positions)."""
    # k=1 keeps the diagonal False: a token may look at itself.
    return np.triu(np.ones((n, n), dtype=bool), k=1)



def attention(x, weights, layer):
    """Causal multi-head self-attention for one layer."""
    n = x.shape[0]
    p = f"h.{layer}.attn."

    qkv = x @ weights[p + "c_attn.weight"] + weights[p + "c_attn.bias"]  # (n, 2304)
    q, k, v = np.split(qkv, 3, axis=-1)                                  # each (n, 768)

    def split_heads(t):
        # (n, 768) -> (n, 12, 64) -> (12, n, 64): one slice per head
        return t.reshape(n, config.N_HEADS, config.HEAD_DIM).transpose(1, 0, 2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    # Scale by sqrt(head size) so scores don't blow up the softmax.
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(config.HEAD_DIM)         # (12, n, n)
    scores = np.where(causal_mask(n), -np.inf, scores)                   # mask broadcasts over heads
    probs = softmax(scores)

    mixed = probs @ v                                                    # (12, n, 64)
    merged = mixed.transpose(1, 0, 2).reshape(n, config.D_MODEL)         # (n, 768)
    return merged @ weights[p + "c_proj.weight"] + weights[p + "c_proj.bias"]