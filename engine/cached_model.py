import numpy as np

import config
from engine.embeddings import embed
from engine.layers import layer_norm, mlp, softmax


def cached_attention(x, weights, layer, cache, past):
    """Attention for n new tokens, given `past` tokens already in the cache."""
    n = x.shape[0]
    p = f"h.{layer}.attn."

    qkv = x @ weights[p + "c_attn.weight"] + weights[p + "c_attn.bias"]   # (n, 2304)
    q, k, v = np.split(qkv, 3, axis=-1)

    def split_heads(t):
        return t.reshape(n, config.N_HEADS, config.HEAD_DIM).transpose(1, 0, 2)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)   # (12, n, 64)

    # Store the new K/V and get back everything: old + new.
    k, v = cache.append(layer, k, v)                           # (12, past+n, 64)
    total = past + n

    scores = q @ k.transpose(0, 2, 1) / np.sqrt(config.HEAD_DIM)   # (12, n, total)
    # New token i sits at absolute position past+i, so it may see columns
    # 0..past+i. Everything from column past+i+1 on is blocked.
    blocked = np.triu(np.ones((n, total), dtype=bool), k=past + 1)
    scores = np.where(blocked, -np.inf, scores)

    mixed = softmax(scores) @ v                                # (12, n, 64)
    merged = mixed.transpose(1, 0, 2).reshape(n, config.D_MODEL)
    return merged @ weights[p + "c_proj.weight"] + weights[p + "c_proj.bias"]


def forward_cached(weights, token_ids, cache):
    """Process NEW tokens only; return the logits of the last one, shape (VOCAB_SIZE,)."""
    past = len(cache)  # read once, before any layer appends to the cache
    x = embed(weights, token_ids, start=past)
    for layer in range(config.N_LAYERS):
        p = f"h.{layer}."
        normed = layer_norm(x, weights[p + "ln_1.weight"], weights[p + "ln_1.bias"])
        x = x + cached_attention(normed, weights, layer, cache, past)
        x = x + mlp(layer_norm(x, weights[p + "ln_2.weight"], weights[p + "ln_2.bias"]), weights, layer)
    # LayerNorm works per row, so keeping only the last row first is safe.
    x = layer_norm(x[-1:], weights["ln_f.weight"], weights["ln_f.bias"])
    return (x @ weights["wte.weight"].T)[0]