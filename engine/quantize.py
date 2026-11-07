import numpy as np

import config

# Avoids dividing by zero if a whole column is exactly 0.
MIN_SCALE = np.float32(1e-12)


def quantize(w, per_column=False):
    """float32 matrix -> (int8 matrix, scale).

    per_column=False: one scale for the whole matrix (scalar).
    per_column=True:  one scale per output column (shape (1, out)).
    """
    if per_column:
        biggest = np.abs(w).max(axis=0, keepdims=True)  # max down each column
    else:
        biggest = np.abs(w).max()
    scale = np.maximum(biggest / config.QUANT_MAX, MIN_SCALE).astype(np.float32)
    q = np.round(w / scale).astype(np.int8)
    return q, scale


def dequantize(q, scale):
    """int8 matrix + scale(s) -> approximate float32 matrix."""
    # astype(float32) first, so the multiply never promotes to float64.
    return q.astype(np.float32) * scale


def fake_quantize_weights(weights, per_column=False):
    """Copy of `weights` where the big matrices went through int8 and back.

    Returns (new_weights, bytes_before, bytes_after) for the quantized tensors.
    """
    new = dict(weights)
    before = after = 0
    for name, w in weights.items():
        if name.endswith(config.QUANT_SUFFIXES):
            q, scale = quantize(w, per_column)
            new[name] = dequantize(q, scale)
            before += w.nbytes
            after += q.nbytes + scale.nbytes
    return new, before, after