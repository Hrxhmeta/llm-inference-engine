import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch
from engine.layers import causal_mask, softmax

mask = causal_mask(4)
print("mask (True = blocked):")
print(mask.astype(int))

rng = np.random.default_rng(0)
scores = rng.normal(size=(4, 4)).astype("float32")
masked = np.where(mask, -np.inf, scores)
probs = softmax(masked)
print("probabilities:")
print(probs.round(3))

assert np.allclose(probs.sum(axis=-1), 1), "rows must sum to 1"
assert (probs[mask] == 0).all(), "blocked positions must be exactly 0"
theirs = torch.softmax(torch.tensor(masked), dim=-1).numpy()
assert np.abs(probs - theirs).max() < 1e-6, "differs from torch"

# Stability: the naive formula gives nan here, ours must not.
big = softmax(np.array([[1000.0, 1001.0]]))
print("big scores:", big)
assert not np.isnan(big).any()
print("all checks passed")