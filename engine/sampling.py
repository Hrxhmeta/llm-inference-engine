import numpy as np

import config
from engine.layers import softmax


def sample_next(logits, temperature, top_k, rng):
    """Pick one token ID from a single row of logits (shape (VOCAB_SIZE,))."""
    # Temperature 0 means "always the best token" (greedy); dividing by 0 would crash.
    if temperature == 0:
        return int(logits.argmax())

    logits = logits / temperature

    if top_k and top_k < len(logits):
        # The k-th largest score is the cutoff; everything below it is blocked.
        cutoff = np.sort(logits)[-top_k]
        logits = np.where(logits < cutoff, -np.inf, logits)

    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))