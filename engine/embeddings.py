import numpy as np

import config


def embed(weights, token_ids, start=0):
    """Turn token IDs into the (n_tokens, D_MODEL) grid the layers expect.

    start: position of the first token. 0 for a fresh prompt; with a KV
    cache it is the number of tokens already processed.
    """
    n = len(token_ids)
    # Position table only has MAX_POSITIONS rows; fail loudly, not silently.
    if start + n > config.MAX_POSITIONS:
        raise ValueError(f"{start + n} tokens, but the model only knows {config.MAX_POSITIONS} positions")

    ids = np.array(token_ids)
    positions = np.arange(start, start + n)
    return weights["wte.weight"][ids] + weights["wpe.weight"][positions]