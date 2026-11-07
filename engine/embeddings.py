import numpy as np

import config


def embed(weights, token_ids):
    """Turn token IDs into the (n_tokens, D_MODEL) grid the layers expect."""
    n = len(token_ids)
    # Position table only has MAX_POSITIONS rows; a longer input would
    # silently index out of range, so fail loudly instead.
    if n > config.MAX_POSITIONS:
        raise ValueError(f"{n} tokens, but the model only knows {config.MAX_POSITIONS} positions")

    ids = np.array(token_ids)
    positions = np.arange(n)
    # Fancy indexing: pick one row per ID / per position, then add.
    return weights["wte.weight"][ids] + weights["wpe.weight"][positions]