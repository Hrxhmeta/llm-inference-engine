import numpy as np

import config
from engine.model import forward
from engine.sampling import sample_next


def generate(weights, token_ids, max_new_tokens=config.DEFAULT_MAX_NEW_TOKENS,
             temperature=config.DEFAULT_TEMPERATURE, top_k=config.DEFAULT_TOP_K, seed=0):
    """Yield new token IDs one at a time (a generator, so the UI can stream them)."""
    rng = np.random.default_rng(seed)  # fixed seed = repeatable output
    ids = list(token_ids)

    for _ in range(max_new_tokens):
        if len(ids) >= config.MAX_POSITIONS:
            break  # the model has no position vectors beyond this
        logits = forward(weights, ids)[-1]  # only the last position predicts the next token
        next_id = sample_next(logits, temperature, top_k, rng)
        if next_id == config.END_OF_TEXT_ID:
            break
        ids.append(next_id)
        yield next_id