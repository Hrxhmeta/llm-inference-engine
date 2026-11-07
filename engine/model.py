import config
from engine.embeddings import embed
from engine.layers import layer_norm, transformer_block


def forward(weights, token_ids):
    """Run GPT-2 on token IDs; return (n_tokens, VOCAB_SIZE) logits."""
    x = embed(weights, token_ids)
    for layer in range(config.N_LAYERS):
        x = transformer_block(x, weights, layer)
    x = layer_norm(x, weights["ln_f.weight"], weights["ln_f.bias"])
    # Weight tying: the output matrix is the embedding table, transposed.
    return x @ weights["wte.weight"].T