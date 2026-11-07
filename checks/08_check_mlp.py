import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import numpy as np
import torch
from engine.embeddings import embed
from engine.layers import layer_norm, mlp
from engine.tokenizer import Tokenizer
from engine.weights import load_weights
from transformers import GPT2Model

weights = load_weights()
tok = Tokenizer()
ref = GPT2Model.from_pretrained(config.WEIGHTS_DIR).eval()

ids = tok.encode("Delhi is hot in June.")
x = embed(weights, ids)

# The MLP in the real model receives a LayerNorm'd input (ln_2),
# so feed it something realistic instead of the raw embeddings.
normed = layer_norm(x, weights["h.0.ln_2.weight"], weights["h.0.ln_2.bias"])

mine = mlp(normed, weights, layer=0)

with torch.no_grad():
    theirs = ref.h[0].mlp(torch.tensor(normed)).numpy()

diff = np.abs(mine - theirs).max()
print("PASS" if diff < 1e-4 else "FAIL", mine.shape, f"max diff {diff:.2e}")