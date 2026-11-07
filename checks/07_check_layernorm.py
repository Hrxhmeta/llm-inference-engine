import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import numpy as np
import torch
from engine.embeddings import embed
from engine.layers import layer_norm
from engine.tokenizer import Tokenizer
from engine.weights import load_weights
from transformers import GPT2Model

weights = load_weights()
tok = Tokenizer()
ref = GPT2Model.from_pretrained(config.WEIGHTS_DIR).eval()

ids = tok.encode("Delhi is hot in June.")
x = embed(weights, ids)

# Mine: layer 1's first LayerNorm (ln_1).
mine = layer_norm(x, weights["h.0.ln_1.weight"], weights["h.0.ln_1.bias"])

# Reference: the same LayerNorm, applied to the same input.
with torch.no_grad():
    theirs = ref.h[0].ln_1(torch.tensor(x)).numpy()

diff = np.abs(mine - theirs).max()
print("PASS" if diff < 1e-4 else "FAIL", mine.shape, f"max diff {diff:.2e}")