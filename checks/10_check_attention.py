import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import numpy as np
import torch
from engine.embeddings import embed
from engine.layers import attention, layer_norm
from engine.tokenizer import Tokenizer
from engine.weights import load_weights
from transformers import GPT2Model

weights = load_weights()
tok = Tokenizer()
ref = GPT2Model.from_pretrained(config.WEIGHTS_DIR).eval()

ids = tok.encode("Delhi is hot in June.")
x = embed(weights, ids)
normed = layer_norm(x, weights["h.0.ln_1.weight"], weights["h.0.ln_1.bias"])

mine = attention(normed, weights, layer=0)

# Capture what the reference attention outputs via a hook, so we don't
# depend on its exact function signature (it changes between versions).
captured = {}
def grab(module, inputs, output):
    captured["out"] = output[0] if isinstance(output, tuple) else output
handle = ref.h[0].attn.register_forward_hook(grab)
with torch.no_grad():
    ref(torch.tensor([ids]))
handle.remove()
theirs = captured["out"][0].numpy()

diff = np.abs(mine - theirs).max()
print("PASS" if diff < 1e-4 else "FAIL", mine.shape, f"max diff {diff:.2e}")