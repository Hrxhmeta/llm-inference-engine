import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import numpy as np
import torch
from engine.embeddings import embed
from engine.tokenizer import Tokenizer
from engine.weights import load_weights
from transformers import GPT2Model

weights = load_weights()
tok = Tokenizer()

# Reference model: used ONLY here, to produce the answer key.
ref = GPT2Model.from_pretrained(config.WEIGHTS_DIR).eval()

for text in ["Hello world", "Delhi is hot in June.", "def f(x):\n    return x * 2"]:
    ids = tok.encode(text)
    mine = embed(weights, ids)

    with torch.no_grad():
        out = ref(torch.tensor([ids]), output_hidden_states=True)
    # hidden_states[0] is the model's input to layer 1, i.e. the embeddings.
    theirs = out.hidden_states[0][0].numpy()

    diff = np.abs(mine - theirs).max()
    ok = mine.shape == theirs.shape and diff < 1e-5
    print("PASS" if ok else "FAIL", repr(text), mine.shape, f"max diff {diff:.2e}")

print("first 5 values of token 0:", mine[0][:5])