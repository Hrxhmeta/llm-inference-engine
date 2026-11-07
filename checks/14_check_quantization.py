import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from engine.layers import softmax
from engine.model import forward
from engine.quantize import fake_quantize_weights
from engine.tokenizer import Tokenizer
from engine.weights import load_weights

weights = load_weights()
tok = Tokenizer()

text = ("The city of Delhi has been the capital of several empires over many centuries. "
        "Today it is a busy modern city, and in June the summer heat can be extreme. "
        "People stay indoors in the afternoon and drink a lot of water.")
ids = tok.encode(text)
print("tokens in sample:", len(ids))


def cross_entropy(logits):
    """Average -log(probability given to the true next token). Lower is better."""
    probs = softmax(logits[:-1])
    return float(-np.log(probs[np.arange(len(ids) - 1), ids[1:]]).mean())


full = forward(weights, ids)
ce_full = cross_entropy(full)
print(f"{'float32':12} perplexity {np.exp(ce_full):6.2f}")

for label, per_column in [("per-matrix", False), ("per-column", True)]:
    qweights, before, after = fake_quantize_weights(weights, per_column)
    quant = forward(qweights, ids)
    ce = cross_entropy(quant)
    agree = (full.argmax(-1) == quant.argmax(-1)).mean() * 100
    print(f"{label:12} perplexity {np.exp(ce):6.2f}  top-1 agreement {agree:3.0f}%  "
          f"size {after / 1e6:.0f} MB ({before / after:.2f}x smaller)")