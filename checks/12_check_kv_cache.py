import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from engine.cached_model import forward_cached
from engine.kv_cache import KVCache
from engine.model import forward
from engine.tokenizer import Tokenizer
from engine.weights import load_weights

weights = load_weights()
tok = Tokenizer()

ids = tok.encode("The capital of France is a big city in Europe.")
full = forward(weights, ids)   # trusted: logits for every position


def compare(label, got, pos):
    diff = np.abs(got - full[pos]).max()
    ok = diff < 1e-3 and got.argmax() == full[pos].argmax()
    print("PASS" if ok else "FAIL", label, f"position {pos}", f"max diff {diff:.2e}")
    return ok


all_ok = True

# Test 1: feed tokens one at a time (pure decode, cache starts empty).
cache = KVCache()
for pos, t in enumerate(ids):
    all_ok &= compare("one-by-one", forward_cached(weights, [t], cache), pos)

# Test 2: prefill the first 5 tokens in one go, then continue one by one.
cache = KVCache()
all_ok &= compare("prefill   ", forward_cached(weights, ids[:5], cache), 4)
for pos in range(5, len(ids)):
    all_ok &= compare("then step ", forward_cached(weights, [ids[pos]], cache), pos)

print("cache length:", len(cache), "tokens =", len(ids))
print(f"cache memory: {cache.nbytes() / 1e6:.2f} MB")
print("ALL PASS" if all_ok else "SOME FAILED")