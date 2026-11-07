import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.cached_model import forward_cached
from engine.kv_cache import KVCache
from engine.tokenizer import Tokenizer
from engine.weights import load_weights

weights = load_weights()
cache = KVCache()
forward_cached(weights, Tokenizer().encode("Hello world"), cache)

print("weights dtype:", weights["wte.weight"].dtype)
for layer in (0, 1, 11):
    print(f"layer {layer} cache dtype:", cache.keys[layer].dtype)