import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.generate import generate, generate_cached
from engine.kv_cache import KVCache
from engine.tokenizer import Tokenizer
from engine.weights import load_weights

weights = load_weights()
tok = Tokenizer()
ids = tok.encode("The capital of France is")
N = 40
SEED = 0


def run(label, make_gen):
    start = time.time()
    out = list(make_gen())
    elapsed = time.time() - start
    print(f"{label:12} {len(out)} tokens in {elapsed:6.1f}s = {len(out) / elapsed:5.2f} tok/s")
    return out, elapsed


plain, t_plain = run("no cache", lambda: generate(weights, ids, N, seed=SEED))

cache = KVCache()
cached, t_cached = run("with cache", lambda: generate_cached(weights, ids, N, seed=SEED, cache=cache))

# Same seed + same logits should give the same text. If not, something is off.
print("identical output:", plain == cached)
print(f"speedup: {t_plain / t_cached:.1f}x")
print(f"cache size: {cache.nbytes() / 1e6:.1f} MB for {len(cache)} tokens")