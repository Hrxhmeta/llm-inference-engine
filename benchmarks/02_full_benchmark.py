import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import numpy as np
from engine.generate import generate, generate_cached
from engine.kv_cache import KVCache
from engine.layers import softmax
from engine.model import forward
from engine.quantize import quantize_weights, weights_nbytes
from engine.tokenizer import Tokenizer
from engine.weights import load_weights

weights = load_weights()
qweights = quantize_weights(weights)
tok = Tokenizer()
prompt_ids = tok.encode(config.BENCH_PROMPT)


def median_speed(make_gen):
    """Run once to warm up (not counted), then BENCH_RUNS timed runs."""
    list(make_gen())
    speeds = []
    for _ in range(config.BENCH_RUNS):
        start = time.time()
        n = len(list(make_gen()))
        speeds.append(n / (time.time() - start))
    return statistics.median(speeds), min(speeds), max(speeds)


configs = [
    ("float32, no cache", weights,
     lambda: generate(weights, prompt_ids, config.BENCH_NEW_TOKENS, seed=0)),
    ("float32, KV cache", weights,
     lambda: generate_cached(weights, prompt_ids, config.BENCH_NEW_TOKENS, seed=0)),
    ("int8, KV cache", qweights,
     lambda: generate_cached(qweights, prompt_ids, config.BENCH_NEW_TOKENS, seed=0)),
]

# Perplexity on a longer text (lower is better).
text = config.EVAL_TEXT_PATH.read_text(encoding="utf-8")
eval_ids = tok.encode(text)[:config.EVAL_MAX_TOKENS]
print(f"perplexity text: {len(eval_ids)} tokens\n")


def perplexity(w):
    probs = softmax(forward(w, eval_ids)[:-1])
    ce = -np.log(probs[np.arange(len(eval_ids) - 1), eval_ids[1:]]).mean()
    return float(np.exp(ce))


ppl_f32, ppl_int8 = perplexity(weights), perplexity(qweights)

print(f"{'config':20} {'tok/s median':>12} {'min':>7} {'max':>7} {'weights MB':>11} {'perplexity':>11}")
base = None
for label, w, make_gen in configs:
    med, lo, hi = median_speed(make_gen)
    base = base or med
    ppl = ppl_int8 if w is qweights else ppl_f32
    print(f"{label:20} {med:12.2f} {lo:7.2f} {hi:7.2f} {weights_nbytes(w) / 1e6:11.0f} {ppl:11.2f}"
          f"   ({med / base:.1f}x vs no cache)")

# Cache memory at the benchmark length.
cache = KVCache()
list(generate_cached(weights, prompt_ids, config.BENCH_NEW_TOKENS, seed=0, cache=cache))
print(f"\nKV cache: {cache.nbytes() / 1e6:.1f} MB for {len(cache)} tokens")