import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from engine.generate import generate_cached
from engine.model import forward
from engine.quantize import fake_quantize_weights, quantize_weights, weights_nbytes
from engine.tokenizer import Tokenizer
from engine.weights import load_weights

weights = load_weights()
tok = Tokenizer()
ids = tok.encode("The capital of France is")

fake, _, _ = fake_quantize_weights(weights, per_column=True)
real = quantize_weights(weights)

# Test 1: real int8 must match the simulation (same rounding, same math).
diff = np.abs(forward(real, ids) - forward(fake, ids)).max()
print("PASS" if diff < 1e-3 else "FAIL", f"real vs simulated logits, max diff {diff:.2e}")

# Test 2: memory actually held by the weights.
m_f, m_q = weights_nbytes(weights), weights_nbytes(real)
print(f"weights in RAM: float32 {m_f / 1e6:.0f} MB, int8 {m_q / 1e6:.0f} MB "
      f"({m_f / m_q:.2f}x smaller)")

# Test 3: speed, 40 tokens with the KV cache.
for label, w in [("float32", weights), ("int8", real)]:
    start = time.time()
    out = list(generate_cached(w, ids, 40, seed=0))
    t = time.time() - start
    print(f"{label:8} {len(out)} tokens in {t:5.1f}s = {len(out) / t:5.2f} tok/s")