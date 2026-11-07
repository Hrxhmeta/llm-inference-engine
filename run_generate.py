import time

import config
from engine.generate import generate
from engine.tokenizer import Tokenizer
from engine.weights import load_weights

weights = load_weights()
tok = Tokenizer()

prompt = "The capital of France is"
ids = tok.encode(prompt)

print(prompt, end="", flush=True)
start = time.time()
count = 0
for new_id in generate(weights, ids):
    print(tok.decode([new_id]), end="", flush=True)
    count += 1
elapsed = time.time() - start

print(f"\n\n{count} tokens in {elapsed:.1f}s = {count / elapsed:.2f} tokens/sec")