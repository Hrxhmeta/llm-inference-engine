# LLM Inference Engine: Build Log

Goal: run GPT-2 small (124M) in pure NumPy, verify every piece against the
Hugging Face reference, then measure each optimization honestly.
Rule: `torch` and `transformers` are used only in `checks/` and for the download,
never inside `engine/`.

## Setup
- Python 3.12.7 in a venv, numpy, torch (CPU, reference only), transformers.
- Model: openai-community/gpt2. License: MIT (see `notes/license.md`).
- Weights are in `weights/` (git-ignored). `download_weights.py` fetches them.

## Module 1: Tokenizer (done, 10/10 vs reference)
- A token is a chunk of text (word, word piece, or byte) with an integer ID.
  " world" (with the space) and "world" are different tokens, so different IDs.
- GPT-2 maps all 256 bytes to visible characters first (space becomes "Ġ"), so
  vocab.json and merges.txt contain only printable text.
- BPE loop: split into chunks with a regex, then repeatedly merge the adjacent
  pair with the best (lowest) rank in merges.txt until no pair is in the list.
- Test set included emoji, Hindi, tabs, and the empty string.

## Module 2: Weights (done)
- File has 160 tensors, 137,022,720 values. 12 of them are `attn.bias` causal
  masks (a stored buffer, not learned). Skipping them gives 148 tensors and
  exactly 124,439,808 parameters.
- Weight matrices are stored as (input, output), so the math is `x @ W`.

## Module 3: Forward pass (done, each piece checked vs reference)
| Piece | Max diff vs reference |
|---|---|
| Embeddings | 0 |
| LayerNorm | 8.9e-08 |
| MLP (GELU, tanh version) | 2.8e-05 |
| Attention (layer 0) | 7.9e-06 |
| Full 12 layers, logits | 4e-05 to 1e-04, top tokens match |
- Attention in five moves: make Q/K/V, split into 12 heads, score and mask,
  softmax and mix, merge heads and project.
- Residuals: `x = x + attention(ln1(x))`, then `x = x + mlp(ln2(x))`.
- Output uses weight tying: logits = x @ wte.T.

## Module 4: Generation (done)
- Loop: forward, take the last row of logits, sample, append, repeat.
- Sampling: temperature, then top-k, then softmax, then a random draw.
- First baseline (with the float64 bug, see below): 0.60 tok/s, 40 tokens.
- Text is grammatical but factually wrong. That is GPT-2 small, not a bug,
  because the logits match the reference.

## Module 5: KV cache (done)
- Cached logits match the full forward pass (one-by-one and prefill + steps).
- Cache size is about 74 KB per token (12 layers x 2 x 768 x 4 bytes),
  roughly 75 MB at 1024 tokens.
- Bug found through a number that did not add up: cache was 6.4 MB for 45
  tokens instead of 3.3 MB. Cause: `np.sqrt(...)` returns a float64 scalar,
  which promoted float32 arrays to float64. Fix: wrap in `np.float32(...)`.
| | Before fix | After fix |
|---|---|---|
| No cache | 0.63 tok/s | 6.43 tok/s |
| With cache | 0.85 tok/s | 18.52 tok/s |
| Cache speedup | 1.3x | 2.9x |
| Cache size, 45 tokens | 6.4 MB | 3.3 MB |
- Caveat: the 2.9x mixes two savings (no recompute of past tokens, and
  projecting only the last row to 50257 logits). Report both.
- Caveat: 40-token run is short; untested whether the gap grows with longer text.

## Module 6: int8 quantization (done)
- Scope: only the 48 big layer matrices. Biases, LayerNorms, and `wte` stay
  float32.
- Simulated quantization, 46-token sample (noisy, so treat as rough):
| | Perplexity | Top-1 agreement | Size of quantized matrices |
|---|---|---|---|
| float32 | 32.32 | reference | 340 MB |
| one scale per matrix | 39.29 (+21%) | 76% | 85 MB |
| one scale per column | 32.58 (+0.8%) | 96% | 85 MB |
- Hypothesis (consistent with result, not proven): outlier weights stretch a
  single scale and waste precision; per-column scales confine the damage.
- Real int8 in RAM matches the simulation exactly (max diff 0).
- Total weights in RAM: 498 MB float32, 243 MB int8 (2.05x, not 4x, because
  `wte` stays float32).
| | float32 | int8, rebuild matrix each call | int8, scale after matmul |
|---|---|---|---|
| Speed (cached, 40 tokens) | ~15 tok/s | 2.04 tok/s | 3.47 tok/s |
- Int8 is slower in NumPy. Hypothesis (not profiled): decoding one token is
  limited by memory traffic, and the int8 path reads 1 byte, writes a 4-byte
  copy, and reads it again. NumPy has no fast int8 matmul.
- Timings drift about 10% between runs on this laptop. Compare within one run.

## Honest limits so far
- Perplexity numbers come from one 46-token sample. Rerun on 400+ tokens.
- Timings are single runs. Repeat and take the median.
- The int8 speed explanation is a hypothesis until profiled.
- Not yet compared with llama.cpp.


## Module 7a: Full benchmark (3 timed runs after 1 warmup, median)
Prompt "The capital of France is", 40 new tokens, perplexity text 352 tokens.
| config | tok/s median (min-max) | weights | perplexity |
|---|---|---|---|
| float32, no cache | 7.99 (6.22-8.35) | 498 MB | 23.48 |
| float32, KV cache | 17.72 (17.39-17.97) | 498 MB | 23.48 |
| int8, KV cache | 4.55 (4.25-4.81) | 243 MB | 23.40 |
- Cache speedup about 2.2x (2.1x-2.8x given the noise in the no-cache runs).
- Int8 is about 3.9x slower than float32 with the same cache, for 2.05x less RAM.
- Int8 perplexity gap is -0.3%, within noise: no measurable loss on this text.
- Not tested: longer outputs, other texts, int8 without cache.

## Module 7b: Streaming server (done)
- `server.py`: GET /generate streams one JSON message per token (SSE) with
  tokens, tok/s, cache MB, weights MB. Toggles: use_cache, int8.
- Verified: cache off gives cache_mb 0 and about 1.9x slower (5.4 vs 10.0 tok/s,
  20 tokens). int8 gives 243 MB and about 3.6x slower (2.8 vs 10.0 tok/s).
- Cached and uncached text identical on two prompts and two seeds.
- int8 text diverged from float32 at token 13 (sampling sensitivity; perplexity
  is the better quality evidence).
- Limit: tok/s in the server includes prefill in the first token, and one run
  each is not a benchmark.







  ## Hardware
- ## Hardware
- CPU: Intel Core i5, 10th generation, 1.70 GHz (laptop), 8 GB RAM, Windows,
  Python 3.12.7, NumPy 2.x, CPU only.

## Module 7c: Web dashboard
- `static/index.html` served at `/`: prompt box, KV cache and int8 toggles,
  live token streaming, tiles for tokens, tok/s, KV cache MB, weights MB.
- Status: verified: streams live, toggles change the tiles

## Not done
- Batching, speculative decoding, comparison with llama.cpp.
- Longer-output benchmark (does the cache gap grow with length?).
- Profiling the int8 slowdown (the memory-traffic explanation is a hypothesis).

## Reproduce
1. `pip install -r requirements.txt`, then `python download_weights.py`
2. Checks: `python checks\11_check_full_model.py` (and 12, 15)
3. Benchmark: `python benchmarks\02_full_benchmark.py`
4. Dashboard: `python server.py`, open http://127.0.0.1:8000/