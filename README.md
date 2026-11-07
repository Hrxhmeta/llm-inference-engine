# GPT-2 inference engine from scratch (NumPy)

GPT-2 small (124M) running in pure NumPy: tokenizer, weights loader, attention,
MLP, sampling, KV cache, and int8 quantization, with a live browser dashboard.
`torch` and `transformers` are used only in `checks/` (as the answer key) and
for downloading the weights. Nothing in `engine/` imports them.

## Correctness first
Every piece was checked against the Hugging Face reference before the next was built.

| Piece | Max diff vs reference |
|---|---|
| Tokenizer | identical IDs on 10/10 test strings (emoji, Hindi, tabs, empty) |
| Embeddings | 0 |
| LayerNorm | 8.9e-08 |
| MLP | 2.8e-05 |
| Attention (layer 0) | 7.9e-06 |
| Full 12 layers, logits | 4e-05 to 1e-04, top tokens match |

The KV-cached path matches the full forward pass, and cached and uncached
generation give identical text for the same seed (checked on two prompts).

## Results
Hardware: <full CPU model>, 8 GB RAM, Windows, CPU only. Prompt "The capital of
France is", 40 new tokens, median of 3 runs after 1 warmup, perplexity on a
352-token text.

| config | tok/s median (min-max) | weights | perplexity |
|---|---|---|---|
| float32, no cache | 7.99 (6.22-8.35) | 498 MB | 23.48 |
| float32, KV cache | 17.72 (17.39-17.97) | 498 MB | 23.48 |
| int8, KV cache | 4.55 (4.25-4.81) | 243 MB | 23.40 |

- **KV cache:** about 2.2x faster at 40 tokens. The cache costs about 74 KB per token.
  The speedup includes projecting only the last position to the vocabulary,
  which is a separate saving from the cache itself.
- **int8:** 2.05x less RAM (4x on the quantized matrices; the embedding table
  stays float32) with no measurable perplexity loss on this text. But it is
  about 3.9x slower, because NumPy has no int8 matmul and every multiply
  converts to float32. I think the cause is memory traffic, but I did not
  profile it. A C++ kernel multiplying int8 directly, as llama.cpp does, is
  where the speed would come from.
- **Per-matrix vs per-column scales:** one scale per matrix raised perplexity
  by 21% on a 46-token sample; one scale per output column brought it back to
  +0.8%. The outlier-weight explanation fits but is unproven.

## A bug the numbers caught
The first benchmark showed a 1.3x speedup and a cache twice the expected size.
`np.sqrt` returns a float64 scalar, which silently promoted float32 arrays to
float64. Wrapping it in `np.float32(...)` made both paths about 10x faster.

## Run it
```
pip install -r requirements.txt
python download_weights.py
python checks\11_check_full_model.py
python benchmarks\02_full_benchmark.py
python server.py        # then open http://127.0.0.1:8000/
```
If `pip` can't find the torch line, install the CPU build with
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.
Torch and transformers are needed only for `checks/`.

## Limits and not done
- Perplexity comes from one text of one style; speeds are from one laptop and drift about 10% between runs.
- Not done: batching, speculative decoding, comparison with llama.cpp, a longer-output benchmark.
- int8 sampled text can diverge from float32 after a few tokens; perplexity is the better quality measure.
- The server is a single-user demo.

## License
Code: MIT (see `LICENSE`). Model: GPT-2 small, MIT, from `openai-community/gpt2`
(see `notes/license.md`). Weights are downloaded, not redistributed.