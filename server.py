import json
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

import config
from engine.generate import generate, generate_cached
from engine.kv_cache import KVCache
from engine.quantize import quantize_weights, weights_nbytes
from engine.tokenizer import Tokenizer
from engine.weights import load_weights

app = FastAPI()

# Loaded once at startup and shared by all requests. The int8 copy reuses
# the float32 arrays it doesn't quantize, so it costs only ~85 MB extra.
tok = Tokenizer()
WEIGHTS = {"float32": load_weights()}
WEIGHTS["int8"] = quantize_weights(WEIGHTS["float32"])


def sse(payload):
    """Format one Server-Sent Event."""
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/generate")
def generate_stream(
    prompt: str = Query(..., min_length=1),
    use_cache: bool = True,
    int8: bool = False,
    max_new_tokens: int = Query(config.DEFAULT_MAX_NEW_TOKENS, ge=1, le=config.SERVER_MAX_NEW_TOKENS),
    temperature: float = Query(config.DEFAULT_TEMPERATURE, ge=0, le=2),
    top_k: int = Query(config.DEFAULT_TOP_K, ge=0),
    seed: int = 0,
):
    ids = tok.encode(prompt)
    if len(ids) >= config.MAX_POSITIONS:
        raise HTTPException(400, f"prompt is {len(ids)} tokens; the limit is {config.MAX_POSITIONS - 1}")

    w = WEIGHTS["int8" if int8 else "float32"]
    weights_mb = weights_nbytes(w) / 1e6

    def events():
        cache = KVCache() if use_cache else None
        if use_cache:
            gen = generate_cached(w, ids, max_new_tokens, temperature, top_k, seed, cache=cache)
        else:
            gen = generate(w, ids, max_new_tokens, temperature, top_k, seed)

        new_ids, sent = [], ""
        start = time.time()

        def metrics(done):
            elapsed = time.time() - start
            return {
                "tokens": len(new_ids),
                "tok_per_s": len(new_ids) / elapsed if elapsed else 0,
                "cache_mb": cache.nbytes() / 1e6 if cache else 0,
                "weights_mb": weights_mb,
                "done": done,
            }

        for new_id in gen:
            new_ids.append(new_id)
            text = tok.decode(new_ids)
            # A multi-byte character may be split across tokens: wait for the rest.
            if text.endswith("\ufffd"):
                yield sse({"text": "", **metrics(False)})
                continue
            delta, sent = text[len(sent):], text
            yield sse({"text": delta, **metrics(False)})

        # Flush anything held back, then signal the end.
        yield sse({"text": tok.decode(new_ids)[len(sent):], **metrics(True)})

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)