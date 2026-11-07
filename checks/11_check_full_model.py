import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import numpy as np
import torch
from engine.model import forward
from engine.tokenizer import Tokenizer
from engine.weights import load_weights
from transformers import GPT2LMHeadModel

weights = load_weights()
tok = Tokenizer()
ref = GPT2LMHeadModel.from_pretrained(config.WEIGHTS_DIR).eval()

for text in ["Hello world", "Delhi is hot in June.", "The capital of France is"]:
    ids = tok.encode(text)
    mine = forward(weights, ids)
    with torch.no_grad():
        theirs = ref(torch.tensor([ids])).logits[0].numpy()

    diff = np.abs(mine - theirs).max()
    same_top = (mine.argmax(-1) == theirs.argmax(-1)).all()
    print("PASS" if diff < 1e-2 and same_top else "FAIL",
          repr(text), mine.shape, f"max diff {diff:.2e}", "top tokens match:", same_top)

# Preview of Module 4: what does the model think comes next?
print("next token after the last text:", repr(tok.decode([int(mine[-1].argmax())])))