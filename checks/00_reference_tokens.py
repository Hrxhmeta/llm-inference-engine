import sys
from pathlib import Path

# Make `import config` work when running from the checks/ folder.
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from transformers import GPT2TokenizerFast

tok = GPT2TokenizerFast.from_pretrained(config.WEIGHTS_DIR)

for text in ["Hello world", "world", " world", "unbelievable", "Delhi is hot in June."]:
    ids = tok.encode(text)
    pieces = [tok.decode([i]) for i in ids]
    print(f"{text!r:28} -> {ids}  {pieces}")