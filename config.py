from pathlib import Path

# Anchor everything to this file's location so scripts work from any
# working directory (and with spaces in the path).
ROOT = Path(__file__).parent

WEIGHTS_DIR = ROOT / "weights" / "gpt2"

# Hugging Face repo id for GPT-2 small (124M parameters).
HF_REPO_ID = "openai-community/gpt2"

# Only fetch what we need; the repo also holds TF/Flax/ONNX copies
# of the same weights, which would waste gigabytes.
DOWNLOAD_PATTERNS = ["*.json", "merges.txt", "vocab.json", "model.safetensors"]

VOCAB_PATH = WEIGHTS_DIR / "vocab.json"    # token string -> ID
MERGES_PATH = WEIGHTS_DIR / "merges.txt"   # merge rules, highest priority first

# GPT-2's pre-split rule: contractions, words, numbers, symbols, whitespace.
# The optional leading space ( ?) is what keeps " world" as one chunk.
PRETOKEN_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""