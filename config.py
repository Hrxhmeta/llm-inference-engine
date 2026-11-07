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



WEIGHTS_FILE = WEIGHTS_DIR / "model.safetensors"

# GPT-2 small architecture. These numbers define every tensor shape.
N_LAYERS = 12
N_HEADS = 12
D_MODEL = 768          # width of the hidden vector for each token
D_FF = 4 * D_MODEL     # feed-forward inner width (GPT-2 uses 4x)
VOCAB_SIZE = 50257
MAX_POSITIONS = 1024   # longest sequence GPT-2 was trained on

# Tensors in the file that are not learned weights (the causal mask
# is stored as a buffer). We build the mask ourselves instead.
SKIP_SUFFIXES = (".attn.bias",)

LAYER_NORM_EPS = 1e-5  # matches GPT-2's config; avoids dividing by zero

# Constants of GPT-2's tanh-approximate GELU.
GELU_COEFF = 0.044715