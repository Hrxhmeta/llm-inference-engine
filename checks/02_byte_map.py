import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.byte_map import build_byte_to_char

byte_to_char = build_byte_to_char()

print("space   ->", byte_to_char[ord(" ")])
print("newline ->", byte_to_char[ord("\n")])
print("A       ->", byte_to_char[ord("A")])

# Test 1: 256 bytes must map to 256 different characters.
assert len(set(byte_to_char.values())) == 256

# Test 2: every mapped character must exist as a token in vocab.json.
with open(config.VOCAB_PATH, encoding="utf-8") as f:
    vocab = json.load(f)
missing = [c for c in byte_to_char.values() if c not in vocab]
assert not missing, f"not in vocab: {missing}"

# Test 3: turn real text into the mapped form and look it up.
text = "Hello world"
mapped = "".join(byte_to_char[b] for b in text.encode("utf-8"))
print(mapped, "->", vocab["Ġworld"])
print("all checks passed")