import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

with open(config.VOCAB_PATH, encoding="utf-8") as f:
    vocab = json.load(f)

print("vocab size:", len(vocab))
print("Hello  ->", vocab["Hello"])
print("Ġworld ->", vocab["Ġworld"])
print("world  ->", vocab["world"])

# merges.txt: first line is a version header, then one merge per line.
with open(config.MERGES_PATH, encoding="utf-8") as f:
    lines = f.read().split("\n")

print("first 6 lines of merges.txt:")
for line in lines[:6]:
    print("  ", repr(line))