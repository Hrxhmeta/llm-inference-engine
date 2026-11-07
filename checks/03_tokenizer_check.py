import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.tokenizer import Tokenizer
from transformers import GPT2TokenizerFast

mine = Tokenizer()
ref = GPT2TokenizerFast.from_pretrained(config.WEIGHTS_DIR)

tests = [
    "Hello world",
    "unbelievable",
    "Delhi is hot in June.",
    "I don't think it'll work.",
    "Numbers: 12345 and 3.14",
    "Two  spaces,\nnewline,\ttab",
    "   leading spaces",
    "Emoji: 😀 and Hindi: दिल्ली",
    "def f(x):\n    return x * 2",
    "",
]

failed = 0
for text in tests:
    a, b = mine.encode(text), ref.encode(text)
    ok = a == b and mine.decode(a) == text
    failed += not ok
    print("PASS" if ok else "FAIL", repr(text))
    if not ok:
        print("   mine:", a)
        print("   ref :", b)

print(f"{len(tests) - failed}/{len(tests)} passed")