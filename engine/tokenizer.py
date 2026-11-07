import json

import regex

import config
from engine.byte_map import build_byte_to_char


class Tokenizer:
    def __init__(self):
        with open(config.VOCAB_PATH, encoding="utf-8") as f:
            self.vocab = json.load(f)  # token string -> ID
        self.id_to_token = {i: t for t, i in self.vocab.items()}

        with open(config.MERGES_PATH, encoding="utf-8") as f:
            lines = f.read().split("\n")
        # Skip the '#version' header and the empty last line.
        merges = [tuple(line.split(" ")) for line in lines[1:] if line]
        # Line number = priority. Lower rank means merge earlier.
        self.ranks = {pair: i for i, pair in enumerate(merges)}

        self.byte_to_char = build_byte_to_char()
        self.char_to_byte = {c: b for b, c in self.byte_to_char.items()}
        self.pattern = regex.compile(config.PRETOKEN_PATTERN)

    def _bpe(self, word):
        """Merge a chunk (already byte-mapped) into the fewest known pieces."""
        parts = list(word)
        while len(parts) > 1:
            pairs = set(zip(parts, parts[1:]))
            # Pick the pair that appears earliest in merges.txt.
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break  # nothing left to merge
            merged = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return parts

    def encode(self, text):
        ids = []
        for chunk in self.pattern.findall(text):
            mapped = "".join(self.byte_to_char[b] for b in chunk.encode("utf-8"))
            for piece in self._bpe(mapped):
                ids.append(self.vocab[piece])
        return ids

    def decode(self, ids):
        text = "".join(self.id_to_token[i] for i in ids)
        raw = bytes(self.char_to_byte[c] for c in text)
        # errors="replace": a half-finished multi-byte character shows as �
        # instead of crashing (this can happen mid-generation later).
        return raw.decode("utf-8", errors="replace")