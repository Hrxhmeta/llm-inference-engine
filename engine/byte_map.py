# GPT-2 never merges raw bytes. It first maps each byte to a visible
# character, so every token in vocab.json/merges.txt is printable text.

# Bytes that are already visible characters keep their own character.
# Using ord() on the edge characters is clearer than raw numbers.
PRINTABLE_BYTES = (
    list(range(ord("!"), ord("~") + 1))
    + list(range(ord("¡"), ord("¬") + 1))
    + list(range(ord("®"), ord("ÿ") + 1))
)

# Every byte that is NOT visible gets a new character starting here,
# above the 0-255 range, so it can't collide with the printable ones.
FIRST_EXTRA_CODEPOINT = 256


def build_byte_to_char():
    """Return {byte value 0..255: a single visible character}."""
    byte_to_char = {}
    extra = 0
    for b in range(256):
        if b in PRINTABLE_BYTES:
            byte_to_char[b] = chr(b)
        else:
            byte_to_char[b] = chr(FIRST_EXTRA_CODEPOINT + extra)
            extra += 1
    return byte_to_char