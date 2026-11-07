import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.weights import load_weights

weights = load_weights()

total = 0
for name, arr in weights.items():
    print(f"{name:40} {str(arr.shape):16} {arr.dtype}")
    total += arr.size

print()
print("tensors:", len(weights))
print(f"parameters: {total:,}")
print(f"memory (float32): {total * 4 / 1e6:.0f} MB")