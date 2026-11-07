import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from engine.weights import load_weights

D, FF = config.D_MODEL, config.D_FF

# What GPT-2 small must contain. Note: weight matrices are stored as
# (input size, output size), so the math later is x @ W with no transpose.
expected = {
    "wte.weight": (config.VOCAB_SIZE, D),      # token embeddings
    "wpe.weight": (config.MAX_POSITIONS, D),   # position embeddings
    "ln_f.weight": (D,),
    "ln_f.bias": (D,),
}
for i in range(config.N_LAYERS):
    p = f"h.{i}."
    expected.update({
        p + "ln_1.weight": (D,),
        p + "ln_1.bias": (D,),
        p + "attn.c_attn.weight": (D, 3 * D),  # Q, K, V in one matrix
        p + "attn.c_attn.bias": (3 * D,),
        p + "attn.c_proj.weight": (D, D),
        p + "attn.c_proj.bias": (D,),
        p + "ln_2.weight": (D,),
        p + "ln_2.bias": (D,),
        p + "mlp.c_fc.weight": (D, FF),
        p + "mlp.c_fc.bias": (FF,),
        p + "mlp.c_proj.weight": (FF, D),
        p + "mlp.c_proj.bias": (D,),
    })

weights = load_weights()

problems = 0
for name, shape in expected.items():
    if name not in weights:
        print("MISSING   ", name)
        problems += 1
    elif weights[name].shape != shape:
        print("BAD SHAPE ", name, weights[name].shape, "expected", shape)
        problems += 1
for name in weights:
    if name not in expected:
        print("UNEXPECTED", name)
        problems += 1

total = sum(a.size for a in weights.values())
print(f"tensors: {len(weights)} (expected {len(expected)})")
print(f"parameters: {total:,} (expected 124,439,808)")
print("ALL SHAPES OK" if problems == 0 and total == 124_439_808 else f"{problems} problem(s)")