from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
root = PROJECT_ROOT / "data" / "demo_data"

rng = np.random.default_rng(7)

# Synthetic smoke-test data only — NOT a security benchmark.
for cls in ["benign", "malicious"]:
    (root / cls).mkdir(parents=True, exist_ok=True)

for i in range(120):
    # Different distributions make it possible to verify the pipeline quickly.
    if i < 60:
        x = rng.integers(0, 80, 1000, dtype=np.uint8)
        cls = "benign"
    else:
        x = rng.integers(150, 256, 1000, dtype=np.uint8)
        cls = "malicious"
    (root / cls / f"{cls}_{i:03d}.bin").write_bytes(x.tobytes())

print(f"Created {root} with 120 synthetic samples.")
