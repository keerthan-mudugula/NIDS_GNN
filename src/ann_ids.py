from pathlib import Path
import numpy as np
import torch
from torch import nn

INPUT_SIZE = 1000
N_CLASSES = 2

class ShellcodeANN(nn.Module):
    # Matches the paper's final topology: 1000 -> 30 -> 30 -> 2.
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, N_CLASSES),
        )
        self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def bytes_to_vector(path, offset=0):
    raw = Path(path).read_bytes()
    chunk = raw[offset:offset + INPUT_SIZE]
    x = np.zeros(INPUT_SIZE, dtype=np.float32)
    x[:len(chunk)] = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32)
    # Byte integers are scaled to [0, 1] before entering the ANN.
    return x / 255.0

def load_dataset(root, offset=0):
    root = Path(root)
    xs, ys, paths = [], [], []
    class_map = {"benign": 0, "malicious": 1}

    for name, label in class_map.items():
        folder = root / name
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                try:
                    xs.append(bytes_to_vector(p, offset))
                    ys.append(label)
                    paths.append(str(p))
                except OSError:
                    pass

    if not xs:
        raise RuntimeError(
            f"No files found. Expected {root}/benign and {root}/malicious."
        )

    return np.asarray(xs), np.asarray(ys), paths

def make_model(device="cpu"):
    return ShellcodeANN().to(device)
