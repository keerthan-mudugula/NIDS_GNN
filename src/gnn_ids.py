from pathlib import Path
import numpy as np
import torch
from torch import nn

INPUT_SIZE = 1000
N_CLASSES = 2


def build_sequence_laplacian(num_nodes=INPUT_SIZE, window=1):
    """
    Constructs the normalized adjacency matrix (Kipf & Welling GCN style)
    for a sequential payload graph of `num_nodes` bytes.
    Edges connect consecutive bytes within a local window (default 1: i <-> i+1).
    """
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for w in range(1, window + 1):
        for i in range(num_nodes - w):
            adj[i, i + w] = 1.0
            adj[i + w, i] = 1.0

    # Add self-loops: A_tilde = A + I
    adj_tilde = adj + torch.eye(num_nodes, dtype=torch.float32)

    # Degree matrix D_tilde
    deg = adj_tilde.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Symmetric normalization: D^(-1/2) * A_tilde * D^(-1/2)
    d_mat = torch.diag(deg_inv_sqrt)
    norm_adj = torch.mm(torch.mm(d_mat, adj_tilde), d_mat)
    return norm_adj


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Network (GCN) layer:
    H^{(l+1)} = sigma(A_norm * H^{(l)} * W + b)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._init_xavier()

    def _init_xavier(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, norm_adj):
        # x: (batch_size, num_nodes, in_features)
        # norm_adj: (num_nodes, num_nodes)
        ax = torch.matmul(norm_adj, x)
        return self.linear(ax)


class ShellcodeGNN(nn.Module):
    """
    Graph Neural Network for Shellcode & Payload Intrusion Detection.
    Parallels the ANN paper topology (Shenfield et al., 2018):
      Nodes: 1000 byte positions
      GCN layer 1: 1 -> 30 features
      ReLU
      GCN layer 2: 30 -> 30 features
      ReLU
      Global Readout: Mean pooling over all 1000 node representations (-> 30)
      Classification Head: 30 -> 2 (benign, malicious)

    Weight initialization: Xavier/Glorot uniform (identical to paper ANN).
    """
    def __init__(self, num_nodes=INPUT_SIZE, window=1, hidden_dim=30):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Register normalized adjacency buffer on device
        norm_adj = build_sequence_laplacian(num_nodes=num_nodes, window=window)
        self.register_buffer("norm_adj", norm_adj)

        # Graph convolution layers
        self.conv1 = GCNLayer(1, hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = GCNLayer(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

        # Classification readout head
        self.classifier = nn.Linear(hidden_dim, N_CLASSES)
        self._init_xavier()

    def _init_xavier(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # Support both (B, 1000) and (B, 1000, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # 1st GCN layer + ReLU
        h1 = self.relu1(self.conv1(x, self.norm_adj))

        # 2nd GCN layer + ReLU
        h2 = self.relu2(self.conv2(h1, self.norm_adj))

        # Global graph readout: mean pooling across all 1000 byte nodes
        graph_embed = torch.mean(h2, dim=1)

        # Output logits (benign, malicious)
        return self.classifier(graph_embed)


def bytes_to_vector(path, offset=0):
    raw = Path(path).read_bytes()
    chunk = raw[offset:offset + INPUT_SIZE]
    x = np.zeros(INPUT_SIZE, dtype=np.float32)
    x[:len(chunk)] = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32)
    # Byte integers are scaled to [0, 1] before entering the GNN/ANN.
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


def make_model(device="cpu", window=1):
    return ShellcodeGNN(window=window).to(device)
