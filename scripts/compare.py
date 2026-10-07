import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ann_ids
import gnn_ids

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "demo_data"
DEFAULT_PLOT = PROJECT_ROOT / "comparison_results.png"
DEFAULT_JSON = PROJECT_ROOT / "comparison_results.json"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval(model_name, model, train_loader, Xtr, Xte, yte, epochs, lr, device):
    optimizer = torch.optim.Rprop(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_history = []
    start_train_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        epoch_loss = running_loss / len(Xtr)
        loss_history.append(epoch_loss)

    train_time = time.perf_counter() - start_train_time

    # Latency benchmarking
    model.eval()
    test_tensor = torch.tensor(Xte).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_tensor)

    start_infer_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(30):
            logits = model(test_tensor)
    total_infer_time = time.perf_counter() - start_infer_time
    avg_latency_ms = (total_infer_time / (30 * len(Xte))) * 1000.0
    throughput = (30 * len(Xte)) / total_infer_time

    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(yte, preds)
    prec = precision_score(yte, preds, zero_division=0)
    rec = recall_score(yte, preds, zero_division=0)
    f1 = f1_score(yte, preds, zero_division=0)
    cm = confusion_matrix(yte, preds, labels=[0, 1])

    try:
        auroc = roc_auc_score(yte, probs) if len(np.unique(yte)) == 2 else 1.0
    except ValueError:
        auroc = 1.0

    params = count_parameters(model)

    return {
        "name": model_name,
        "parameters": params,
        "train_time_sec": train_time,
        "avg_latency_ms": avg_latency_ms,
        "throughput_samples_per_sec": throughput,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auroc": float(auroc),
        "confusion_matrix": cm.tolist(),
        "final_loss": float(loss_history[-1]),
        "loss_history": loss_history,
        "preds": preds.tolist(),
        "probs": probs.tolist(),
    }


def plot_comparison(ann_res, gnn_res, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0F172A")

    plt.style.use("dark_background")

    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.set_facecolor("#1E293B")
    epochs = range(1, len(ann_res["loss_history"]) + 1)
    ax1.plot(epochs, ann_res["loss_history"], label=f"ANN (Params: {ann_res['parameters']:,})",
             color="#F43F5E", linewidth=2.2)
    ax1.plot(epochs, gnn_res["loss_history"], label=f"GNN (Params: {gnn_res['parameters']:,})",
             color="#06B6D4", linewidth=2.2, linestyle="--")
    ax1.set_title("Training Loss Convergence (Rprop, lr=0.01)", fontsize=13, fontweight="bold", color="#F8FAFC")
    ax1.set_xlabel("Epoch", fontsize=11, color="#94A3B8")
    ax1.set_ylabel("CrossEntropy Loss", fontsize=11, color="#94A3B8")
    ax1.legend(loc="upper right", framealpha=0.8)
    ax1.grid(True, linestyle=":", alpha=0.4)

    # 2. Metric comparison bar chart
    ax2 = axes[0, 1]
    ax2.set_facecolor("#1E293B")
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUROC"]
    ann_vals = [ann_res["accuracy"], ann_res["precision"], ann_res["recall"], ann_res["f1"], ann_res["auroc"]]
    gnn_vals = [gnn_res["accuracy"], gnn_res["precision"], gnn_res["recall"], gnn_res["f1"], gnn_res["auroc"]]
    x = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x - width / 2, ann_vals, width, label="ANN", color="#F43F5E", alpha=0.9)
    ax2.bar(x + width / 2, gnn_vals, width, label="GNN", color="#06B6D4", alpha=0.9)
    ax2.set_title("Performance Metrics Comparison", fontsize=13, fontweight="bold", color="#F8FAFC")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=10, color="#E2E8F0")
    ax2.set_ylim([0, 1.15])
    for i in range(len(metrics)):
        ax2.text(x[i] - width / 2, ann_vals[i] + 0.02, f"{ann_vals[i]:.2f}", ha="center", fontsize=9, color="#FECDD3")
        ax2.text(x[i] + width / 2, gnn_vals[i] + 0.02, f"{gnn_vals[i]:.2f}", ha="center", fontsize=9, color="#A5F3FC")
    ax2.legend(loc="lower right", framealpha=0.8)
    ax2.grid(True, linestyle=":", alpha=0.4, axis="y")

    # 3. Model Efficiency (Parameters & Latency)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#1E293B")
    models = ["ANN (Shenfield et al.)", "GNN (Sequence Payload)"]
    param_counts = [ann_res["parameters"], gnn_res["parameters"]]
    colors = ["#F43F5E", "#06B6D4"]
    bars = ax3.bar(models, param_counts, color=colors, width=0.5, alpha=0.9)
    ax3.set_title("Trainable Parameter Footprint", fontsize=13, fontweight="bold", color="#F8FAFC")
    ax3.set_ylabel("Total Parameters", fontsize=11, color="#94A3B8")
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 500, f"{h:,} params",
                 ha="center", fontsize=10, fontweight="bold", color="#F8FAFC")
    ax3.grid(True, linestyle=":", alpha=0.4, axis="y")

    # 4. Latency & Throughput Comparison
    ax4 = axes[1, 1]
    ax4.set_facecolor("#1E293B")
    latencies = [ann_res["avg_latency_ms"], gnn_res["avg_latency_ms"]]
    bars2 = ax4.bar(models, latencies, color=colors, width=0.5, alpha=0.9)
    ax4.set_title("Per-Sample Inference Latency (ms)", fontsize=13, fontweight="bold", color="#F8FAFC")
    ax4.set_ylabel("Latency (ms / sample)", fontsize=11, color="#94A3B8")
    for bar in bars2:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.4f} ms",
                 ha="center", fontsize=10, fontweight="bold", color="#F8FAFC")
    ax4.grid(True, linestyle=":", alpha=0.4, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved visual comparison plot -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Empirical Comparison: ANN vs GNN for NIDS")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-plot", default=str(DEFAULT_PLOT))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=================================================================")
    print("        NIDS COMPARISON BENCHMARK: ANN vs GNN")
    print("=================================================================")

    # Load dataset
    X, y, paths = gnn_ids.load_dataset(args.data, args.offset)
    Xtr, Xte, ytr, yte, _, test_paths = train_test_split(
        X, y, paths, test_size=0.2, stratify=y, random_state=args.seed
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dataset      : {args.data} ({len(X)} total samples, {len(Xtr)} train, {len(Xte)} test)")
    print(f"Device       : {device}")
    print(f"Epochs       : {args.epochs}")
    print(f"Learning Rate: {args.lr} (Rprop)")
    print(f"Batch Size   : {args.batch_size}\n")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)),
        batch_size=args.batch_size, shuffle=True
    )

    # 1. Train ANN
    print("[1/2] Training ANN baseline (Shenfield et al., 2018 topology: 1000->30->30->2)...")
    torch.manual_seed(args.seed)
    ann_model = ann_ids.make_model(device)
    ann_res = train_and_eval("ANN", ann_model, train_loader, Xtr, Xte, yte, args.epochs, args.lr, device)

    # Save ANN model
    ann_save_path = PROJECT_ROOT / "models" / "ann_model.pt"
    torch.save({
        "model_type": "ann",
        "state_dict": ann_model.state_dict(),
        "offset": args.offset,
        "input_size": 1000,
        "classes": ["benign", "malicious"],
        "test_files": [
            str(Path(p).resolve().relative_to(Path(args.data).resolve())) for p in test_paths
        ],
        "test_labels": yte.tolist(),
    }, str(ann_save_path))

    # 2. Train GNN
    print("\n[2/2] Training GNN model (Sequence Payload Graph: 1000 nodes, 2 GCN layers, Readout->2)...")
    torch.manual_seed(args.seed)
    gnn_model = gnn_ids.make_model(device, window=1)
    gnn_res = train_and_eval("GNN", gnn_model, train_loader, Xtr, Xte, yte, args.epochs, args.lr, device)

    # Save GNN model
    gnn_save_path = PROJECT_ROOT / "models" / "gnn_model.pt"
    torch.save({
        "model_type": "gnn",
        "state_dict": gnn_model.state_dict(),
        "offset": args.offset,
        "input_size": 1000,
        "classes": ["benign", "malicious"],
        "window": 1,
        "test_files": [
            str(Path(p).resolve().relative_to(Path(args.data).resolve())) for p in test_paths
        ],
        "test_labels": yte.tolist(),
    }, str(gnn_save_path))

    # Print Comparison Table
    param_reduction = (1.0 - (gnn_res["parameters"] / ann_res["parameters"])) * 100.0

    print("\n" + "=" * 65)
    print(f"{'METRIC / ATTRIBUTE':<30} | {'ANN (Baseline)':<15} | {'GNN (Proposed)':<15}")
    print("-" * 65)
    print(f"{'Architecture':<30} | {'MLP 1000-30-30-2':<15} | {'GCN 1000-30-30-2':<15}")
    print(f"{'Trainable Parameters':<30} | {ann_res['parameters']:<15,d} | {gnn_res['parameters']:<15,d} (-{param_reduction:.1f}%)")
    print(f"{'Training Time (sec)':<30} | {ann_res['train_time_sec']:<15.3f} | {gnn_res['train_time_sec']:<15.3f}")
    print(f"{'Accuracy':<30} | {ann_res['accuracy']:<15.4f} | {gnn_res['accuracy']:<15.4f}")
    print(f"{'Precision':<30} | {ann_res['precision']:<15.4f} | {gnn_res['precision']:<15.4f}")
    print(f"{'Recall':<30} | {ann_res['recall']:<15.4f} | {gnn_res['recall']:<15.4f}")
    print(f"{'F1 Score':<30} | {ann_res['f1']:<15.4f} | {gnn_res['f1']:<15.4f}")
    print(f"{'AUROC':<30} | {ann_res['auroc']:<15.4f} | {gnn_res['auroc']:<15.4f}")
    print(f"{'Final Train Loss':<30} | {ann_res['final_loss']:<15.5f} | {gnn_res['final_loss']:<15.5f}")
    print(f"{'Latency per sample (ms)':<30} | {ann_res['avg_latency_ms']:<15.4f} | {gnn_res['avg_latency_ms']:<15.4f}")
    print(f"{'Throughput (samples/sec)':<30} | {ann_res['throughput_samples_per_sec']:<15.1f} | {gnn_res['throughput_samples_per_sec']:<15.1f}")
    print("=" * 65)

    print("\nConfusion Matrix - ANN [benign, malicious]:")
    print(np.array(ann_res["confusion_matrix"]))
    print("Confusion Matrix - GNN [benign, malicious]:")
    print(np.array(gnn_res["confusion_matrix"]))

    # Save JSON results
    results_summary = {
        "dataset": str(args.data),
        "total_samples": len(X),
        "train_samples": len(Xtr),
        "test_samples": len(Xte),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "ann": {k: v for k, v in ann_res.items() if k not in ("loss_history", "preds", "probs")},
        "gnn": {k: v for k, v in gnn_res.items() if k not in ("loss_history", "preds", "probs")},
        "parameter_reduction_percent": param_reduction,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved structured comparison metrics -> {args.output_json}")

    # Plot visual chart
    plot_comparison(ann_res, gnn_res, args.output_plot)


if __name__ == "__main__":
    main()
