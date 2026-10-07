import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
import ann_ids
import gnn_ids

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "demo_model.pt"
DEFAULT_DATA = PROJECT_ROOT / "data" / "demo_data"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"
CLASS_NAMES = ("benign", "malicious")


def parse_args():
    ap = argparse.ArgumentParser(description="Predict intrusion with trained GNN or ANN model")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--data", default=str(DEFAULT_DATA))
    source.add_argument("--file", nargs="+")
    ap.add_argument("--offset", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--no-print-files", action="store_true")
    return ap.parse_args()


def load_test_files(checkpoint, data_root):
    test_files = checkpoint.get("test_files")
    test_labels = checkpoint.get("test_labels")
    if not isinstance(test_files, list) or not isinstance(test_labels, list):
        raise RuntimeError(
            "This model does not include its held-out test split. "
            "Retrain it with src/train.py to save the 20% test split."
        )
    if not test_files or len(test_files) != len(test_labels):
        raise RuntimeError("The model checkpoint contains an invalid test split.")

    root = Path(data_root).resolve()
    paths = []
    for relative_path in test_files:
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"Test file path is outside the dataset directory: {relative_path}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"Test file not found: {path}")
        paths.append(path)
    return paths, np.asarray(test_labels, dtype=np.int64)


def infer_labels_from_paths(paths):
    labels = []
    for path in paths:
        label = next(
            (parent.name.lower() for parent in path.parents
             if parent.name.lower() in CLASS_NAMES),
            None,
        )
        labels.append(CLASS_NAMES.index(label) if label else None)
    if all(label is not None for label in labels):
        return np.asarray(labels, dtype=np.int64)
    return None


def write_predictions(output_path, paths, actual, predicted, probabilities):
    with Path(output_path).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "file", "actual", "predicted", "benign_probability",
            "malicious_probability", "correct",
        ])
        for index, path in enumerate(paths):
            actual_label = "" if actual is None else CLASS_NAMES[actual[index]]
            correct = (
                "" if actual is None else int(actual[index] == predicted[index])
            )
            writer.writerow([
                path,
                actual_label,
                CLASS_NAMES[predicted[index]],
                f"{probabilities[index][0]:.6f}",
                f"{probabilities[index][1]:.6f}",
                correct,
            ])


def print_metrics(actual, predicted, probabilities):
    if actual is None:
        return
    print("\n=== Test results ===")
    print("Accuracy :", f"{accuracy_score(actual, predicted):.4f}")
    print("Precision:", f"{precision_score(actual, predicted, zero_division=0):.4f}")
    print("Recall   :", f"{recall_score(actual, predicted, zero_division=0):.4f}")
    if len(np.unique(actual)) == 2:
        print("AUROC    :", f"{roc_auc_score(actual, probabilities[:, 1]):.4f}")
    print("\nConfusion matrix [benign, malicious]:")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))
    print("\n" + classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    ))


def load_model_from_checkpoint(checkpoint, device):
    state_dict = checkpoint["state_dict"]
    model_type = checkpoint.get("model_type")

    # Auto-detect if model_type wasn't explicitly saved
    if model_type is None:
        if any("conv" in k or "norm_adj" in k for k in state_dict.keys()):
            model_type = "gnn"
        else:
            model_type = "ann"

    window = checkpoint.get("window", 1)
    if model_type == "gnn":
        model = gnn_ids.make_model(device, window=window)
    else:
        model = ann_ids.make_model(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model, model_type


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model, map_location=device)
    offset = checkpoint["offset"] if args.offset is None else args.offset

    model, model_type = load_model_from_checkpoint(checkpoint, device)
    print(f"Loaded {model_type.upper()} model from {args.model}")

    if args.file:
        paths = [Path(file).resolve() for file in args.file]
        actual = infer_labels_from_paths(paths)
    else:
        paths, actual = load_test_files(checkpoint, args.data)

    if not paths:
        raise RuntimeError("No files were selected for prediction.")

    # Both models consume raw byte vectors of length 1000
    vectors = np.asarray([gnn_ids.bytes_to_vector(path, offset) for path in paths])
    all_probabilities = []
    with torch.no_grad():
        for start in range(0, len(vectors), args.batch_size):
            batch = torch.as_tensor(
                vectors[start:start + args.batch_size], device=device
            )
            all_probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    probabilities = np.concatenate(all_probabilities)
    predicted = (probabilities[:, 1] >= args.threshold).astype(np.int64)

    if not args.no_print_files:
        for index, path in enumerate(paths):
            print(
                f"{path}: {CLASS_NAMES[predicted[index]]} "
                f"(benign={probabilities[index, 0]:.4f}, "
                f"malicious={probabilities[index, 1]:.4f})"
            )
    print_metrics(actual, predicted, probabilities)
    write_predictions(args.output, paths, actual, predicted, probabilities)
    print(f"\nSaved predictions -> {args.output}")


if __name__ == "__main__":
    main()
