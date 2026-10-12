from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from perceptron import MLP


def load_dataset(path: str | Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    features = []
    labels = []
    with Path(path).open(newline="", encoding="utf-8") as source:
        for line_number, row in enumerate(csv.reader(source), start=1):
            if len(row) not in {31, 32}:
                raise ValueError(f"Line {line_number}: expected 31 or 32 columns, got {len(row)}")
            label_column = 1 if len(row) == 32 else 0
            label = row[label_column].strip()
            if label not in {"B", "M"}:
                raise ValueError(f"Line {line_number}: diagnosis must be B or M")
            try:
                features.append([float(value) for value in row[label_column + 1 :]])
            except ValueError as error:
                raise ValueError(f"Line {line_number}: a feature is not numeric") from error
            labels.append(1 if label == "M" else 0)
    if not features:
        raise ValueError("Dataset is empty")
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def split_indices(y: npt.NDArray[np.int64], validation_fraction: float, seed: int):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    train_parts = []
    validation_parts = []
    for class_id in np.unique(y):
        indices = np.flatnonzero(y == class_id)
        rng.shuffle(indices)
        validation_count = max(1, round(len(indices) * validation_fraction))
        validation_parts.append(indices[:validation_count])
        train_parts.append(indices[validation_count:])
    train = np.concatenate(train_parts)
    validation = np.concatenate(validation_parts)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def split_command(args: argparse.Namespace) -> None:
    x, y = load_dataset(args.dataset)
    train_indices, validation_indices = split_indices(y, args.validation_fraction, args.seed)
    for path, indices in ((args.train, train_indices), (args.validation, validation_indices)):
        with Path(path).open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            for index in indices:
                writer.writerow(["M" if y[index] else "B", *x[index]])
    print(f"Train: {len(train_indices)} rows; validation: {len(validation_indices)} rows")


def plot_history(history: dict[str, list[float]], path: str | Path) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title="Cross-entropy", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, history["accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def train_command(args: argparse.Namespace) -> None:
    x_train, y_train = load_dataset(args.train)
    x_validation, y_validation = load_dataset(args.validation)
    activations = args.activation
    if len(activations) == 1:
        activations *= len(args.layers)
    model = MLP(
        input_size=x_train.shape[1],
        hidden_layers=args.layers,
        activations=activations,
        seed=args.seed,
    )
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train - mean) / scale
    x_validation = (x_validation - mean) / scale
    print(f"Architecture: {x_train.shape[1]} -> {' -> '.join(map(str, args.layers))} -> 2")
    print(f"Activations: {', '.join(activations)} -> softmax; parameters: {model.parameter_count}")
    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        validation=(x_validation, y_validation),
    )
    for epoch in range(args.epochs):
        print(
            f"epoch {epoch + 1:03d}/{args.epochs} "
            f"loss={history['loss'][epoch]:.4f} val_loss={history['val_loss'][epoch]:.4f} "
            f"accuracy={history['accuracy'][epoch]:.4f} "
            f"val_accuracy={history['val_accuracy'][epoch]:.4f}"
        )
    model.save(args.model, mean, scale)
    plot_history(history, args.plot)
    print(f"Model saved to {args.model}; learning curves saved to {args.plot}")


def predict_command(args: argparse.Namespace) -> None:
    model, mean, scale = MLP.load(args.model)
    x, y = load_dataset(args.dataset)
    probabilities = model.predict_proba((x - mean) / scale)
    malignant_probability = np.clip(probabilities[:, 1], 1e-12, 1 - 1e-12)
    binary_loss = -np.mean(
        y * np.log(malignant_probability) + (1 - y) * np.log(1 - malignant_probability)
    )
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predictions == y))
    print(f"Binary cross-entropy: {binary_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    if args.output:
        with Path(args.output).open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(["actual", "predicted", "p_B", "p_M"])
            for actual, prediction, row in zip(y, predictions, probabilities):
                writer.writerow(["M" if actual else "B", "M" if prediction else "B", *row])
        print(f"Predictions saved to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and use a NumPy multilayer perceptron")
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="Make a reproducible stratified train/validation split")
    split.add_argument("--dataset", default="data.csv")
    split.add_argument("--train", default="train.csv")
    split.add_argument("--validation", default="validation.csv")
    split.add_argument("--validation-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(run=split_command)

    train = commands.add_parser("train", help="Train the network and save weights and learning curves")
    train.add_argument("--train", default="train.csv")
    train.add_argument("--validation", default="validation.csv")
    train.add_argument("--model", default="model.json")
    train.add_argument("--plot", default="learning_curves.png")
    train.add_argument("--layers", nargs="+", type=int, default=[24, 24])
    train.add_argument("--activation", nargs="+", choices=sorted(MLP.ACTIVATIONS), default=["relu"])
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(run=train_command)

    predict = commands.add_parser("predict", help="Load a saved model and evaluate a labeled CSV")
    predict.add_argument("--dataset", required=True)
    predict.add_argument("--model", default="model.json")
    predict.add_argument("--output", default="predictions.csv")
    predict.set_defaults(run=predict_command)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.run(arguments)
