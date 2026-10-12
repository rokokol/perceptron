from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class MLP:
    """A configurable fully connected classifier built with NumPy"""

    ACTIVATIONS = {"relu", "sigmoid", "tanh"}

    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int] | tuple[int, ...] = (24, 24),
        activations: list[str] | tuple[str, ...] = ("relu", "relu"),
        output_size: int = 2,
        seed: int = 42,
    ) -> None:
        if input_size < 1 or output_size < 2:
            raise ValueError(
                "input_size must be positive and output_size must be at least 2"
            )
        if not hidden_layers or any(size < 1 for size in hidden_layers):
            raise ValueError(
                "At least one hidden layer with a positive width is required"
            )
        if len(activations) != len(hidden_layers):
            raise ValueError("Specify exactly one activation for each hidden layer")
        invalid = set(activations) - self.ACTIVATIONS
        if invalid:
            raise ValueError(f"Unsupported activations: {', '.join(sorted(invalid))}")

        self.input_size = input_size
        self.hidden_layers = list(hidden_layers)
        self.activations = list(activations)
        self.output_size = output_size
        sizes = [input_size, *hidden_layers, output_size]
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1]))
            for i in range(len(sizes) - 1)
        ]
        self.biases = [np.zeros((1, size), dtype=np.float64) for size in sizes[1:]]

    @property
    def parameter_count(self) -> int:
        return sum(
            weight.size + bias.size for weight, bias in zip(self.weights, self.biases)
        )

    @staticmethod
    def _activate(values: FloatArray, name: str) -> FloatArray:
        if name == "relu":
            return np.maximum(0.0, values)
        if name == "sigmoid":
            clipped = np.clip(values, -500.0, 500.0)
            return 1.0 / (1.0 + np.exp(-clipped))
        if name == "tanh":
            return np.tanh(values)
        raise ValueError(f"Unsupported activation: {name}")

    def _forward(self, x: FloatArray) -> tuple[list[FloatArray], list[FloatArray]]:
        activations = [x]
        pre_activations = []
        current = x
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = current @ weight + bias
            pre_activations.append(z)
            if index == len(self.weights) - 1:
                shifted = z - np.max(z, axis=1, keepdims=True)
                exp_values = np.exp(shifted)
                current = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            else:
                current = self._activate(z, self.activations[index])
            activations.append(current)
        return activations, pre_activations

    def predict_proba(self, x: FloatArray) -> FloatArray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError(f"Expected input with shape (n, {self.input_size})")
        return self._forward(values)[0][-1]

    def fit(
        self,
        x: FloatArray,
        y: npt.NDArray[np.int64],
        epochs: int = 200,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        seed: int = 42,
        validation: tuple[FloatArray, npt.NDArray[np.int64]] | None = None,
    ) -> dict[str, list[float]]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if x.ndim != 2 or x.shape != (len(y), self.input_size):
            raise ValueError("Training data dimensions do not match the model")
        if epochs < 1 or learning_rate <= 0 or batch_size < 1:
            raise ValueError(
                "epochs and batch_size must be positive; learning_rate must be positive"
            )

        rng = np.random.default_rng(seed)
        targets = np.eye(self.output_size)[y]
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                indices = order[start : start + batch_size]
                batch_x, batch_y = x[indices], targets[indices]
                layer_outputs, pre_activations = self._forward(batch_x)
                delta = (layer_outputs[-1] - batch_y) / len(indices)
                weight_gradients: list[FloatArray] = []
                bias_gradients: list[FloatArray] = []
                for layer in range(len(self.weights) - 1, -1, -1):
                    weight_gradients.append(layer_outputs[layer].T @ delta)
                    bias_gradients.append(np.sum(delta, axis=0, keepdims=True))
                    if layer:
                        delta = delta @ self.weights[layer].T
                        name = self.activations[layer - 1]
                        z = pre_activations[layer - 1]
                        if name == "relu":
                            delta *= z > 0
                        elif name == "sigmoid":
                            activated = layer_outputs[layer]
                            delta *= activated * (1.0 - activated)
                        else:
                            activated = layer_outputs[layer]
                            delta *= 1.0 - activated**2
                for layer, (weight_gradient, bias_gradient) in enumerate(
                    zip(reversed(weight_gradients), reversed(bias_gradients))
                ):
                    self.weights[layer] -= learning_rate * weight_gradient
                    self.biases[layer] -= learning_rate * bias_gradient

            train_probabilities = self.predict_proba(x)
            history["loss"].append(self._cross_entropy(train_probabilities, targets))
            history["accuracy"].append(
                float(np.mean(np.argmax(train_probabilities, axis=1) == y))
            )
            if validation is not None:
                val_x, val_y = validation
                val_probabilities = self.predict_proba(val_x)
                val_targets = np.eye(self.output_size)[val_y]
                history["val_loss"].append(
                    self._cross_entropy(val_probabilities, val_targets)
                )
                history["val_accuracy"].append(
                    float(np.mean(np.argmax(val_probabilities, axis=1) == val_y))
                )
        return history

    @staticmethod
    def _cross_entropy(probabilities: FloatArray, targets: FloatArray) -> float:
        return float(
            -np.mean(
                np.sum(targets * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
            )
        )

    def save(self, path: str | Path, mean: FloatArray, scale: FloatArray) -> None:
        data = {
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "activations": self.activations,
            "output_size": self.output_size,
            "weights": [weight.tolist() for weight in self.weights],
            "biases": [bias.tolist() for bias in self.biases],
            "mean": np.asarray(mean).tolist(),
            "scale": np.asarray(scale).tolist(),
            "classes": ["B", "M"],
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> tuple[MLP, FloatArray, FloatArray]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            input_size=data["input_size"],
            hidden_layers=data["hidden_layers"],
            activations=data["activations"],
            output_size=data["output_size"],
        )
        model.weights = [
            np.asarray(weight, dtype=np.float64) for weight in data["weights"]
        ]
        model.biases = [np.asarray(bias, dtype=np.float64) for bias in data["biases"]]
        return (
            model,
            np.asarray(data["mean"], dtype=np.float64),
            np.asarray(data["scale"], dtype=np.float64),
        )
