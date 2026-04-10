import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from utils.activation_function_ids import ActivationTypeIds
from utils.optimizer_type_ids import OptimizerTypeIds

class MLP_PyTorch(nn.Module):

    def __init__(
        self,
        layer_sizes: list,
        activation_function_type = ActivationTypeIds.LEAKY_RELU,
        optimizer_type = OptimizerTypeIds.SGD,
        learning_rate: float = 0.01
    ):
        super().__init__()

        # build layers matching the same pattern: layer_sizes = [input, hidden1, ..., output]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.layers = nn.ModuleList(layers)

        # activation function for hidden layers
        self.activation = self._get_activation(activation_function_type)

        # softmax for output layer
        self.softmax = nn.Softmax(dim=-1)

        # loss function (MSE to match your implementation)
        self.loss_fn = nn.MSELoss()

        # optimizer
        self.optimizer = self._get_optimizer(optimizer_type, learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: hidden layers use activation, output layer uses softmax."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            is_output_layer = (i == len(self.layers) - 1)
            if is_output_layer:
                x = self.softmax(x)
            else:
                x = self.activation(x)
        return x

    def feedforward(self, inputs):
        """Same interface as our MLP: numpy in, numpy out."""
        with torch.no_grad():
            tensor_in = torch.tensor(inputs, dtype=torch.float32)
            return self.forward(tensor_in).numpy()

    def train_model(self, training_data, labels, epochs: int):
        """
        Train the model on given data for a set amount of epochs.
        Returns train_results dict with loss per epoch for plotting.
        """
        self.train()
        train_results = defaultdict(list)

        for epoch in range(epochs):
            total_loss = 0
            for inputs, expected_output in zip(training_data, labels):
                self.optimizer.zero_grad()
                prediction = self.forward(torch.tensor(inputs, dtype=torch.float32))
                loss = self.loss_fn(prediction, torch.tensor(expected_output, dtype=torch.float32))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            train_results['loss'].append(total_loss)
            print(f"Epoch {epoch + 1}/{epochs} => loss {total_loss:.4f}")

        return train_results

    # private functions
    def _get_activation(self, activation_type: ActivationTypeIds):
        """Map activation enum to PyTorch activation function."""
        match activation_type:
            case ActivationTypeIds.SIGMOID:
                return nn.Sigmoid()
            case ActivationTypeIds.RELU:
                return nn.ReLU()
            case ActivationTypeIds.LEAKY_RELU:
                return nn.LeakyReLU(negative_slope=0.01)
            case ActivationTypeIds.TANH:
                return nn.Tanh()
            case _:
                raise ValueError(f"Unknown activation function: {activation_type}")

    def _get_optimizer(self, optimizer_type: OptimizerTypeIds, learning_rate: float):
        """Map optimizer enum to PyTorch optimizer."""
        match optimizer_type:
            case OptimizerTypeIds.SGD:
                return torch.optim.SGD(self.parameters(), lr=learning_rate)
            case OptimizerTypeIds.ADAM:
                return torch.optim.Adam(self.parameters(), lr=learning_rate)
            case _:
                raise ValueError(f"Unknown optimizer: {optimizer_type}")
