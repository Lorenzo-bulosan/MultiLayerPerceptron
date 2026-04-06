from model.base_model import BaseModel
from utils.activation_function_ids import ActivationTypeIds
from utils.activation_functions import ActivationFunctions
import numpy as np
import numpy.typing as npt

class MLP(BaseModel):

    def __init__(
        self,
        layer_sizes: list, # [neurons in input layer, hidden layer 1, ..., output layer]
        activation_function_type = ActivationTypeIds.LEAKY_RELU
    ):
        self.layer_sizes = layer_sizes
        self.activation_function = ActivationFunctions(activation_function_type)

        # create weight matrices and bias vectors for each layer connection
        self.weights, self.biases = self._create_layer_matrices(layer_sizes)

        # stored during feedforward, used during backprop
        self.activations = []
        self.logits = []

    # public functions
    def feedforward(self, inputs: npt.NDArray[np.float64], **kwargs):
        """
        Forward pass through all layers using internal weights and biases.
        """
        try:
            self.activations = [inputs]
            self.logits = []

            a = inputs
            for w, b in zip(self.weights, self.biases):
                # pre-activation: z = a · W + b
                z = np.dot(a, w) + b
                # post-activation: a = activation(z)
                a = self.activation_function.apply(z)

                self.logits.append(z)
                self.activations.append(a)

            return a

        except Exception as e:
            print(f"Error in feedforward: {e}")
            raise

    def backpropagate(
        self,
        prediction: npt.NDArray[np.float64],
        expected_output: npt.NDArray[np.float64],
        **kwargs
    ):
        """
        Compute gradients for all layers using the chain rule (backpropagation).
        Goes through layers in reverse order.
        """
        try:
            weight_grads = []
            bias_grads = []

            # dL/da — start with output error
            dl_da = -2 * (expected_output - prediction)

            # go through layers in reverse
            for i in reversed(range(len(self.weights))):
                # da/dz — derivative of activation at this layer
                da_dz = self.activation_function.derivative(self.logits[i])

                # combined gradient: dL/dz = dL/da * da/dz
                gradient = dl_da * da_dz

                # weight gradients: outer product of this layer's input and gradient
                weight_grads.insert(0, np.outer(self.activations[i], gradient))
                bias_grads.insert(0, gradient)

                # propagate error back to previous layer
                dl_da = np.dot(gradient, self.weights[i].T)

            return weight_grads, bias_grads

        except Exception as e:
            print(f"Error in backpropagate: {e}")
            raise

    def optimize_weights(
        self,
        weight_gradients: list,
        bias_gradients: list,
        learning_rate: np.float64 = 0.01,
        **kwargs
    ):
        """
        Update weights using SGD: move opposite to gradient at a given learning rate.
        Updates internal weights/biases directly.
        """
        try:
            for i in range(len(self.weights)):
                self.weights[i] -= learning_rate * weight_gradients[i]
                self.biases[i] -= learning_rate * bias_gradients[i]

            return self.weights, self.biases

        except Exception as e:
            print(f"Error in optimize_weights: {e}")
            raise

    # private functions
    def _create_layer_matrices(self, layer_sizes: list):
        """
        Create weight matrices and bias vectors for each layer connection.
        e.g. [2, 4, 4, 2] creates:
            weights[0]: (2, 4)  — input → hidden1
            weights[1]: (4, 4)  — hidden1 → hidden2
            weights[2]: (4, 2)  — hidden2 → output
            biases[0]:  (4,)
            biases[1]:  (4,)
            biases[2]:  (2,)
        """
        weights = []
        biases = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            b = np.random.randn(layer_sizes[i + 1])
            weights.append(w)
            biases.append(b)

        return weights, biases
