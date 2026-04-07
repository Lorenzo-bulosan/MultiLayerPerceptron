from model.base_model import BaseModel
from utils.activation_function_ids import ActivationTypeIds
from utils.activation_functions import ActivationFunctions
from utils.optimizer_type_ids import OptimizerTypeIds
from utils.optimizers import Optimizer
import numpy as np
import numpy.typing as npt

class MLP(BaseModel):

    def __init__(
        self,
        layer_sizes: list, # [neurons in input layer, hidden layer 1, ..., output layer]
        activation_function_type = ActivationTypeIds.LEAKY_RELU,
        optimizer_type = OptimizerTypeIds.SGD
    ):
        self.layer_sizes = layer_sizes
        self.activation_function = ActivationFunctions(activation_function_type)

        # create weight matrices and bias vectors for each layer connection
        self.weights, self.biases = self._create_layer_matrices(layer_sizes)

        # optimizer
        self.optimizer = Optimizer(optimizer_type, self.weights, self.biases)

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
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                # pre-activation: z = a · W + b
                z = np.dot(a, w) + b

                # post-activation: softmax on output layer, chosen activation on hidden layers
                is_output_layer = (i == len(self.weights) - 1)
                if is_output_layer:
                    a = self._softmax(z)
                else:
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

            # dL/da — MSE gradient: -2/n * (y - ŷ)
            n = len(expected_output)
            dl_da = -2 / n * (expected_output - prediction)

            # go through layers in reverse
            for i in reversed(range(len(self.weights))):
                is_output_layer = (i == len(self.weights) - 1)

                if is_output_layer:
                    # softmax Jacobian: da/dz = diag(a) - a·aᵀ
                    a = self.activations[i + 1]
                    jacobian = np.diagflat(a) - np.outer(a, a)
                    gradient = dl_da @ jacobian
                else:
                    # element-wise activation derivative for hidden layers
                    da_dz = self.activation_function.derivative(self.logits[i])
                    gradient = dl_da * da_dz

                # weight gradients: outer product of this layer's input and gradient
                weight_grads.insert(0, np.outer(self.activations[i], gradient)) # the dot products sums, outer doesnt. Also insert because we are going on reverse
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
        Update weights using the chosen optimizer (SGD, Adam, etc.).
        Updates internal weights/biases directly.
        """
        try:
            return self.optimizer.update(self.weights, self.biases, weight_gradients, bias_gradients, learning_rate)

        except Exception as e:
            print(f"Error in optimize_weights: {e}")
            raise

    # private functions
    def _softmax(self, z: npt.NDArray[np.float64]):
        """
        Softmax: converts raw logits into probabilities that sum to 1.
        Subtracts max for numerical stability (prevents overflow in exp).
        """
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

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
