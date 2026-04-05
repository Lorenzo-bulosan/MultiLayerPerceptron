import numpy as np
import numpy.typing as npt
from utils.activation_function_ids import ActivationTypeIds
from utils.activation_functions import ActivationFunctions
from model.base_model import BaseModel

class Perceptron(BaseModel):

    def __init__(
        self, 
        activation_type = ActivationTypeIds.SIGMOID
    ):
        # todo: add validation for activation type
        self.activation_function = ActivationFunctions(activation_type)
    
    def feedforward(
        self,
        inputs: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64]
    ):
        """
        Computes forward pass by calculating the weighted sum of inputs + bias and applies activation function
        """
        # validate inputs
        if inputs is None or weights is None or bias is None:
            raise ValueError("Inputs, weights, and bias cannot be None")
        if not isinstance(inputs, np.ndarray):
            raise TypeError(f"Inputs is expected to be a numpy array, instead its a: {type(inputs).__name__}")
        if not isinstance(weights, np.ndarray):
            raise TypeError(f"Weights is expected to be a numpy array, instead its a: {type(weights).__name__}")
        if not isinstance(bias, np.ndarray):
            raise TypeError(f"Bias is expected to be a numpy array, instead its a: {type(bias).__name__}")
        if weights.ndim == 1 and len(inputs) != len(weights):
            raise ValueError(f"Inputs length ({len(inputs)}) must match weights length ({len(weights)})")
        if weights.ndim == 2 and len(inputs) != weights.shape[0]:
            raise ValueError(f"Inputs length ({len(inputs)}) must match weights rows ({weights.shape[0]})")

        # feedforward algorithm
        try:
            # weighted sum
            weighted_sum = np.dot(inputs, weights)

            # calculate logit or pre-activation, normally denoted by z
            logit = weighted_sum + bias

            # apply activation function
            result = self.activation_function.apply(logit)

            return result
        
        except Exception as e:
            print(f"Error in feedforward: {e}")
            raise # to not silently fail

    def backpropagate(
        self,
        inputs: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64],
        prediction: npt.NDArray[np.float64],
        expected_output: npt.NDArray[np.float64]
    ):
        """
        Compute gradients using the chain rule (backpropagation)
        """
        try:
            # dL/da — how much the loss changes with the prediction
            dl_da = -2 * (expected_output - prediction)

            # da/dz — how much the prediction changes with the pre-activation
            logit = np.dot(inputs, weights) + bias
            da_dz = self.activation_function.derivative(logit)

            # combined gradient: dL/dz = dL/da * da/dz
            gradient = dl_da * da_dz

            # weight gradients: outer product of inputs and gradient
            weight_grads = np.outer(inputs, gradient)
            bias_grads = gradient

            return weight_grads, bias_grads

        except Exception as e:
            print(f"Error in backpropagate: {e}")
            raise

    def optimize_weights(
        self,
        weights: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64],
        weight_grads: npt.NDArray[np.float64],
        bias_grads: npt.NDArray[np.float64],
        learning_rate: np.float64
    ):
        """
        Update weights using SGD: move opposite to gradient at a given learning rate
        """
        try:
            weights = weights - learning_rate * weight_grads
            bias = bias - learning_rate * bias_grads
            return weights, bias

        except Exception as e:
            print(f"Error in optimize_weights: {e}")
            raise