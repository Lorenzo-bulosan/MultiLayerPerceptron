import numpy as np
from utils.activation_function_ids import ActivationTypeIds
from utils.activation_functions import ActivationFunctions

class Perceptron:

    def __init__(
        self, 
        inputs, 
        weights, 
        bias, 
        activation_type = ActivationTypeIds.SIGMOID
    ):
        
        if inputs is None or weights is None:
            raise ValueError("Inputs and weights cannot be None")
        if not isinstance(inputs, np.ndarray):
            raise TypeError(f"Inputs is expected to be a numpy array, instead its a: {type(inputs).__name__}")
        if not isinstance(weights, np.ndarray):
            raise TypeError(f"Weights is expected to be a numpy array, instead its a: {type(weights).__name__}")        
        if len(inputs) != len(weights):
            raise ValueError(f"Inputs length ({len(inputs)}) must match weights length ({len(weights)})")
        if not isinstance(bias, (int, float)):
            raise TypeError(f"Bias must be a number, got {type(bias).__name__}")

        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.activation_function = ActivationFunctions(activation_type)
    
    def feedforward(self) -> float:
        """
        Computes forward pass by calculating the weighted sum of inputs + bias and applies activation function
        """

        try:
            # weighted sum
            weighted_sum = np.dot(self.x, self.weights)

            # calculate logit or pre-activation, normally denoted by z
            logit = weighted_sum + self.bias

            # apply activation function
            result = self.activation_function.apply(logit)

            return result
        
        except Exception as e:
            print(f"Error in feedforward: {e}")
            raise # to not silently fail
