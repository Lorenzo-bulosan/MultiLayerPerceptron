from utils.activation_function_ids import ActivationTypeIds
import numpy as np

class ActivationFunctions:

    def __init__(self, activation_function_id: ActivationTypeIds, leaky_relu_alpha = 0.01):
        self.activation_function_id = activation_function_id
        self.leaky_relu_alpha = leaky_relu_alpha # common default number https://apxml.com/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/relu-variants
    
    def apply(self, logit: float) -> float:
        """
        Compute output of activation function for the logit, pre-activation value z
        """

        if not isinstance(logit, (int, float)):
            raise TypeError(f"Expected a number, got type {type(logit).__name__}")

        result = 0
        z = logit # z for readability when comparing with math formulas

        try:
            match self.activation_function_id:
    
                # pros: outputs between 0 and 1 (useful for probabilities)
                # cons: vanishing gradient for large/small z (saturates at 0 and 1), outputs not zero-centered
                case ActivationTypeIds.SIGMOID:
                    result = 1 / (1 + np.exp(-z))

                # pros: no vanishing gradient problem
                # cons: can suffer from dying ReLU (neurons can get stuck at 0 and never recover and never fire), exploding gradients
                case ActivationTypeIds.RELU:
                    result = max(0, z)

                # pros: does not suffer from dying ReLU, no vanishing gradient problem
                # cons: exploding gradients
                case ActivationTypeIds.LEAKY_RELU:                    
                    result = max(self.leaky_relu_alpha * z, z) # gradient is never fully 0

                # pros: zero-centered output (-1 to 1), stronger gradients than sigmoid
                # cons: vanishing gradient for large/small z (saturates at -1 and 1)
                case ActivationTypeIds.TANH:
                    result = np.tanh(z)

                case _:
                    raise ValueError(f"Unknown activation function: {self.activation_function_id}")
                
            return result
        
        except Exception as e:
            print(f"Error applying activation function: {e}")
            raise

    def derivative(self, logit: float) -> float:
        """
        Compute the derivative of the activation function for the logit, pre-activation value z.
        Used during backpropagation to calculate gradients.
        """

        if not isinstance(logit, (int, float)):
            raise TypeError(f"Expected a number, got type {type(logit).__name__}")

        result = 0
        z = logit

        try:
            match self.activation_function_id:

                # d/dz sigmoid(z) = sigmoid(z) * (1 - sigmoid(z))
                case ActivationTypeIds.SIGMOID:
                    s = 1 / (1 + np.exp(-z))
                    result = s * (1 - s)

                # d/dz relu(z) = 1 if z > 0, else 0
                case ActivationTypeIds.RELU:
                    result = 1.0 if z > 0 else 0.0

                # d/dz leaky_relu(z) = 1 if z > 0, else alpha
                case ActivationTypeIds.LEAKY_RELU:
                    result = 1.0 if z > 0 else self.leaky_relu_alpha

                # d/dz tanh(z) = 1 - tanh(z)^2
                case ActivationTypeIds.TANH:
                    result = 1 - np.tanh(z) ** 2

                case _:
                    raise ValueError(f"Unknown activation function: {self.activation_function_id}")

            return result

        except Exception as e:
            print(f"Error computing derivative: {e}")
            raise