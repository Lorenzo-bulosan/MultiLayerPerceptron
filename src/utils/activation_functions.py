from utils.activation_function_ids import ActivationTypeIds
import numpy as np

class ActivationFunctions:

    def __innit__(self, activation_function_id: ActivationTypeIds, leaky_relu_alpha = 0.01):
        self.activation_function_id = activation_function_id
        self.leaky_relu_alpha = leaky_relu_alpha # common default number https://apxml.com/courses/introduction-to-deep-learning/chapter-2-activation-functions-architecture/relu-variants
    
    def apply(self, logit: float) -> float:
        """
        Compute output of activation function for the logit, pre-activation value z
        """

        if not isinstance(logit, (int, float)):
            raise TypeError(f"Expected a number, got type {type(logit).__name__}")

        result = 0
        z = logit # for readability when comparing with math formulas

        try:
            match self.activation_function_id:
    
                case ActivationTypeIds.SIGMOID:
                    result = 1 / (1 + np.exp(-z))

                case ActivationTypeIds.RELU:
                    result = max(0, z)

                case ActivationTypeIds.LEAKY_RELU:
                    # normal relu suffer from dying ReLU as neuron can get stuck on 0 and never recover and never fire
                    result = max(self.leaky_relu_alpha * z, z) # gradient is never fully 0

                case ActivationTypeIds.TANH:
                    result = np.tanh(z)

                case _:
                    raise ValueError(f"Unknown activation function: {self.activation_function_id}")
                
            return result
        
        except Exception as e:
            print(f"Error applying activation function: {e}")
            raise
