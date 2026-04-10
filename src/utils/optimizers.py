from utils.optimizer_type_ids import OptimizerTypeIds
import numpy as np

class Optimizer:

    def __init__(
        self,
        optimizer_type: OptimizerTypeIds,
        weights: list,
        biases: list,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        self.optimizer_type = optimizer_type

        # Adam hyperparameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Adam state: momentum (m) and velocity (v) for each layer's weights and biases
        if optimizer_type == OptimizerTypeIds.ADAM:
            self.t = 0  # timestep
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]

    def update(self, weights: list, biases: list, weight_grads: list, bias_grads: list, learning_rate: float):
        """Update weights and biases using the chosen optimizer."""

        match self.optimizer_type:

            case OptimizerTypeIds.SGD:
                for i in range(len(weights)):
                    weights[i] -= learning_rate * weight_grads[i]
                    biases[i] -= learning_rate * bias_grads[i]

            case OptimizerTypeIds.ADAM:
                self.t += 1

                for i in range(len(weights)):
                    # update momentum: m = β₁·m + (1-β₁)·gradient
                    self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * weight_grads[i]
                    self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * bias_grads[i]

                    # update velocity: v = β₂·v + (1-β₂)·gradient²
                    self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * weight_grads[i] ** 2
                    self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * bias_grads[i] ** 2

                    # bias correction: m̂ = m / (1-β₁ᵗ), v̂ = v / (1-β₂ᵗ)
                    m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
                    v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
                    m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                    v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

                    # update: w -= lr · m̂ / (√v̂ + ε)
                    weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                    biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

            case _:
                raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        return weights, biases
