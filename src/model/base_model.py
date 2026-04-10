from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

class BaseModel(ABC):

    @abstractmethod
    def feedforward(self, inputs: npt.NDArray[np.float64], **kwargs):
        """ Must implement a feedforward"""
        pass

    @abstractmethod
    def backpropagate(self, prediction: npt.NDArray[np.float64], expected_output: npt.NDArray[np.float64], **kwargs):
        """ Must implement backpropagation to compute gradients"""
        pass

    @abstractmethod
    def optimize_weights(self, weight_gradients, bias_gradients, learning_rate: np.float64, **kwargs):
        """ Must implement an optimizer to update weights (e.g. SGD, Adam)"""
        pass