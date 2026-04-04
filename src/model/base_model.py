from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

class BaseModel(ABC):

    @abstractmethod
    def feedforward(
        self,
        inputs: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64]
    ):
        """ Must implement a feedforward"""
        pass

    @abstractmethod
    def backpropagate(
        self,
        inputs: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64],
        prediction: npt.NDArray[np.float64],
        expected_output: npt.NDArray[np.float64]
    ):
        """ Must implement backpropagation to compute gradients"""
        pass

    @abstractmethod
    def optimize_weights(
        self,
        weights: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64],
        weight_gradients: npt.NDArray[np.float64],
        bias_gradients: npt.NDArray[np.float64],
        learning_rate: np.float64
    ):
        """ Must implement an optimizer to update weights (e.g. SGD, Adam)"""
        pass