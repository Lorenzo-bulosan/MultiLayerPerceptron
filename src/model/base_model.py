from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

class BaseModel(ABC):

    @abstractmethod
    def feedforward(
        self,
        inputs: npt.NDArray[np.float64], 
        weights: npt.NDArray[np.float64],
        bias: np.float64
    ):
        """ Must implement a feedforward"""
        pass

    # @abstractmethod
    # def train_weights(
    #     self,
    #     inputs: npt.NDArray[np.float64], 
    #     weights: npt.NDArray[np.float64],
    #     bias: np.float64,        
    #     error: np.float64,
    #     learning_rate: np.float64
    # ):
    #     """ Must implement a method to update weights"""
    #     pass