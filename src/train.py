from model.base_model import BaseModel
import numpy as np
import numpy.typing as npt
from collections import defaultdict

def train(
    model: BaseModel,
    training_data: npt.NDArray[np.float64],  # 2D matrix (samples x features)
    labels: npt.NDArray[np.float64],         # 2D matrix (samples x classes) for one-hot encoding
    epochs: int,
    learning_rate: float = 0.01
):
    """
    Train a model with interface BaseModel given training data and labels for a set amount of epochs.
    Learning rate default is 0.01, dictates how big the step is for learning.
        If its too high it might overshoot the optimal value and training bounces around back and forth or diverges
        If its too low it will take much longer to converge and can get stuck in a valley
    """

    # for visualization and analysis
    results_to_analyse = defaultdict(list) # autocreates empty list

    for epoch in range(epochs):

        # reset total loss on every epoch
        total_loss = 0

        # train on each value from training data and expected output
        for inputs, expected_output in zip(training_data, labels):

            # get current prediction by forward passing through a model
            prediction = model.feedforward(inputs)

            # compute gradients via backpropagation
            weight_grads, bias_grads = model.backpropagate(prediction, expected_output)

            # update weights using optimizer
            model.optimize_weights(weight_grads, bias_grads, learning_rate)

            # track total loss
            error = np.sum((expected_output - prediction)**2)
            total_loss += error
            results_to_analyse['loss'].append(total_loss)

        print(f"Epoch {epoch + 1}/{epochs} => loss {total_loss:.4f}")

    return results_to_analyse
