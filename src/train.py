from model.base_model import BaseModel
import numpy as np
import numpy.typing as npt
from collections import defaultdict

def train(
    model: BaseModel,
    training_data: npt.NDArray[np.float64],  # 2D matrix (samples x features)
    labels: npt.NDArray[np.float64],         # 2D matrix (samples x classes) for one-hot encoding
    epochs: int,
    learning_rate: float = 0.01,
    batch_size: int = None                    # None = full dataset as one batch
):
    """
    Train a model with interface BaseModel given training data and labels for a set amount of epochs.
    Learning rate default is 0.01, dictates how big the step is for learning.
        If its too high it might overshoot the optimal value and training bounces around back and forth or diverges
        If its too low it will take much longer to converge and can get stuck in a valley

    batch_size: number of samples to accumulate gradients over before updating weights.
        None = full dataset (full batch gradient descent)
        1 = stochastic gradient descent (SGD) — update after every sample
        N = mini-batch gradient descent — update after every N samples
    """

    # default batch size is full dataset
    if batch_size is None:
        batch_size = len(training_data)

    # for visualization and analysis
    results_to_analyse = defaultdict(list) # autocreates empty list

    for epoch in range(epochs):

        # reset total loss on every epoch
        total_loss = 0

        # zero accumulated gradients
        total_w_grads = None
        total_b_grads = None
        samples_in_batch = 0

        # train on each value from training data and expected output
        for i, (inputs, expected_output) in enumerate(zip(training_data, labels)):

            # get current prediction by forward passing through a model
            prediction = model.feedforward(inputs)

            # compute gradients via backpropagation
            weight_grads, bias_grads = model.backpropagate(prediction, expected_output)

            # accumulate gradients
            if total_w_grads is None:
                total_w_grads = [np.zeros_like(g) for g in weight_grads]
                total_b_grads = [np.zeros_like(g) for g in bias_grads]

            for j in range(len(weight_grads)):
                total_w_grads[j] += weight_grads[j]
                total_b_grads[j] += bias_grads[j]

            samples_in_batch += 1

            # update weights when batch is full
            if samples_in_batch == batch_size or i == len(training_data) - 1:
                # average gradients over batch
                avg_w_grads = [g / samples_in_batch for g in total_w_grads]
                avg_b_grads = [g / samples_in_batch for g in total_b_grads]

                model.optimize_weights(avg_w_grads, avg_b_grads, learning_rate)

                # zero gradients for next batch
                total_w_grads = [np.zeros_like(g) for g in weight_grads]
                total_b_grads = [np.zeros_like(g) for g in bias_grads]
                samples_in_batch = 0

            # track total loss
            error = np.sum((expected_output - prediction)**2)
            total_loss += error

        results_to_analyse['loss'].append(total_loss)
        print(f"Epoch {epoch + 1}/{epochs} => loss {total_loss:.4f}")

    return results_to_analyse
