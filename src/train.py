from model.base_model import BaseModel
import numpy as np
import numpy.typing as npt
from collections import defaultdict
import matplotlib.pyplot as plt

def train(
    model: BaseModel,
    training_data: npt.NDArray[np.float64],  # 2D matrix (samples x features)
    labels: npt.NDArray[np.float64],         # 1D vector of expected outputs
    epochs: int,
    learning_rate: float = 0.01
):
    """
    Train a model with interface BaseModel given training data and labels for a set amount of epochs. 
    Learning rate default is 0.01, dictates how big the step is for learning. 
        If its too high it might overshoot the optimal value and training bounces around back and forth or diverges
        If its too low it will take much longer to converge and can get stuck in a valley
    """

    # dimensions
    input_size = training_data.shape[1]
    output_size = labels.shape[1] if labels.ndim > 1 else 1

    # for visualization and analysis
    results_to_analyse = defaultdict(list) # autocreates empty list

    # initializing weights as matrix (input_size x output_size) and bias as vector (output_size)
    weights = np.random.randn(input_size, output_size) # todo: random initialization for now but later add "He" and "Xavier" initializations
    bias = np.random.randn(output_size) # random bias initialization (vector)

    for epoch in range(epochs):
        print(f"Starting epoch: {epoch}")
        
        # reset total loss on every epoch
        total_loss = 0

        # train on each value from training data and expected output
        for inputs, expected_output in zip(training_data, labels):
            # inputs now vector of first sample from features
            # output now a vector containing the correct output given the set of inputs (one-hot encoding for multiclass classification)
            
            # get current prediction by forward passing through a model
            prediction = model.feedforward(inputs, weights, bias)

            # update weights to adjust for this error at a given training rate
            weights, bias = model.train_weights(inputs, weights, bias, prediction, expected_output, learning_rate)

            # track total loss
            error = np.sum((expected_output - prediction)**2)
            total_loss += error # todo: allow for different errors like RMS etc
            results_to_analyse['loss'].append(total_loss)

        print(f"Results epoch: {epoch} => total error {total_loss}")

    return weights, bias, results_to_analyse

# todo: save weights on csv or something that makes sense