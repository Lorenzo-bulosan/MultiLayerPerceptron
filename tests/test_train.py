import sys
import os
import numpy as np
import pytest
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.multilayer_perceptron import MLP
from utils.activation_function_ids import ActivationTypeIds
from train import train


class TestTrain:

    def test_simple_leaky_relu_training(self):
        """
        Simple AND gate with Leaky ReLU:
        [0, 0] → 0
        [0, 1] → 0
        [1, 0] → 0
        [1, 1] → 1

        2 features, 4 samples, 2 output classes
        MLP architecture: 2 inputs → 4 hidden → 2 outputs
        """
        model = MLP(
            layer_sizes=[2, 4, 2],
            activation_function_type=ActivationTypeIds.LEAKY_RELU
        )

        training_data = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])

        labels = np.array([
            [1.0, 0.0],  # class 0 (false)
            [1.0, 0.0],  # class 0 (false)
            [1.0, 0.0],  # class 0 (false)
            [0.0, 1.0],  # class 1 (true)
        ])

        results = train(model, training_data, labels, epochs=5000, learning_rate=0.05)

        # after training, verify predictions are close to expected
        # prediction is a vector [class0, class1], check class1 (true) is highest for [1,1]
        pred_true = model.feedforward(np.array([1.0, 1.0]))
        assert pred_true[1] > pred_true[0]

        # check class0 (false) is highest for [0,0]
        pred_false = model.feedforward(np.array([0.0, 0.0]))
        assert pred_false[0] >= pred_false[1]

    def test_loss_decreases(self):
        """Loss should decrease over epochs"""
        model = MLP(
            layer_sizes=[2, 4, 2],
            activation_function_type=ActivationTypeIds.LEAKY_RELU
        )

        training_data = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])

        labels = np.array([
            [1.0, 0.0],  # class 0 (false)
            [1.0, 0.0],  # class 0 (false)
            [1.0, 0.0],  # class 0 (false)
            [0.0, 1.0],  # class 1 (true)
        ])

        results = train(model, training_data, labels, epochs=5000, learning_rate=0.05)

        losses = results['loss']
        # loss at end should be less than loss at start
        assert losses[-1] < losses[0]

    @pytest.mark.skip(reason="visual testing")
    def test_visual_loss(self):
        """Visual test to plot loss over training"""
        model = MLP(
            layer_sizes=[2, 4, 2],
            activation_function_type=ActivationTypeIds.LEAKY_RELU
        )

        training_data = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ])

        labels = np.array([
            [1.0, 0.0],  # class 0 (false)
            [1.0, 0.0],  # class 0 (false)
            [1.0, 0.0],  # class 0 (false)
            [0.0, 1.0],  # class 1 (true)
        ])

        results = train(model, training_data, labels, epochs=1000, learning_rate=0.01)

        plt.figure(figsize=(10, 5))
        plt.plot(results['loss'])
        plt.title('Loss over training')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
