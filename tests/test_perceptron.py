import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.perceptron import Perceptron
from utils.activation_function_ids import ActivationTypeIds

# --- Constructor validation tests ---

class TestPerceptronInit:

    def test_valid_creation(self):
        p = Perceptron()
        assert p.activation_function.activation_function_id == ActivationTypeIds.SIGMOID

    def test_default_activation_is_sigmoid(self):
        p = Perceptron()
        assert p.activation_function.activation_function_id == ActivationTypeIds.SIGMOID

    def test_custom_activation(self):
        p = Perceptron(ActivationTypeIds.RELU)
        assert p.activation_function.activation_function_id == ActivationTypeIds.RELU


# --- Feedforward validation tests ---

class TestFeedforwardValidation:

    def test_none_inputs_raises(self):
        p = Perceptron()
        with pytest.raises(ValueError, match="cannot be None"):
            p.feedforward(None, np.array([1.0]), np.array([0.0]))

    def test_none_weights_raises(self):
        p = Perceptron()
        with pytest.raises(ValueError, match="cannot be None"):
            p.feedforward(np.array([1.0]), None, np.array([0.0]))

    def test_none_bias_raises(self):
        p = Perceptron()
        with pytest.raises(ValueError, match="cannot be None"):
            p.feedforward(np.array([1.0]), np.array([0.5]), None)

    def test_inputs_not_numpy_raises(self):
        p = Perceptron()
        with pytest.raises(TypeError, match="Inputs is expected to be a numpy array"):
            p.feedforward([1.0, 2.0], np.array([0.5, 0.5]), np.array([0.0]))

    def test_weights_not_numpy_raises(self):
        p = Perceptron()
        with pytest.raises(TypeError, match="Weights is expected to be a numpy array"):
            p.feedforward(np.array([1.0, 2.0]), [0.5, 0.5], np.array([0.0]))

    def test_bias_not_numpy_raises(self):
        p = Perceptron()
        with pytest.raises(TypeError, match="Bias is expected to be a numpy array"):
            p.feedforward(np.array([1.0]), np.array([0.5]), "bad")

    def test_mismatched_lengths_raises(self):
        p = Perceptron()
        with pytest.raises(ValueError, match="must match weights length"):
            p.feedforward(np.array([1.0, 2.0]), np.array([0.5]), np.array([0.0]))


# --- Feedforward tests ---

class TestPerceptronFeedforward:

    class TestSigmoid:
        def test_positive_preactivation(self):
            """z = (5*0.6 + 8*0.4) + (-4.0) = 2.2, sigmoid(2.2) ≈ 0.9002"""
            inputs = np.array([5.0, 8.0])
            weights = np.array([0.6, 0.4])
            p = Perceptron(ActivationTypeIds.SIGMOID)
            result = p.feedforward(inputs, weights, np.array([-4.0]))
            assert result == pytest.approx(0.9002, abs=0.001)

        def test_negative_preactivation(self):
            """z = (1*0.6 + 3*0.4) + (-4.0) = -2.2, sigmoid(-2.2) ≈ 0.0998"""
            inputs = np.array([1.0, 3.0])
            weights = np.array([0.6, 0.4])
            p = Perceptron(ActivationTypeIds.SIGMOID)
            result = p.feedforward(inputs, weights, np.array([-4.0]))
            assert result == pytest.approx(0.0998, abs=0.001)

        def test_zero_value_preactivation(self):
            """z = 0, sigmoid(0) = 0.5"""
            inputs = np.array([0.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.SIGMOID)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == pytest.approx(0.5, abs=0.001)

    class TestRelu:
        def test_relu_positive(self):
            """z = (2*1) + 0 = 2, relu(2) = 2"""
            inputs = np.array([2.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.RELU)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == 2.0

        def test_relu_negative(self):
            """z = (-3*1) + 0 = -3, relu(-3) = 0"""
            inputs = np.array([-3.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.RELU)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == 0.0

        def test_relu_zero(self):
            """z = 0, relu(0) = 0"""
            inputs = np.array([0.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.RELU)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == 0.0

        def test_relu_large_input(self):
            """z = 1000, relu(1000) = 1000 (no saturation)"""
            inputs = np.array([1000.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.RELU)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == 1000.0

    class TestLeakyRelu:
        def test_leaky_relu_positive(self):
            """z = (2*1) + 0 = 2, leaky_relu(2) = 2"""
            inputs = np.array([2.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.LEAKY_RELU)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == 2.0

        def test_leaky_relu_negative(self):
            """z = (-3*1) + 0 = -3, leaky_relu(-3) = 0.01 * -3 = -0.03"""
            inputs = np.array([-3.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.LEAKY_RELU)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == pytest.approx(-0.03, abs=0.001)

        def test_leaky_relu_zero(self):
            """z = 0, leaky_relu(0) = 0"""
            inputs = np.array([0.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.LEAKY_RELU)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == 0.0

        def test_leaky_relu_small_negative(self):
            """z = -0.001, leaky_relu(-0.001) = 0.01 * -0.001 = -0.00001"""
            inputs = np.array([-0.001])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.LEAKY_RELU)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == pytest.approx(-0.00001, abs=0.000001)

    class TestTanh:
        def test_positive_preactivation(self):
            """z = (1*1) + 0 = 1, tanh(1) ≈ 0.7616"""
            inputs = np.array([1.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.TANH)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == pytest.approx(0.7616, abs=0.001)

        def test_negative_preactivation(self):
            """z = (-1*1) + 0 = -1, tanh(-1) ≈ -0.7616"""
            inputs = np.array([-1.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.TANH)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == pytest.approx(-0.7616, abs=0.001)

        def test_zero_preactivation(self):
            """z = 0, tanh(0) = 0"""
            inputs = np.array([0.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.TANH)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == pytest.approx(0.0, abs=0.001)

        def test_large_positive_saturation(self):
            """z = 10, tanh(10) ≈ 1.0 (saturates near 1)"""
            inputs = np.array([10.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.TANH)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == pytest.approx(1.0, abs=0.001)

        def test_large_negative_saturation(self):
            """z = -10, tanh(-10) ≈ -1.0 (saturates near -1)"""
            inputs = np.array([-10.0])
            weights = np.array([1.0])
            p = Perceptron(ActivationTypeIds.TANH)
            result = p.feedforward(inputs, weights, np.array([0.0]))
            assert result == pytest.approx(-1.0, abs=0.001)


# --- Backpropagate tests ---

class TestPerceptronBackpropagate:

    def test_returns_weight_and_bias_gradients(self):
        """backpropagate should return weight_grads and bias_grads"""
        inputs = np.array([1.0, 2.0])
        weights = np.array([[0.5], [0.5]])
        bias = np.array([0.0])
        prediction = np.array([0.8])
        expected_output = np.array([1.0])
        p = Perceptron(ActivationTypeIds.SIGMOID)
        weight_grads, bias_grads = p.backpropagate(inputs, weights, bias, prediction, expected_output)
        assert weight_grads.shape == weights.shape
        assert bias_grads.shape == bias.shape

    def test_zero_error_produces_zero_gradients(self):
        """When prediction equals expected output, gradients should be zero"""
        inputs = np.array([1.0, 2.0])
        weights = np.array([[0.5], [0.5]])
        bias = np.array([0.0])
        prediction = np.array([1.0])
        expected_output = np.array([1.0])
        p = Perceptron(ActivationTypeIds.RELU)
        weight_grads, bias_grads = p.backpropagate(inputs, weights, bias, prediction, expected_output)
        assert np.allclose(weight_grads, 0.0)
        assert np.allclose(bias_grads, 0.0)

    def test_positive_error_produces_negative_gradients(self):
        """When prediction < expected, dl_da is negative (gradient should push weights up)"""
        inputs = np.array([1.0])
        weights = np.array([[1.0]])
        bias = np.array([0.0])
        prediction = np.array([0.5])
        expected_output = np.array([1.0])
        p = Perceptron(ActivationTypeIds.RELU)
        weight_grads, bias_grads = p.backpropagate(inputs, weights, bias, prediction, expected_output)
        # dl_da = -2 * (1.0 - 0.5) = -1.0, relu derivative at z=1 is 1
        # gradient = -1.0 * 1.0 = -1.0
        assert weight_grads[0][0] < 0
        assert bias_grads[0] < 0

    def test_negative_error_produces_positive_gradients(self):
        """When prediction > expected, dl_da is positive (gradient should push weights down)"""
        inputs = np.array([1.0])
        weights = np.array([[1.0]])
        bias = np.array([0.0])
        prediction = np.array([1.5])
        expected_output = np.array([1.0])
        p = Perceptron(ActivationTypeIds.RELU)
        weight_grads, bias_grads = p.backpropagate(inputs, weights, bias, prediction, expected_output)
        assert weight_grads[0][0] > 0
        assert bias_grads[0] > 0


# --- Optimize weights tests ---

class TestPerceptronOptimizeWeights:

    def test_weights_unchanged_with_zero_gradients(self):
        """Zero gradients should not change weights"""
        weights = np.array([[0.5], [0.5]])
        bias = np.array([0.0])
        weight_grads = np.array([[0.0], [0.0]])
        bias_grads = np.array([0.0])
        p = Perceptron(ActivationTypeIds.RELU)
        new_weights, new_bias = p.optimize_weights(weights, bias, weight_grads, bias_grads, 0.01)
        assert np.array_equal(new_weights, weights)
        assert np.array_equal(new_bias, bias)

    def test_weights_decrease_with_positive_gradients(self):
        """Positive gradients should decrease weights (moving opposite to gradient)"""
        weights = np.array([[1.0]])
        bias = np.array([1.0])
        weight_grads = np.array([[0.5]])
        bias_grads = np.array([0.5])
        p = Perceptron(ActivationTypeIds.RELU)
        new_weights, new_bias = p.optimize_weights(weights, bias, weight_grads, bias_grads, 0.1)
        assert new_weights[0][0] < weights[0][0]
        assert new_bias[0] < bias[0]

    def test_weights_increase_with_negative_gradients(self):
        """Negative gradients should increase weights"""
        weights = np.array([[1.0]])
        bias = np.array([1.0])
        weight_grads = np.array([[-0.5]])
        bias_grads = np.array([-0.5])
        p = Perceptron(ActivationTypeIds.RELU)
        new_weights, new_bias = p.optimize_weights(weights, bias, weight_grads, bias_grads, 0.1)
        assert new_weights[0][0] > weights[0][0]
        assert new_bias[0] > bias[0]

    def test_learning_rate_scales_update(self):
        """Higher learning rate should produce larger weight changes"""
        weights = np.array([[1.0]])
        bias = np.array([1.0])
        weight_grads = np.array([[1.0]])
        bias_grads = np.array([1.0])
        p = Perceptron(ActivationTypeIds.RELU)

        w_small, b_small = p.optimize_weights(weights, bias, weight_grads, bias_grads, 0.01)
        w_large, b_large = p.optimize_weights(weights, bias, weight_grads, bias_grads, 0.1)

        # larger learning rate = bigger change from original
        assert abs(w_large[0][0] - weights[0][0]) > abs(w_small[0][0] - weights[0][0])
        assert abs(b_large[0] - bias[0]) > abs(b_small[0] - bias[0])
