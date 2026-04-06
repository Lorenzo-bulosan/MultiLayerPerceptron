from model.base_model import BaseModel
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def evaluate(
    model: BaseModel,
    test_data: npt.NDArray[np.float64],
    test_labels: npt.NDArray[np.float64]
):
    """
    Evaluate a trained model on test data.
    Returns accuracy, average loss, and per-sample predictions.
    """
    correct = 0
    total_loss = 0
    predictions = []

    for inputs, expected_output in zip(test_data, test_labels):
        prediction = model.feedforward(inputs)
        predictions.append(prediction)

        # loss (MSE)
        error = np.sum((expected_output - prediction)**2)
        total_loss += error

        # accuracy: compare which class has highest value
        if np.argmax(prediction) == np.argmax(expected_output):
            correct += 1

    accuracy = correct / len(test_labels)
    avg_loss = total_loss / len(test_labels)

    return {
        'accuracy': accuracy,
        'avg_loss': avg_loss,
        'total_samples': len(test_labels),
        'correct': correct,
        'predictions': predictions
    }


def plot_results(results, test_labels, train_results=None):
    """Plot evaluation results: loss over training, accuracy bar, and prediction vs expected comparison."""
    num_plots = 3 if train_results else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))

    plot_idx = 0

    # 0. loss over training
    if train_results:
        axes[plot_idx].plot(train_results['loss'])
        axes[plot_idx].set_title('Loss over training')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].grid(True)
        plot_idx += 1

    # 1. accuracy bar
    axes[plot_idx].bar(['Correct', 'Incorrect'], [results['correct'], results['total_samples'] - results['correct']], color=['green', 'red'])
    axes[plot_idx].set_title(f"Accuracy: {results['accuracy']:.2%}")
    axes[plot_idx].set_ylabel('Number of samples')
    plot_idx += 1

    # 2. predicted vs expected per sample
    predictions = np.array(results['predictions'])
    expected = np.array(test_labels)
    num_samples = len(test_labels)
    num_classes = predictions.shape[1]

    x = np.arange(num_samples)
    width = 0.35

    for c in range(num_classes):
        axes[plot_idx].bar(x - width/2 + c * width/num_classes, expected[:, c], width/num_classes, label=f'Expected class {c}', alpha=0.5)
        axes[plot_idx].bar(x + width/2 + c * width/num_classes, predictions[:, c], width/num_classes, label=f'Predicted class {c}', alpha=0.8)

    axes[plot_idx].set_title(f"Predictions vs Expected (Avg Loss: {results['avg_loss']:.4f})")
    axes[plot_idx].set_xlabel('Sample')
    axes[plot_idx].set_ylabel('Value')
    axes[plot_idx].set_xticks(x)
    axes[plot_idx].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    import os

    # paths
    src_dir = os.path.dirname(__file__)
    project_root = os.path.join(src_dir, '..')
    data_dir = os.path.join(project_root, 'data')

    sys.path.insert(0, src_dir)

    from model.multilayer_perceptron import MLP
    from utils.activation_function_ids import ActivationTypeIds
    from train import train

    # --- load data from csv ---
    training_data = np.loadtxt(os.path.join(data_dir, 'training_data.csv'), delimiter=',')
    training_labels = np.loadtxt(os.path.join(data_dir, 'training_labels.csv'), delimiter=',')
    test_data = np.loadtxt(os.path.join(data_dir, 'test_data.csv'), delimiter=',')
    test_labels = np.loadtxt(os.path.join(data_dir, 'test_labels.csv'), delimiter=',')

    # --- train ---
    model = MLP(
        layer_sizes=[2, 4, 2],
        activation_function_type=ActivationTypeIds.LEAKY_RELU
    )

    print("Training...")
    train_results = train(model, training_data, training_labels, epochs=5000, learning_rate=0.05)

    # --- evaluate ---
    print("\nEvaluating...")
    results = evaluate(model, test_data, test_labels)
    print(f"Accuracy:  {results['accuracy']:.2%}")
    print(f"Avg Loss:  {results['avg_loss']:.4f}")
    print(f"Correct:   {results['correct']}/{results['total_samples']}")

    plot_results(results, test_labels, train_results)
