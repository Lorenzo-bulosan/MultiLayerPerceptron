from model.base_model import BaseModel
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from utils.optimizer_type_ids import OptimizerTypeIds


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
        error = np.mean((expected_output - prediction)**2)
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


def _build_confusion(results, test_labels):
    """Build confusion matrix from results."""
    predictions = np.array(results['predictions'])
    expected = np.array(test_labels)
    num_classes = predictions.shape[1]
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(expected, axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_classes, pred_classes):
        confusion[t][p] += 1
    return confusion, num_classes


def _plot_confusion(ax, confusion, num_classes, title):
    """Plot a confusion matrix on a given axis."""
    im = ax.imshow(confusion, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(confusion[i][j]), ha='center', va='center',
                    color='white' if confusion[i][j] > confusion.max() / 2 else 'black', fontsize=14)


def plot_results(results, test_labels, path_to_save_figures, train_results=None, results2=None, train_results2=None):
    """Plot combined loss curve and confusion matrices in a 1x3 layout."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. loss curves (both on same plot)
    if train_results:
        axes[0].plot(train_results['loss'], label='Our MLP')
    if train_results2:
        axes[0].plot(train_results2['loss'], label='PyTorch MLP')
    axes[0].set_title('Loss over training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 2. our confusion matrix
    confusion, num_classes = _build_confusion(results, test_labels)
    _plot_confusion(axes[1], confusion, num_classes, f"Our MLP (Accuracy: {results['accuracy']:.2%})")

    # 3. pytorch confusion matrix
    if results2:
        confusion2, num_classes2 = _build_confusion(results2, test_labels)
        _plot_confusion(axes[2], confusion2, num_classes2, f"PyTorch MLP (Accuracy: {results2['accuracy']:.2%})")

    plt.tight_layout()
    fig.savefig(os.path.join(path_to_save_figures, "ErrorGraphAndConfusionMatrix.png"), dpi=300, bbox_inches="tight")
    plt.show()


def plot_moon_data(X, y, path_to_save_figures):
    """Scatter plot of the raw moons dataset, colored by class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0', edgecolors='black', s=40, alpha=0.7)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1', edgecolors='black', s=40, alpha=0.7)
    ax.set_title("Moons Dataset")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(path_to_save_figures, "DataToClasify_Moon.png"), dpi=300, bbox_inches="tight")
    plt.show(block=False)


if __name__ == "__main__":
    import sys
    import os
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split

    src_dir = os.path.dirname(__file__)
    sys.path.insert(0, src_dir)
    path_to_save_figures = os.path.join(src_dir, "..", "output")

    from model.multilayer_perceptron import MLP
    from model.pytorch.mlp_pytorch import MLP_PyTorch
    from utils.activation_function_ids import ActivationTypeIds
    from train import train

    # --- generate moons dataset ---
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    plot_moon_data(X, y, path_to_save_figures)

    # convert labels to one-hot: 0 → [1, 0], 1 → [0, 1]
    labels = np.zeros((len(y), 2))
    labels[np.arange(len(y)), y] = 1.0

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # MODEL params
    LAYER_SIZE = [2, 8, 8, 2]
    ACTIVATION_TYPE = ActivationTypeIds.LEAKY_RELU
    OPTIMIZER = OptimizerTypeIds.ADAM
    
    # Training params
    EPOCHS = 100
    LEARNING_RATE = 0.01

    # === Our MLP ===
    model = MLP(
        layer_sizes=LAYER_SIZE,
        activation_function_type=ACTIVATION_TYPE,
        optimizer_type=OPTIMIZER
    )

    print("Training our MLP...")
    train_results = train(model, X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    print("\nEvaluating our MLP...")
    results = evaluate(model, X_test, y_test)
    print(f"Accuracy:  {results['accuracy']:.2%}")
    print(f"Avg Loss:  {results['avg_loss']:.4f}")
    print(f"Correct:   {results['correct']}/{results['total_samples']}")

    # === PyTorch MLP ===
    model2 = MLP_PyTorch(
        layer_sizes=LAYER_SIZE,
        activation_function_type=ACTIVATION_TYPE,
        optimizer_type=OPTIMIZER,
        learning_rate=LEARNING_RATE
    )

    print("\nTraining PyTorch MLP...")
    train_results2 = model2.train_model(X_train, y_train, epochs=EPOCHS)

    print("\nEvaluating PyTorch MLP...")
    model2.eval()
    results2 = evaluate(model2, X_test, y_test)
    print(f"Accuracy:  {results2['accuracy']:.2%}")
    print(f"Avg Loss:  {results2['avg_loss']:.4f}")
    print(f"Correct:   {results2['correct']}/{results2['total_samples']}")

    # --- plot both side by side ---
    plot_results(results, y_test.tolist(), path_to_save_figures, train_results, results2, train_results2)
