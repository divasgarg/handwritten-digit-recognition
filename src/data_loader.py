import numpy as np
from tensorflow.keras.datasets import mnist


def load_mnist(normalize: bool = True, reshape_for_cnn: bool = True):
    """Load MNIST dataset and return train/test splits.

    Parameters
    ----------
    normalize : bool
        If True, scales pixel values to [0, 1].
    reshape_for_cnn : bool
        If True, reshapes to (n_samples, 28, 28, 1).

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    if reshape_for_cnn:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test)


def flatten_for_classical(x_train, x_test):
    """Flatten image tensors for classical ML models (n_samples, 784)."""
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    return x_train.reshape(n_train, -1), x_test.reshape(n_test, -1)
