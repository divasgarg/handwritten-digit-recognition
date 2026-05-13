import os

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from . import data_loader
from .cnn_model import build_cnn
from .config import MODELS_DIR, CNN_MODEL_PATH


def main(epochs: int = 10, batch_size: int = 128):
    (x_train, y_train), (x_test, y_test) = data_loader.load_mnist()

    model = build_cnn(input_shape=x_train.shape[1:])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(CNN_MODEL_PATH)
    print(f"Saved CNN model to {CNN_MODEL_PATH}")

    # Plot training history
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('CNN Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, 'cnn_training_accuracy.png')
    plt.savefig(plot_path)
    print(f"Saved training accuracy plot to {plot_path}")


if __name__ == '__main__':
    main()
