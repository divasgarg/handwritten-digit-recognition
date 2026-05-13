import argparse
import os

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import data_loader
from . import classical_models
from .config import MODELS_DIR, LR_MODEL_PATH, SVM_MODEL_PATH, KNN_MODEL_PATH


def main(model_type: str = 'lr'):
    (x_train, y_train), (x_test, y_test) = data_loader.load_mnist(reshape_for_cnn=False)
    x_train_flat, x_test_flat = data_loader.flatten_for_classical(x_train, x_test)

    if model_type == 'lr':
        model = classical_models.build_logistic_regression()
        model_path = LR_MODEL_PATH
    elif model_type == 'svm':
        model = classical_models.build_svm()
        model_path = SVM_MODEL_PATH
    elif model_type == 'knn':
        model = classical_models.build_knn()
        model_path = KNN_MODEL_PATH
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = classical_models.train_model(model, x_train_flat, y_train)
    y_pred = model.predict(x_test_flat)

    print(f"Model type: {model_type}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:
", classification_report(y_test, y_pred))
    print("Confusion matrix:
", confusion_matrix(y_test, y_pred))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train classical ML models on MNIST')
    parser.add_argument('--model-type', type=str, default='lr', choices=['lr', 'svm', 'knn'])
    args = parser.parse_args()
    main(args.model_type)
