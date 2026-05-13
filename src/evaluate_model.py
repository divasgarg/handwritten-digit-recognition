import argparse

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from . import data_loader
from .config import (
    CNN_MODEL_PATH,
    LR_MODEL_PATH,
    SVM_MODEL_PATH,
    KNN_MODEL_PATH,
)


def evaluate_cnn():
    (x_train, y_train), (x_test, y_test) = data_loader.load_mnist()
    model = load_model(CNN_MODEL_PATH)
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("CNN Evaluation Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:
", classification_report(y_test, y_pred))
    print("Confusion matrix:
", confusion_matrix(y_test, y_pred))


def evaluate_classical(model_path: str):
    (x_train, y_train), (x_test, y_test) = data_loader.load_mnist(reshape_for_cnn=False)
    x_train_flat, x_test_flat = data_loader.flatten_for_classical(x_train, x_test)

    model = joblib.load(model_path)
    y_pred = model.predict(x_test_flat)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:
", classification_report(y_test, y_pred))
    print("Confusion matrix:
", confusion_matrix(y_test, y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained MNIST models')
    parser.add_argument('--model-type', type=str, default='cnn', choices=['cnn', 'lr', 'svm', 'knn'])
    args = parser.parse_args()

    if args.model_type == 'cnn':
        evaluate_cnn()
    elif args.model_type == 'lr':
        evaluate_classical(LR_MODEL_PATH)
    elif args.model_type == 'svm':
        evaluate_classical(SVM_MODEL_PATH)
    elif args.model_type == 'knn':
        evaluate_classical(KNN_MODEL_PATH)
