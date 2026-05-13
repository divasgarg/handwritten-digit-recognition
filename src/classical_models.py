from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def build_logistic_regression() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')


def build_svm() -> SVC:
    # RBF kernel SVM for non-linear decision boundaries
    return SVC(kernel='rbf', gamma='scale')


def build_knn(n_neighbors: int = 5) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def train_model(model, x_train: np.ndarray, y_train: np.ndarray):
    model.fit(x_train, y_train)
    return model


def predict(model, x: np.ndarray) -> np.ndarray:
    return model.predict(x)
