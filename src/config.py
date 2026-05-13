import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Filenames for saved models
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_mnist.h5')
LR_MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression_mnist.joblib')
SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'svm_mnist.joblib')
KNN_MODEL_PATH = os.path.join(MODELS_DIR, 'knn_mnist.joblib')
