# Handwritten Digit Recognition (MNIST)

This project implements a full handwritten digit recognition system as described in your training report using the MNIST dataset and multiple machine learning models, including a Convolutional Neural Network (CNN).

## Features

- Uses MNIST dataset (70,000 grayscale 28x28 images of digits 0–9)
- Implements multiple models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Convolutional Neural Network (CNN)
- End-to-end pipeline:
  - Data loading and preprocessing
  - Model training
  - Evaluation (accuracy, precision, recall, F1-score, confusion matrix)
  - Model saving/loading
- Simple web app using Streamlit for digit prediction from an uploaded image

## Project Structure

```text
handwritten_digit_project/
  README.md
  requirements.txt
  src/
    __init__.py
    config.py
    data_loader.py
    classical_models.py
    cnn_model.py
    train_classical.py
    train_cnn.py
    evaluate_model.py
    predict_image.py
    app_streamlit.py
  models/
    (saved models will be stored here)
  notebooks/
    mnist_exploration.ipynb (optional placeholder)
```

## Quick Start

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the CNN model:

```bash
python -m src.train_cnn
```

4. Evaluate the trained CNN model:

```bash
python -m src.evaluate_model --model-type cnn
```

5. Run prediction on a single image (28x28 grayscale PNG):

```bash
python -m src.predict_image --image-path path/to/image.png --model-type cnn
```

6. Launch the Streamlit web app:

```bash
streamlit run src/app_streamlit.py
```

Make sure you have trained the CNN at least once so that a model file exists in the `models/` folder.

## Notes

- This code is intentionally kept clear and educational to match an academic final-year/semester training project.
- You can extend it with more layers, regularization, or hyperparameter tuning experiments to match your report discussion.
