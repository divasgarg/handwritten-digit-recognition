import argparse
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import joblib

from . import data_loader
from .config import (
    CNN_MODEL_PATH,
    LR_MODEL_PATH,
    SVM_MODEL_PATH,
    KNN_MODEL_PATH,
)


def preprocess_image(path: str, for_cnn: bool = True):
    img = Image.open(path).convert('L').resize((28, 28))
    img_arr = np.array(img).astype('float32') / 255.0
    if for_cnn:
        img_arr = np.expand_dims(img_arr, axis=-1)
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


def main(image_path: str, model_type: str = 'cnn'):
    if model_type == 'cnn':
        model = load_model(CNN_MODEL_PATH)
        img_arr = preprocess_image(image_path, for_cnn=True)
        preds = model.predict(img_arr)
        pred_label = int(np.argmax(preds, axis=1)[0])
    else:
        # Classical models expect flattened input
        img_arr = preprocess_image(image_path, for_cnn=False)
        img_arr_flat = img_arr.reshape((img_arr.shape[0], -1))
        if model_type == 'lr':
            model = joblib.load(LR_MODEL_PATH)
        elif model_type == 'svm':
            model = joblib.load(SVM_MODEL_PATH)
        elif model_type == 'knn':
            model = joblib.load(KNN_MODEL_PATH)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        pred_label = int(model.predict(img_arr_flat)[0])

    print(f"Predicted digit: {pred_label}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict digit from image using trained MNIST models')
    parser.add_argument('--image-path', type=str, required=True)
    parser.add_argument('--model-type', type=str, default='cnn', choices=['cnn', 'lr', 'svm', 'knn'])
    args = parser.parse_args()
    main(args.image_path, args.model_type)
