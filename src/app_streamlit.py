import os
import sys
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

# Ensure project root is on sys.path so we can import src.config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import CNN_MODEL_PATH


st.title("Handwritten Digit Recognition (MNIST)")
st.write(
    "Upload a 28x28 grayscale image of a handwritten digit, or any small digit "
    "image (it will be resized)."
)

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show the image (scaled up for display)
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(img.resize((140, 140)), caption="Input image (resized)", use_column_width=False)

    # Preprocess for CNN
    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=-1)  # (28, 28, 1)
    img_arr = np.expand_dims(img_arr, axis=0)   # (1, 28, 28, 1)

    try:
        model = load_model(CNN_MODEL_PATH)
    except Exception as e:
        st.error(
            "Could not load CNN model. Train the model first locally and push "
            "models/cnn_mnist.h5 to GitHub."
        )
        st.text(str(e))
    else:
        preds = model.predict(img_arr)
        pred_label = int(np.argmax(preds, axis=1)[0])

        st.subheader(f"Predicted digit: {pred_label}")
        st.bar_chart(preds[0])