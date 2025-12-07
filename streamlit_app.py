import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

st.title("UNET Segmentation – Image Upload Prediction")

# ============================
# 🔧 Load Model
# ============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model_unet.h5",
        compile=False
    )
    return model

model = load_model()

IMG_SIZE = 128

# ============================
# 🔧 Upload Image
# ============================
uploaded = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # Predict
    pred = model.predict(img_input)[0]
    pred = (pred > 0.5).astype(np.uint8) * 255

    # Resize back to original
    pred_resized = cv2.resize(pred, (img.shape[1], img.shape[0]))

    st.subheader("Prediction Mask")
    st.image(pred_resized, caption="Predicted Mask", use_column_width=True)