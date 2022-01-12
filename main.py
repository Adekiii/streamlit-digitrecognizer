import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import joblib
import cv2

clf = joblib.load('classifier.joblib')

STROKE_WIDTH = 25
STROKE_COLOR = "#000000"
BG_COLOR = "#FFFFFF"

st.title("Handwritten Digit Recognition")
st.write("Trained on MNIST data using a Naive Bayes classifier")
st.caption("by Adem Kaya")

canvas_result = st_canvas(
    stroke_width=STROKE_WIDTH,
    stroke_color=STROKE_COLOR,
    background_color=BG_COLOR,
    update_streamlit=True,
    height=200,
    width=200,
    key="canvas",
)

def preprocess_image(img):
    img = np.array(np.uint8(img))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (8, 8))
    img = (255 - img).reshape(1, -1)

    return img

if st.button('Predict'):
    img = preprocess_image(canvas_result.image_data)
    prediction = clf.predict(img)
    st.write(f"Prediction: {prediction[0]}")
