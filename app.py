import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf


labels = ["Miner", "No disease", "Phoma", "Rust"]

@st.cache_data
def load_model():
    model = tf.keras.models.load_model('model_20.h5',compile=False) 
    return model

model = load_model()

def predict(image):
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (256, 256))
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    return prediction

def main():
    st.title("Coffee Leaf Disease Detection App")
    st.write("Upload an image of a coffee leaf to detect the disease.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Predict!"):
            prediction = predict(image)
            predicted_class_index = np.argmax(prediction)
            predicted_class = labels[predicted_class_index]
            st.write("Prediction :", predicted_class)

if __name__ == "__main__":
    main()
