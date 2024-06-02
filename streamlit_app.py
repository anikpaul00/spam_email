import streamlit as st
import base64
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import Orthogonal, GlorotUniform
import os
import tensorflow as tf

# Apply custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Title and description
st.title("Spam Email Detector")
st.markdown("### A Streamlit App with a Cyber Security Theme")

# Load a cyber security image
def load_image(file_name):
    with open(file_name, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

cyber_image = load_image("security.jpg")
st.markdown(f'<img src="data:image/jpg;base64,{cyber_image}" alt="Cyber Security" width="700">', unsafe_allow_html=True)

# Check TensorFlow version
st.write(f"TensorFlow version: {tf.__version__}")

# Paths to the model and tokenizer files
model_path = 'my_model.h5'
tokenizer_path = 'tokenizer.pkl'

# Check if the files exist
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
if not os.path.exists(tokenizer_path):
    st.error(f"Tokenizer file not found at {tokenizer_path}")

# Load the model with custom objects
custom_objects = {
    'Orthogonal': Orthogonal(),
    'GlorotUniform': GlorotUniform()
}

try:
    model = load_model(model_path, custom_objects=custom_objects)
except ValueError as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the tokenizer
try:
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    st.error(f"Tokenizer file not found at {tokenizer_path}")
    st.stop()
except pickle.UnpicklingError as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

# Function to preprocess the input text
def preprocess_text(text, tokenizer, max_len):
    text = text.lower()
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

# Streamlit app
st.title("Spam Email Detection")

email_input = st.text_area("Enter the email content:")
if st.button("Check Email"):
    if email_input:
        # Preprocess the input email content with the correct max_len
        processed_input = preprocess_text(email_input, tokenizer, max_len=500)  # Use max_len=500
        # Predict
        try:
            prediction = model.predict(processed_input)
            is_spam = (prediction > 0.5).astype("int32")[0][0]

            if is_spam:
                st.error("Warning: This email is likely spam!")
            else:
                st.success("This email seems to be safe.")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.error("Please enter email content to check.")

# Sidebar content
st.sidebar.title("About")
st.sidebar.info("""
    This is a spam email detection app.
    Themed around Cyber Security.
""")
