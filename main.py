# Import libraries
import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the trained model
model = load_model("simple_rnn_imdb.h5")


# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]  # 2 = unknown word
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded


# Function to predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input, verbose=0)

    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    return sentiment, prediction[0][0]


# ----------------------------------------------------
# Streamlit App
# ----------------------------------------------------

st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below to predict whether it is Positive or Negative."
)

# User Input
user_review = st.text_area(
    "Enter your movie review here:"
)

# Prediction Button
if st.button("Classify"):

    if user_review.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        sentiment, score = predict_sentiment(user_review)

        st.success(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.4f}")

else:
    st.info("Enter a review and click 'Classify'.")