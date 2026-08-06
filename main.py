import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Constants
# -----------------------------
MAX_FEATURES = 10000
MAXLEN = 500

# -----------------------------
# Load Model
# -----------------------------
model = load_model("simple_rnn_imdb.h5")

# -----------------------------
# Load IMDB Word Index
# -----------------------------
word_index = imdb.get_word_index()

# Reserve special tokens exactly like IMDB
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


# -----------------------------
# Preprocess User Review
# -----------------------------
def preprocess_text(text):

    words = text.lower().split()

    encoded = [1]  # Start token

    for word in words:
        if word in word_index and word_index[word] < MAX_FEATURES:
            encoded.append(word_index[word])
        else:
            encoded.append(2)  # Unknown word

    padded = pad_sequences(
        [encoded],
        maxlen=MAXLEN,
        padding="pre",
        truncating="pre"
    )

    return padded


# -----------------------------
# Predict Sentiment
# -----------------------------
def predict_sentiment(review):

    processed = preprocess_text(review)

    prediction = model.predict(processed, verbose=0)[0][0]

    sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"

    return sentiment, prediction


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and the trained RNN model will predict whether it is **Positive** or **Negative**."
)

review = st.text_area(
    "Movie Review",
    height=180,
    placeholder="Example: This movie was fantastic. I loved every minute of it."
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        sentiment, score = predict_sentiment(review)

        st.subheader("Prediction")

        if score >= 0.5:
            st.success(sentiment)
        else:
            st.error(sentiment)

        st.write(f"Prediction Score: **{score:.4f}**")