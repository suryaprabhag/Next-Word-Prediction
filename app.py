
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        BASE_DIR / "next_word_lstm.keras",
        compile=False
    )

    with open(BASE_DIR / "tokenizer.json", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    with open(BASE_DIR / "model_config.json", encoding="utf-8") as f:
        config = json.load(f)

    return model, tokenizer, config


try:
    model, tokenizer, config = load_artifacts()
except Exception:
    st.error("Unable to load model artifacts.")
    st.stop()


context_length = config["context_length"]
vocab_size = config["vocab_size"]

index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
    if index < vocab_size
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_words(text, top_n=5):
    cleaned = clean_text(text)

    if not cleaned:
        return []

    tokens = tokenizer.texts_to_sequences([cleaned])[0]

    if not tokens:
        return []

    sequence = pad_sequences(
        [tokens],
        maxlen=context_length,
        padding="pre",
        truncating="pre"
    )

    probabilities = model.predict(
        sequence,
        verbose=0
    )[0].copy()

    # Exclude padding and unknown-word tokens
    probabilities[:2] = 0

    top_ids = np.argsort(probabilities)[-top_n:][::-1]

    return [
        {
            "word": index_to_word.get(int(idx), "<OOV>"),
            "probability": float(probabilities[idx])
        }
        for idx in top_ids
    ]


def generate_text(seed_text, num_words):
    output = seed_text.strip()

    for _ in range(num_words):
        predictions = predict_words(output, top_n=1)

        if not predictions:
            break

        output += " " + predictions[0]["word"]

    return output


st.title("🧠 Next Word Predictor")

st.markdown(
    """
    **Deep Learning NLP Project — Innomatics Research Labs**

    Explore how a Long Short-Term Memory (LSTM)
    model predicts the next word in a sentence.

    Trained on *Alice's Adventures in Wonderland*
    by Lewis Carroll.
    """
)

st.divider()

user_text = st.text_area(
    "Enter a partial sentence",
    value="the white rabbit",
    height=120,
    max_chars=1000
)

top_n = st.slider(
    "Number of suggestions",
    min_value=1,
    max_value=10,
    value=5
)

if st.button(
    "Predict Next Words",
    type="primary",
    use_container_width=True
):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        results = predict_words(user_text, top_n)

        if results:
            st.subheader("Predicted Next Words")

            df = pd.DataFrame({
                "Word": [item["word"] for item in results],
                "Probability (%)": [
                    round(item["probability"] * 100, 2)
                    for item in results
                ]
            })

            st.dataframe(df, hide_index=True)

            st.bar_chart(
                df.set_index("Word")["Probability (%)"]
            )

            st.success(
                f"Most likely next word: **{results[0]['word']}**"
            )
        else:
            st.warning("Unable to predict from this input.")

st.divider()

st.subheader("Generate Text")

num_words = st.slider(
    "Words to generate",
    min_value=1,
    max_value=20,
    value=5
)

if st.button("Generate Continuation", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        generated = generate_text(user_text, num_words)
        st.info(generated)

st.divider()

st.caption(
    "Model: LSTM | Vocabulary: 2,299 | "
    "Top-1 Validation Accuracy: 13.92% | "
    "Top-5 Validation Accuracy: 29.87%"
)

st.caption(
    "Educational NLP demonstration. Predictions may be "
    "grammatically incorrect or contextually inaccurate."
)
