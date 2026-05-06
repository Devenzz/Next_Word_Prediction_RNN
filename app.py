import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load model & tokenizer
# -----------------------------
@st.cache_resource
def load_resources():
    model = load_model("next_word_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# ⚠️ SAME AS TRAINING
max_sequence_len = 20   # 🔁 change if different

# -----------------------------
# 🔥 Top-K Sampling Function
# -----------------------------
def top_k_sampling(preds, k=5, temperature=0.6):
    preds = np.asarray(preds).astype("float64")

    top_k_indices = preds.argsort()[-k:]
    top_k_probs = preds[top_k_indices]

    top_k_probs = np.log(top_k_probs + 1e-8) / temperature
    top_k_probs = np.exp(top_k_probs)
    top_k_probs = top_k_probs / np.sum(top_k_probs)

    return np.random.choice(top_k_indices, p=top_k_probs)


# -----------------------------
# Sentence Generation Function
# -----------------------------
def generate_text(seed_text, next_words=5, temperature=0.6, k=5):
    seed_text = seed_text.lower()

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        if len(token_list) == 0:
            return seed_text + " [unknown input]"

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len-1,
            padding='pre'
        )

        predicted_probs = model.predict(token_list, verbose=0)[0]

        # 🔥 FIXED SAMPLING
        predicted_index = top_k_sampling(predicted_probs, k=k, temperature=temperature)

        output_word = tokenizer.index_word.get(predicted_index, "")

        if output_word == "" or output_word == "<OOV>":
            break

        seed_text += " " + output_word

    return seed_text


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Next Word Generator", layout="centered")

st.title("🧠 Next Word Sentence Generator")
st.write("Generate realistic next words using your trained RNN model")

user_input = st.text_area("Enter your sentence:")

num_words = st.slider("Number of words to generate", 1, 10, 4)

temperature = st.slider("Creativity (Temperature)", 0.3, 1.2, 0.6)

top_k = st.slider("Top-K choices", 2, 10, 5)

if st.button("Generate"):
    if user_input.strip() != "":
        result = generate_text(user_input, num_words, temperature, top_k)

        st.success("✨ Generated Sentence:")
        st.write(result)
    else:
        st.warning("Please enter some text")


# -----------------------------
# Debug (optional)
# -----------------------------
with st.expander("⚙️ Debug Info"):
    st.write("Vocabulary size:", len(tokenizer.word_index))
    st.write("Model input shape:", model.input_shape)