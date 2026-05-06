import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border-radius: 12px;
    border: 2px solid #3b82f6;
    font-size: 18px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    border: none;
    font-weight: bold;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #7f00ff, #e100ff);
    color: white;
}

.css-1d391kg {
    background-color: #111827;
}

h1, h2, h3 {
    color: #ffffff;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load model & tokenizer
# -----------------------------
@st.cache_resource
def load_resources():
    model = load_model("next_word_model (2).h5")
    with open("tokenizer (2).pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# ⚠️ SAME AS TRAINING
max_sequence_len = 31   # change if different

# -----------------------------
# Stopwords (to reduce junk words)
# -----------------------------
STOPWORDS = {"a","the","to","of","and","in","on","is","are","was","were"}

# -----------------------------
# Repetition penalty
# -----------------------------
def apply_repetition_penalty(probs, generated_indices, penalty=1.2):
    probs = probs.copy()
    for idx in generated_indices:
        if idx < len(probs):
            probs[idx] /= penalty
    return probs

# -----------------------------
# Stopword penalty
# -----------------------------
def penalize_stopwords(probs, tokenizer, factor=0.8):
    probs = probs.copy()
    for word, idx in tokenizer.word_index.items():
        if word in STOPWORDS and idx < len(probs):
            probs[idx] *= factor
    return probs

# -----------------------------
# Sentence generation (BEST)
# -----------------------------
def generate_text(seed_text, next_words=5, temperature=0.6, k=5):
    seed_text = seed_text.lower()
    generated_indices = []

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        if len(token_list) == 0:
            return seed_text + " [unknown input]"

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len-1,
            padding='pre'
        )

        preds = model.predict(token_list, verbose=0)[0]

        # 🔥 Improve predictions
        preds = apply_repetition_penalty(preds, generated_indices, penalty=1.2)
        preds = penalize_stopwords(preds, tokenizer, factor=0.8)

        # 🔥 Top-K Sampling
        top_k_indices = preds.argsort()[-k:]
        top_k_probs = preds[top_k_indices]

        top_k_probs = np.log(top_k_probs + 1e-8) / temperature
        top_k_probs = np.exp(top_k_probs)
        top_k_probs = top_k_probs / np.sum(top_k_probs)

        predicted_index = np.random.choice(top_k_indices, p=top_k_probs)

        generated_indices.append(predicted_index)

        output_word = tokenizer.index_word.get(predicted_index, "")

        if output_word == "" or output_word == "<OOV>":
            break

        seed_text += " " + output_word

    return seed_text


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Next Word Generator", layout="centered")

st.title("🧠 Next Word Prediction")
st.markdown(
    "<h5 style='text-align: center; color: lightgray;'>Generate realistic words using trained RNN model.</h4>",
    unsafe_allow_html=True
)
user_input = st.text_area("Enter your sentence:")

num_words = st.slider("Words to generate", 1, 10, 4)
temperature = st.slider("Creativity (Temperature)", 0.3, 1.2, 0.6)
top_k = st.slider("Top-K choices", 2, 10, 5)

if st.button("Generate"):
    if user_input.strip() != "":
        result = generate_text(user_input, num_words, temperature, top_k)

        st.success("✨ Generated Sentence:")
        st.markdown(f"""
<div style="
padding:20px;
border-radius:15px;
background:#1e293b;
border:2px solid #3b82f6;
font-size:22px;
color:white;
">
✨ {result}
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text")


# -----------------------------
# Debug Panel
# -----------------------------
#with st.expander("⚙️ Debug Info"):
   # st.write("Vocabulary size:", len(tokenizer.word_index))
    #st.write("Model input shape:", model.input_shape)