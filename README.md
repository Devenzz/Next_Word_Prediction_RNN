# 🧠 Next Word Prediction using RNN (Deep Learning NLP Project)

An end-to-end **Deep Learning & Natural Language Processing (NLP)** project that predicts the next word in a sentence using a **Recurrent Neural Network (RNN)**.  
This project demonstrates how AI-powered text generation systems work similarly to search suggestions and smart typing applications.

---

## 🔥 Features

- 🧠 RNN / LSTM-based next word prediction  
- ✍️ Generate intelligent text predictions  
- ⚡ Real-time word generation  
- 💻 Interactive and modern Streamlit dashboard  
- 🎯 Adjustable creativity using Temperature & Top-K sampling  
- 🚀 Smooth and responsive UI experience  

---

## 🧠 Tech Stack

- **Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Frontend:** Streamlit  
- **Libraries:** NumPy, Pickle, NLTK  

---

## 📂 Project Structure

```plaintext
next-word-prediction-rnn/
│
├── app.py                          # Streamlit application
├── next_word_model.h5              # Trained RNN model
├── tokenizer.pkl                   # Saved tokenizer
├── notebook
│     └── next_word_prediction.ipynb   # Model training notebook
├── dataset
│     └── text_dataset.txt          # Training dataset
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── .gitignore                      # Ignored files
└── .gitattributes                  # Git LFS tracking
```

---

## 🚀 How It Works

1. User enters a sentence  
2. Text is tokenized and converted into sequences  
3. Input sequence is padded for the model  
4. RNN model predicts the next probable word  
5. Generated word is appended to the sentence  
6. Final generated sentence is displayed  

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Model Performance

- ✅ Validation Accuracy: **~34%**
- 📚 Vocabulary Size: **8K+ words**
- 🧠 Model Type: **RNN / LSTM**
- 🎯 Optimized using:
  - Temperature Sampling
  - Top-K Prediction
  - Repetition Penalty
  - Stopword Penalization

---

## 💡 Use Cases

- Smart typing applications  
- Search engine suggestions  
- AI chat assistants  
- Sentence auto-completion  
- NLP learning projects  
- Text generation systems  

---

## 🎯 Future Improvements

- Add Transformer / LLM architecture  
- Improve prediction accuracy with larger datasets  
- Deploy using cloud platforms  
- Add voice-to-text generation  
- Integrate multilingual support  
- Add prediction probability visualization  

---

## 🖥️ Dashboard Features

- 🎨 Modern AI-themed UI  
- 🌈 Gradient dashboard design  
- ⚙️ Adjustable model settings  
- 📈 Interactive controls  
- ✨ Real-time generated output  

---

## 👨‍💻 Author

**Devendra Gangurde**  
Aspiring Data Scientist | AI & ML Enthusiast  

## 🔗 LinkedIn

https://www.linkedin.com/in/devendra-gangurde-43620a262/

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

AI-powered intelligent text generation using Deep Learning & NLP 🚀
