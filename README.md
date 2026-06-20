# 🤖 CUST University Query Chatbot

An intelligent NLP-powered chatbot for **Capital University of Science & Technology (CUST)** that answers student queries about admissions, fees, programs, scholarships, faculties, and campus life — through a modern web interface.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![NLP](https://img.shields.io/badge/NLP-Intent%20Detection-blue?style=flat)

---

## 📌 Overview

Students often struggle to find quick answers to university-related questions — from admission requirements to exam schedules. This chatbot provides an instant, conversational interface for university information using **semantic similarity-based intent matching** powered by Sentence Transformers.

Unlike basic keyword chatbots, this system understands the **meaning** of questions — so variations like *"how much is the fee?"* and *"what's the tuition cost?"* both get accurate answers.

---

## ✨ Features

- ✅ Semantic intent detection using **Sentence Transformers** (`all-MiniLM-L6-v2`)
- ✅ Covers **admissions, fees, programs, scholarships, HODs, faculties**, and campus life
- ✅ Modern **Flask web app** with a polished chat UI
- ✅ **Quick suggestion buttons** for common queries
- ✅ Typing indicator and smooth animations
- ✅ Cosine similarity matching for robust query understanding
- ✅ Auto-opens in browser on launch
- ✅ Graceful fallback responses for unknown queries

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Flask | Web framework & REST API |
| Sentence Transformers | Semantic similarity & intent matching |
| TensorFlow / Keras | Deep learning model backbone |
| NLTK | Text tokenization & preprocessing |
| JSON | Intent & response knowledge base |

---

## 📂 Project Structure

```
university-query-chatbot/
│
├── app.py                    # Main Flask app — chatbot logic & web UI
├── train_model.py            # Model training script
├── generate_pickle_files.py  # Generates word/class pickle files
├── intents.json              # Knowledge base — intents, patterns & responses
├── model.h5                  # Trained Keras model weights
├── words.pkl                 # Tokenized vocabulary
├── classes.pkl               # Intent class labels
└── README.md
```

---

## 🧠 How It Works

```
User Question (text input)
        │
        ▼
Sentence Transformer Encoding
(all-MiniLM-L6-v2 — converts text to semantic vector)
        │
        ▼
Cosine Similarity vs. All Patterns
(compares against pre-encoded intent patterns)
        │
        ▼
Best Match Intent (threshold: 0.65)
        │
        ├── Score ≥ 0.65 → Return intent response
        └── Score < 0.65 → Return fallback response
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/HasnainSherazi/university-query-chatbot.git
cd university-query-chatbot
```

### 2. Install dependencies

```bash
pip install flask flask-cors tensorflow sentence-transformers nltk numpy
```

### 3. Launch the chatbot

```bash
python app.py
```

> 🌐 The app auto-opens in your browser at `http://127.0.0.1:5000`

---

## 💬 Example Interactions

```
You:  What are the admission requirements for BS AI?
Bot:  To apply for BS Artificial Intelligence at CUST, you need at least
      60% marks in Matriculation and Intermediate (ICS/FSc Pre-Engineering).

You:  How much is the fee for CS?
Bot:  The fee structure for BS Computer Science at CUST is...

You:  Are there any scholarships available?
Bot:  CUST offers merit-based and need-based scholarships. Students with
      80%+ marks in intermediate are eligible for merit scholarships.
```

---

## 🖥️ UI Preview

The chatbot features a modern, responsive web interface with:
- University-branded header with CUST logo
- Quick-access buttons: **Fees · Admissions · Programs · Scholarships · Faculties**
- Smooth animated chat bubbles
- Typing indicator while processing
- Mobile-friendly responsive layout

---

## 🔮 Future Improvements

- [ ] Integrate with CUST website for **live, real-time data**
- [ ] Add **voice input** support
- [ ] Deploy as a **Telegram / WhatsApp bot**
- [ ] Expand knowledge base with more intents
- [ ] Add multilingual support (Urdu)

---

## 👤 Author

**Syed Muhammad Hasnain Sherazi**
AI Undergraduate @ CUST, Islamabad

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/syed-muhammad-hasnain-sherazi)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/HasnainSherazi)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/ssyyeedd14)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/HasnainSherazi)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
