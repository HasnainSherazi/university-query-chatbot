# 🤖 University Query Chatbot

An NLP-powered chatbot designed to answer student queries about university policies, departments, schedules, and general campus information.

---

## 📌 Overview

Students often struggle to find quick answers about university-related topics — from admission requirements to exam schedules. This chatbot provides an instant, conversational interface for university information, reducing the burden on administrative staff and giving students fast, accurate responses.

---

## 🎯 Features

- ✅ Answers university-related queries in natural language
- ✅ Covers admissions, courses, departments, policies, and more
- ✅ Intent detection and entity extraction
- ✅ Easily extendable knowledge base

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| NLP (NLTK / spaCy) | Text processing & intent detection |
| JSON / CSV | Knowledge base / FAQ store |

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/HasnainSherazi/university-query-chatbot.git
cd university-query-chatbot

# Install dependencies
pip install nltk

# Run the chatbot
python chatbot.py
```

---

## 📁 Project Structure

```
university-query-chatbot/
│
├── chatbot.py              # Main chatbot script
├── intents.json            # Intent definitions and responses
├── training.py             # Model training script
├── model/                  # Saved model
└── README.md
```

---

## 💬 Example Interaction

```
You:  What are the admission requirements for BS AI?
Bot:  To apply for BS Artificial Intelligence at CUST, you need 
      at least 60% marks in Matriculation and Intermediate (ICS/FSc Pre-Engineering).

You:  When does the semester start?
Bot:  The Spring semester typically begins in February and Fall in September.
```

---

## 🔮 Future Improvements

- [ ] Integrate with university website for live data
- [ ] Add voice input support
- [ ] Deploy as a Telegram / WhatsApp bot

---

## 👤 Author

**Syed Muhammad Hasnain Sherazi**  
AI Undergraduate @ CUST  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/syed-muhammad-hasnain-sherazi)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/ssyyeedd14)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD21E?style=flat)](https://huggingface.co/HasnainSherazi)
