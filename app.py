from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS  # optional for future API use
import numpy as np
import tensorflow as tf
import random
import pickle
import json
import os
from sentence_transformers import SentenceTransformer, util  # pip install sentence-transformers
import threading
import webbrowser

app = Flask(__name__)
app.secret_key = os.urandom(24)  # for session
CORS(app)  # optional

# Load intents
with open("intents.json", encoding="utf-8") as file:
    intents_data = json.load(file)

# === Modern Matching with Sentence Transformers ===
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight & excellent

# Precompute embeddings for all patterns
all_patterns = []
intent_map = {}  # pattern index -> intent
for intent in intents_data['intents']:
    for pattern in intent['text']:
        all_patterns.append(pattern)
        intent_map[len(all_patterns)-1] = intent['intent']

pattern_embeddings = embedder.encode(all_patterns, convert_to_tensor=True)

# Fallback responses
fallback_responses = [
    "Sorry, I didn't quite catch that. Could you rephrase?",
    "I'm still learning! Try asking about admissions, fees, programs, or faculties.",
    "Not sure about that one... Try: 'fee for CS' or 'admission dates'?"
]

def get_response(user_input):
    if not user_input.strip():
        return "Please type something!"

    input_embedding = embedder.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(input_embedding, pattern_embeddings)[0]
    top_result = cos_scores.topk(1)

    best_score = top_result.values[0].item()
    best_idx = top_result.indices[0].item()

    if best_score > 0.65:  # good threshold (adjust after testing)
        best_intent = intent_map[best_idx]
        for intent in intents_data['intents']:
            if intent['intent'] == best_intent:
                return random.choice(intent['responses'])
    else:
        return random.choice(fallback_responses)

# === HTML UI - Modern & Attractive Version ===
# ==================== MODERN & IMPRESSIVE GUI ====================
chat_ui = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>🎓 CUST University Intelligent Assistant</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #1e3a8a;          /* Deep CUST navy */
      --primary-light: #3b82f6;
      --accent: #60a5fa;
      --text-dark: #111827;
      --text-light: #f3f4f6;
      --bubble-user: #1e40af;
      --bubble-bot: rgba(255,255,255,0.92);
      --blur-bg: rgba(30,58,138,0.25); /* Darker blue overlay */
    }

    * { margin:0; padding:0; box-sizing:border-box; }

    body {
      font-family: 'Poppins', system-ui, sans-serif;
      background: linear-gradient(135deg, #1e3a8a 0%, #111827 100%);
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
      min-height: 100vh;
      color: var(--text-dark);
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 15px;
    }

    /* Darker blurred campus background */
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background: url('https://images.unsplash.com/photo-1562774053-701939374585?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80') center/cover no-repeat fixed;
      background-blend-mode: overlay;
      background-color: rgba(30,58,138,0.65); /* Darker tint for contrast */
      opacity: 0.85;
      z-index: -2;
      filter: blur(4px);
    }

    .container {
      width: 100%;
      max-width: 980px;
      height: 92vh;
      max-height: 820px;
      background: var(--blur-bg);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border-radius: 28px;
      box-shadow: 0 25px 70px rgba(0,0,0,0.45);
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.15);
      display: flex;
      flex-direction: column;
    }

    .header {
      background: linear-gradient(90deg, var(--primary), var(--primary-light));
      color: white;
      padding: 16px 28px;
      display: flex;
      align-items: center;
      gap: 16px;
      flex-shrink: 0;
    }

    .header img.logo {
      height: 48px;
      width: auto;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    .header h1 {
      font-size: 1.65rem;
      font-weight: 600;
      letter-spacing: 0.6px;
    }

    #chatbox {
      flex: 1;
      overflow-y: auto;
      padding: 28px 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      background: rgba(255,255,255,0.06);
    }

    .quick-buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 0 24px 16px;
      justify-content: center;
      flex-shrink: 0;
    }

    .quick-btn {
      background: rgba(255,255,255,0.25);
      color: white;
      border: none;
      padding: 10px 18px;
      border-radius: 30px;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.3s;
      backdrop-filter: blur(6px);
    }

    .quick-btn:hover {
      background: var(--accent);
      transform: translateY(-3px);
      box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }

    .message {
      max-width: 80%;
      padding: 14px 20px;
      border-radius: 24px;
      line-height: 1.5;
      font-size: 0.98rem;
      position: relative;
      animation: fadeInUp 0.4s ease-out;
    }

    .user {
      align-self: flex-end;
      background: var(--bubble-user);
      color: white;
      border-bottom-right-radius: 6px;
    }

    .bot {
      align-self: flex-start;
      background: var(--bubble-bot);
      color: var(--text-dark);
      border-bottom-left-radius: 6px;
      box-shadow: 0 5px 15px rgba(0,0,0,0.12);
    }

    .typing {
      align-self: flex-start;
      color: #9ca3af;
      font-style: italic;
      font-size: 0.92rem;
      padding: 12px 18px;
    }

    .input-area {
      display: flex;
      padding: 18px 24px;
      background: rgba(255,255,255,0.12);
      border-top: 1px solid rgba(255,255,255,0.18);
      gap: 12px;
      flex-shrink: 0;
    }

    #userInput {
      flex: 1;
      padding: 14px 22px;
      border: none;
      border-radius: 999px;
      font-size: 1rem;
      background: rgba(255,255,255,0.28);
      color: white;
      outline: none;
      backdrop-filter: blur(10px);
      transition: all 0.3s;
    }

    #userInput::placeholder { color: rgba(255,255,255,0.75); }
    #userInput:focus {
      background: rgba(255,255,255,0.38);
      box-shadow: 0 0 0 4px rgba(59,130,246,0.35);
    }

    button.send-btn {
      padding: 0 30px;
      background: var(--primary-light);
      color: white;
      border: none;
      border-radius: 999px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s;
      box-shadow: 0 5px 15px rgba(0,0,0,0.25);
    }

    button.send-btn:hover {
      background: #2563eb;
      transform: translateY(-3px);
    }

    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(18px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
      .container { height: 96vh; border-radius: 0; max-width: none; }
      .header { padding: 14px 20px; }
      .header h1 { font-size: 1.4rem; }
      .quick-buttons { padding: 0 16px 12px; }
      #chatbox { padding: 20px 16px; }
      .input-area { padding: 14px 16px; }
      .message { max-width: 86%; font-size: 0.95rem; }
    }

    @media (max-width: 480px) {
      .quick-btn { padding: 8px 14px; font-size: 0.85rem; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <!-- CUST Logo (replace src with your official logo PNG if available) -->
      <img src="https://seeklogo.com/images/C/capital-university-of-science-technology-logo-8C5E0E6F5E-seeklogo.com.png" 
           alt="CUST Logo" class="logo" 
           onerror="this.src='https://via.placeholder.com/48x48/1e3a8a/ffffff?text=CUST';">
      <h1>CUST University Assistant</h1>
    </div>

    <!-- Quick Suggestion Buttons -->
    <div class="quick-buttons">
      <button class="quick-btn" onclick="quickAsk('fee structure')">Fees</button>
      <button class="quick-btn" onclick="quickAsk('admission dates')">Admissions</button>
      <button class="quick-btn" onclick="quickAsk('programs offered')">Programs</button>
      <button class="quick-btn" onclick="quickAsk('scholarships')">Scholarships</button>
      <button class="quick-btn" onclick="quickAsk('faculties')">Faculties</button>
    </div>

    <div id="chatbox">
      <div class="message bot">
        Welcome to CUST! 👋 I'm your smart assistant.<br>
        Ask anything about <strong>admissions, fees, programs, scholarships, HODs, faculties</strong> or campus life!
      </div>
    </div>

    <div class="input-area">
      <input type="text" id="userInput" placeholder="Ask anything (e.g., fee for BS CS, admission last date)" autocomplete="off"/>
      <button class="send-btn" onclick="sendMessage()">Send</button>
    </div>
  </div>

  <script>
    const chatbox = document.getElementById('chatbox');
    const inputField = document.getElementById('userInput');

    function appendMessage(sender, message) {
      const div = document.createElement('div');
      div.className = `message ${sender}`;
      div.innerHTML = message.replace(/\\n/g, '<br>');
      chatbox.appendChild(div);
      chatbox.scrollTop = chatbox.scrollHeight;
    }

    async function sendMessage() {
      const text = inputField.value.trim();
      if (!text) return;

      appendMessage('user', text);
      inputField.value = "";

      const typing = document.createElement('div');
      typing.className = 'message bot typing';
      typing.textContent = 'Thinking...';
      chatbox.appendChild(typing);
      chatbox.scrollTop = chatbox.scrollHeight;

      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({input: text})
        });
        const data = await res.json();
        typing.remove();
        appendMessage('bot', data.response);
      } catch (err) {
        typing.remove();
        appendMessage('bot', "Sorry, something went wrong. Try again!");
      }
    }

    function quickAsk(question) {
      inputField.value = question;
      sendMessage();
    }

    inputField.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        sendMessage();
      }
    });
  </script>
</body>
</html>
"""

# Routes
@app.route("/")
def index():
    return render_template_string(chat_ui)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        user_input = data.get("input", "").strip()
        if not user_input:
            return jsonify({"response": "Please type a question!"})

        response = get_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": "Oops! Something went wrong on our side."}), 500

@app.route("/shutdown", methods=["POST"])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        return 'Server shutdown not available', 500
    func()
    return 'Chatbot is shutting down...'

if __name__ == "__main__":
    # Auto-open browser
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000")
    threading.Timer(1.8, open_browser).start()

    app.run(host="0.0.0.0", port=5000, debug=False)  # debug=False for production feel