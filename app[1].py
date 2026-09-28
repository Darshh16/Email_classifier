"""
Email/SMS Spam Classifier - Web App
Run: python app.py
Then open: http://127.0.0.1:5000
"""

import pickle
import string

import nltk
from flask import Flask, render_template_string, request

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
STOPWORDS = set(stopwords.words("english"))
PUNCT = set(string.punctuation)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)


def transform_text(text: str) -> str:
    text = text.lower()
    text = nltk.word_tokenize(text)
    tokens = [t for t in text if t.isalnum()]
    tokens = [t for t in tokens if t not in STOPWORDS and t not in PUNCT]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# HTML template (single page, embedded CSS)
# ---------------------------------------------------------------------------
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spam Email Classifier</title>
<style>
  :root {
    --bg1: #0f172a;
    --bg2: #1e293b;
    --accent: #6366f1;
    --accent2: #ec4899;
    --spam: #ef4444;
    --ham: #22c55e;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --card: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.12);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: radial-gradient(circle at 20% 20%, #1e1b4b 0%, var(--bg1) 45%, #020617 100%);
    color: var(--text);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }
  .card {
    width: 100%;
    max-width: 640px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 40px;
    backdrop-filter: blur(12px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
  }
  .badge {
    display: inline-block;
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 14px;
  }
  h1 {
    margin: 0 0 6px 0;
    font-size: 28px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  p.sub { color: var(--muted); margin: 0 0 28px 0; font-size: 14px; }
  textarea {
    width: 100%;
    min-height: 160px;
    resize: vertical;
    background: rgba(2,6,23,0.5);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    font-size: 15px;
    line-height: 1.5;
    font-family: inherit;
    outline: none;
    transition: border-color .2s;
  }
  textarea:focus { border-color: var(--accent); }
  button {
    margin-top: 16px;
    width: 100%;
    padding: 14px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: white;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    transition: transform .15s, opacity .15s;
  }
  button:hover { transform: translateY(-1px); opacity: 0.92; }
  .result {
    margin-top: 26px;
    padding: 18px 20px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 12px;
    animation: fadeIn .35s ease;
  }
  .result.spam { background: rgba(239,68,68,0.14); border: 1px solid rgba(239,68,68,0.4); color: #fca5a5; }
  .result.ham  { background: rgba(34,197,94,0.14); border: 1px solid rgba(34,197,94,0.4); color: #86efac; }
  .dot { width: 12px; height: 12px; border-radius: 50%; }
  .dot.spam { background: var(--spam); box-shadow: 0 0 10px var(--spam); }
  .dot.ham { background: var(--ham); box-shadow: 0 0 10px var(--ham); }
  .footer { margin-top: 24px; font-size: 12px; color: var(--muted); text-align: center; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
</style>
</head>
<body>
  <div class="card">
    <span class="badge">Machine Learning</span>
    <h1>Spam Email Classifier</h1>
    <p class="sub">Paste an email or SMS below. TF-IDF + Multinomial Naive Bayes decides if it's spam.</p>

    <form method="POST">
      <textarea name="email_text" placeholder="Paste the email/message text here...">{{ email_text or '' }}</textarea>
      <button type="submit">Check Message</button>
    </form>

    {% if prediction is not none %}
      <div class="result {{ 'spam' if prediction == 1 else 'ham' }}">
        <span class="dot {{ 'spam' if prediction == 1 else 'ham' }}"></span>
        {% if prediction == 1 %}
          🚫 This looks like SPAM
        {% else %}
          ✅ This looks like a NORMAL (ham) email
        {% endif %}
      </div>
    {% endif %}

    <div class="footer">TF-IDF Vectorizer &middot; Multinomial Naive Bayes &middot; Flask</div>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    email_text = ""
    if request.method == "POST":
        email_text = request.form.get("email_text", "")
        if email_text.strip():
            transformed = transform_text(email_text)
            vector = tfidf.transform([transformed])
            prediction = int(model.predict(vector)[0])  # 1 = spam, 0 = ham
    return render_template_string(PAGE, prediction=prediction, email_text=email_text)


if __name__ == "__main__":
    app.run(debug=True)
