# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from flask_mysqldb import MySQL
import random
import re
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import timedelta, datetime
from transformers import pipeline
from fpdf import FPDF
from config import GEMINI_API_KEY
import google.generativeai as genai

# ---- Transformers / Torch ----
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- Flask app init ----
app = Flask(__name__)
app.config.from_pyfile('config.py')   # expects MYSQL_* and SECRET_KEY, MYSQL_CURSORCLASS='DictCursor'
mysql = MySQL(app)
genai.configure(api_key=GEMINI_API_KEY)

# ---- Load emotion model once at startup ----
# Model: nateraw/bert-base-uncased-emotion (light and fine for prototyping)
MODEL_NAME = "nateraw/bert-base-uncased-emotion"
print("Loading emotion model... (this may take a minute the first time)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
id2label = model.config.id2label  # mapping int -> label
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e", device=-1)

def predict_emotion(text: str) -> str:
    """Return single predicted emotion label (string)."""
    # basic safety for empty text
    text = (text or "").strip()
    if not text:
        return "neutral"
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        idx = int(torch.argmax(probs, dim=-1).item())
        label = id2label.get(idx, "neutral")
        return label.lower()

# ---- Crisis detection (keyword + regex based) ----
CRISIS_PATTERNS = [
    r"\bkill myself\b", r"\bi want to die\b", r"\bend my life\b",
    r"\bsuicid(e|al)\b", r"\bcut myself\b", r"\bhurting myself\b",
    r"\bi can't go on\b", r"\bi cant go on\b", r"\bwant to die\b"
]
CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), flags=re.IGNORECASE)

def detect_crisis(text: str) -> bool:
    if not text: 
        return False
    return bool(CRISIS_RE.search(text))

CRISIS_SAMPLES = [
    "I want to die",
    "I can't go on",
    "life is not worth it",
    "I am hopeless",
    "I feel suicidal",
    "I want to end it all",
    "no reason to live",
    "I can't handle this pain"
]

def semantic_crisis_risk_score(text):
    if not text:
        return 0.0
    text_emb = semantic_model.encode(text, convert_to_tensor=True)
    crisis_embs = semantic_model.encode(CRISIS_SAMPLES, convert_to_tensor=True)
    cosine_scores = util.cos_sim(text_emb, crisis_embs)
    return torch.max(cosine_scores).item()

def detect_multimodal_crisis(df):
    df['date'] = pd.to_datetime(df['date'])
    daily_emotion_counts = df.groupby(['date', 'predicted_emotion']).size().unstack(fill_value=0)
    daily_totals = daily_emotion_counts.sum(axis=1)
    roll_std = daily_totals.rolling(window=3).std()
    high_volatility_days = roll_std[roll_std > roll_std.quantile(0.75)].index
    recent_crises = df[(df['crisis_flag'] == 1) & (df['date'] >= df['date'].max() - timedelta(days=3))]
    crisis_signal = False
    if len(high_volatility_days) > 0 or len(recent_crises) > 2:
        crisis_signal = True
    return crisis_signal

# ---- Simple empathetic response templates per emotion ----
EMPATHETIC_TEMPLATES = {
    "anger": [
        "I can hear you're really upset. Do you want to tell me more about what's making you angry?",
        "Anger is a valid feeling. Would talking through it help right now?"
    ],
    "joy": [
        "That's lovely to hear — want to share what's making you feel good?",
        "I'm glad you're feeling this way! Celebrate that moment."
    ],
    "sadness": [
        "I'm sorry you're feeling sad. I'm here to listen — tell me more if you want.",
        "It makes sense to feel down sometimes. Would you like a grounding exercise?"
    ],
    "fear": [
        "Feeling scared can be overwhelming. Try taking a few slow breaths with me.",
        "You're not alone — can you tell me what's making you feel afraid?"
    ],
    "love": [
        "That's sweet — it's nice to feel connected.",
        "Love is powerful. Hold on to that feeling—would you like to talk about it?"
    ],
    "surprise": [
        "That sounds surprising. Would you like to reflect on it together?",
        "Wow — that must have been unexpected. Tell me more."
    ],
    "neutral": [
        "I hear you. Want to share more?",
        "I'm here to listen whenever you're ready."
    ]
}

COPING_STRATEGIES = {
    "anger": [
        "Try deep breathing exercises to calm down.",
        "Take a short walk to release tension."
    ],
    "joy": [
        "Share your joy with a friend or family member.",
        "Take a moment to savor this positive feeling."
    ],
    "sadness": [
        "Write down your feelings in a journal.",
        "Try a grounding exercise, focusing on your surroundings."
    ],
    "fear": [
        "Practice mindfulness or meditation to ease fear.",
        "Talk to someone you trust about what scares you."
    ],
    "love": [
        "Connect with a loved one and express gratitude.",
        "Spend time doing something nurturing for yourself."
    ],
    "surprise": [
        "Take a deep breath and reflect on the surprise positively.",
        "Allow yourself time to process unexpected events."
    ],
    "neutral": [
        "Take a moment to check in with your body and mind.",
        "Try a relaxing activity you enjoy."
    ]
}

def generate_response(user_message: str, predicted_emotion: str) -> str:
    templates = EMPATHETIC_TEMPLATES.get(predicted_emotion.lower(), EMPATHETIC_TEMPLATES["neutral"])
    base_response = random.choice(templates)
    coping_tips = COPING_STRATEGIES.get(predicted_emotion.lower(), [])
    if coping_tips:
        coping = random.choice(coping_tips)
        base_response += f" Also, you might find this helpful: {coping}"
    return base_response

def clean_text(text):
    text = re.sub(r'[#*]+', '', text)
    text = text.strip()
    text = re.sub(r'\n\n+', '</p><p>', text)
    text = text.replace("\n", "<br>")
    return f"<p>{text}<p>"

# ---- Helpers to get current user's id ----
def get_current_user_id():
    if 'username' not in session:
        return None
    cur = mysql.connection.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (session['username'],))
    row = cur.fetchone()
    cur.close()
    if row:
        # when using DictCursor this will be a dict; else tuple
        return int(row['id'])
    return None

def summarize_text(text):
    if not text.strip():
        return "No conversation data available for this period."
    text = text.replace("\n", " ")
    try:
        summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error creating summary: {e}"
    
# ---- Routes ----
@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

# ---- Use your existing signup/login logic --- ensure session['username'] is set on login
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close()
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for('login'))
        cur.execute("INSERT INTO users (username, email, password) VALUES (%s,%s,%s)", (username,email,password))
        mysql.connection.commit()
        cur.close()
        flash("Signup successful! Login now.", "success")
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        cur = mysql.connection.cursor()
        cur.execute("SELECT username, password FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        cur.close()
        if row:
            stored_password = row['password'] if isinstance(row, dict) else row[1]
            username = row['username'] if isinstance(row, dict) else row[0]
            if stored_password == password:
                session['username'] = username
                session['login_time'] = datetime.utcnow().isoformat()
                flash("Logged in successfully.", "success")
                return redirect(url_for('chat'))
        flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('login_time', None)
    flash("Logged out", "info")
    return redirect(url_for('login'))

# ---- Chat route: GET shows page + history, POST processes user msg ----
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'username' not in session:
        return redirect(url_for('login'))

    def fetch_conversation_history(user_id, limit=10):
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT user_message, bot_response FROM conversations
            WHERE user_id = %s ORDER BY timestamp DESC LIMIT %s
        """, (user_id, limit))
        rows = cur.fetchall()
        cur.close()
        return rows[::-1]  # chronological order

    def build_gpt_prompt(history, current_user_message):
        prompt = "You are a helpful and empathetic mental health chatbot.\n"
        for user_msg, bot_msg in history:
            prompt += f"User: {user_msg}\nBot: {bot_msg}\n"
        prompt += f"User: {current_user_message}\nBot:"
        return prompt

    def get_gemini_response_with_context(prompt):
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if not text:
                return None
            return text
        except Exception as e:
            app.logger.error("Gemini AI API error: %s", str(e))
            return None

    if request.method == 'POST':
        user_message = request.form.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        try:
            emotion = predict_emotion(user_message)
        except:
            app.logger.error("Emotion prediction error.")
            emotion = "neutral"

        regex_crisis = detect_crisis(user_message)
        risk_score = semantic_crisis_risk_score(user_message.lower())

        crisis = False
        risk_level = 'Low'
        if regex_crisis or risk_score > 0.65:
            crisis = True
            risk_level = 'High' if risk_score > 0.8 else 'Medium'

        user_id = get_current_user_id()
        if user_id is None:
            return jsonify({'error': 'User not in session.'}), 403

        history = fetch_conversation_history(user_id, limit=10)
        prompt = build_gpt_prompt(history, user_message)
        bot_reply = get_gemini_response_with_context(prompt)

        if not bot_reply:
            # fallback to existing response function
            bot_reply = generate_response(user_message, emotion)

        bot_reply = clean_text(bot_reply)

        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO conversations (user_id, user_message, bot_response, predicted_emotion, crisis_flag) VALUES (%s,%s,%s,%s,%s)",
            (user_id, user_message, bot_reply, emotion, int(crisis))
        )
        mysql.connection.commit()
        cur.close()

        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT predicted_emotion, DATE(timestamp) as date, crisis_flag FROM conversations
            WHERE user_id = %s ORDER BY timestamp DESC LIMIT 50
        """, (user_id,))
        recent_rows = cur.fetchall()
        cur.close()

        df_recent = pd.DataFrame(recent_rows, columns=['predicted_emotion', 'date', 'crisis_flag'])
        multimodal_crisis = detect_multimodal_crisis(df_recent)

        return jsonify({'user': user_message, 'bot': bot_reply, 'emotion': emotion,
                        'crisis': crisis or multimodal_crisis, 'risk_level': risk_level})

    # GET - fetch chat history only from current login time onwards
    user_id = get_current_user_id()
    login_time = session.get('login_time')

    query = """
        SELECT user_message, bot_response, predicted_emotion, crisis_flag, timestamp
        FROM conversations
        WHERE user_id = %s
    """
    params = [user_id]
    if login_time:
        query += " AND timestamp >= %s "
        params.append(login_time)
    query += " ORDER BY timestamp ASC LIMIT 500"

    cur = mysql.connection.cursor()
    cur.execute(query, params)
    history = cur.fetchall()
    cur.close()

    return render_template('chat.html', history=history)

# ---- Mood dashboard route ----
@app.route('/mood-dashboard')
def mood_dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    userid = get_current_user_id()
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT predicted_emotion, DATE(timestamp) as date, HOUR(timestamp) as hour, crisis_flag
        FROM conversations WHERE user_id = %s ORDER BY timestamp ASC
    """, (userid,))
    rows = cur.fetchall()
    cur.close()

    df = pd.DataFrame(rows, columns=['predicted_emotion', 'date', 'hour', 'crisis_flag'])
    if df.empty:
        return render_template("mood_dashboard.html", data={}, day_labels=[], emotion_labels=[], counts=[], pie_data={}, 
                               weekly_labels=[], weekly_data=[], hourly_heatmap_data=[], crisis_messages=0, crisis_percentage=0, 
                               top_emotions={}, change_summary={}, crisis_timeline_labels=[], crisis_timeline_data=[])

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.set_index('date', inplace=True)

    pie_data = df['predicted_emotion'].value_counts().to_dict()

    daily_counts = df.groupby(['date', 'predicted_emotion']).size().unstack(fill_value=0)
    if isinstance(daily_counts.index, pd.DatetimeIndex):
        day_labels = [d.strftime("%Y-%m-%d") for d in daily_counts.index]
    else:
        day_labels = daily_counts.index.astype(str).tolist()
    emotion_labels = daily_counts.columns.tolist()
    counts = daily_counts.values.tolist()

    # Fix FutureWarning by grouping with pd.Grouper for weekly data
    weekly_emotions = df.groupby([pd.Grouper(freq='W'), 'predicted_emotion']).size().unstack(fill_value=0)
    weekly_labels = weekly_emotions.index.strftime('%Y-%m-%d').tolist()
    weekly_data = weekly_emotions.values.tolist()

    hourly = df.groupby(['hour', 'predicted_emotion']).size().unstack(fill_value=0)
    hour_labels = list(range(24))
    hourly_heatmap_data = []
    for emotion in emotion_labels:
        if emotion in hourly.columns:
            hourly_heatmap_data.append([int(hourly.loc[h, emotion]) if h in hourly.index else 0 for h in hour_labels])
        else:
            hourly_heatmap_data.append([0]*24)

    total_messages = len(df)
    crisis_messages = int(df['crisis_flag'].sum())
    crisis_percentage = round(crisis_messages / total_messages * 100, 2) if total_messages > 0 else 0

    top_emotions = df['predicted_emotion'].value_counts().head(3).to_dict()

    recent_df = df[df.index >= (df.index.max() - pd.Timedelta(weeks=4))]
    prev_df = df[df.index < (df.index.max() - pd.Timedelta(weeks=4))]
    recent_counts = recent_df['predicted_emotion'].value_counts()
    prev_counts = prev_df['predicted_emotion'].value_counts()
    change_summary = {}
    all_emotions = set(recent_counts.index).union(set(prev_counts.index))
    for emo in all_emotions:
        recent_count = recent_counts.get(emo, 0)
        prev_count = prev_counts.get(emo, 0)
        change = recent_count - prev_count
        if change > 0:
            change_summary[emo] = f"Increase by {change}"
        elif change < 0:
            change_summary[emo] = f"Decrease by {-change}"
        else:
            change_summary[emo] = "No change"

    # Safe crisis timeline data ensuring no serialization errors
    crisis_timeline = df.groupby('date')['crisis_flag'].sum()
    if not crisis_timeline.empty:
        crisis_timeline_labels = crisis_timeline.index.strftime('%Y-%m-%d').tolist()
        crisis_timeline_data = crisis_timeline.values.tolist()
    else:
        crisis_timeline_labels = []
        crisis_timeline_data = []

    return render_template("mood_dashboard.html",
                           data={'day_labels': day_labels,
                                 'emotion_labels': emotion_labels,
                                 'counts': counts,
                                 'pie_data': pie_data,
                                 'crisis_data': {str(k.date() if hasattr(k, "date") else k): int(v)
                                                 for k, v in df[df['crisis_flag'] == 1].groupby('date').size().to_dict().items()}},
                           weekly_labels=weekly_labels,
                           weekly_data=weekly_data,
                           hourly_heatmap_data=hourly_heatmap_data,
                           crisis_messages=crisis_messages,
                           crisis_percentage=crisis_percentage,
                           top_emotions=top_emotions,
                           change_summary=change_summary,
                           crisis_timeline_labels=crisis_timeline_labels,
                           crisis_timeline_data=crisis_timeline_data)

@app.route('/summary')
def summary():
    if 'username' not in session:
        return redirect(url_for('login'))

    user_id = get_current_user_id()
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT user_message, bot_response, timestamp FROM conversations
        WHERE user_id = %s ORDER BY timestamp ASC
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        daily_summary = "No conversations yet."
        weekly_summary = "No conversations yet."
        return render_template("summary.html",
                               daily_summary=daily_summary,
                               weekly_summary=weekly_summary)

    df = pd.DataFrame(rows, columns=['user_message', 'bot_response', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Daily summary
    today = pd.Timestamp.now().normalize()
    daily_data = df[df['timestamp'].dt.date == today.date()]
    daily_text = " ".join(f"User: {row.user_message} Bot: {row.bot_response}" for row in daily_data.itertuples())
    daily_summary = summarize_text(daily_text)

    # Weekly summary
    one_week_ago = today - pd.Timedelta(days=7)
    weekly_data = df[df['timestamp'] >= one_week_ago]
    weekly_text = " ".join(f"User: {row.user_message} Bot: {row.bot_response}" for row in weekly_data.itertuples())
    weekly_summary = summarize_text(weekly_text)

    return render_template("summary.html",
                           daily_summary=daily_summary,
                           weekly_summary=weekly_summary)

@app.route('/export-summary')
def export_summary():
    if 'username' not in session:
        return redirect(url_for('login'))

    user_id = get_current_user_id()
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT user_message, bot_response, timestamp FROM conversations
        WHERE user_id = %s ORDER BY timestamp ASC
    """, (user_id,))
    rows = cur.fetchall()
    cur.close()

    if not rows:
        return "No chats available for export."

    df = pd.DataFrame(rows, columns=['user_message', 'bot_response', 'timestamp'])
    combined_text = " ".join(f"User: {row.user_message} Bot: {row.bot_response}" for row in df.itertuples())
    summary_txt = summarize_text(combined_text)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Conversation Summary Report", ln=True, align="C")
    pdf.multi_cell(0, 10, summary_txt)
    pdf.output("conversation_summary.pdf")

    return send_file("conversation_summary.pdf", as_attachment=True)

@app.route('/contact')
def contact():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        flash('Thank you, your message has been received!', 'success')

        return redirect(url_for('contact'))
    
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

# ---- Run ----
if __name__ == '__main__':
    # ensure secret key is set (if you used config.py with SECRET_KEY)
    if not app.secret_key:
        app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.run(debug=True)
