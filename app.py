from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
import pickle
import logging
import re

app = Flask(__name__)
app.secret_key = 'your_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

model = SentenceTransformer('all-MiniLM-L6-v2')

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('jobs.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, username FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return User(user[0], user[1]) if user else None

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

def keyword_overlap_score(cv_text, job_text):
    cv_words = set(clean_text(cv_text).split())
    job_words = set(clean_text(job_text).split())
    return len(cv_words & job_words) / max(len(job_words), 1)

def extract_text_from_cv(file_storage):
    filename = file_storage.filename.lower()
    file_bytes = file_storage.read()
    file_storage.seek(0)
    if filename.endswith('.pdf'):
        return extract_pdf_text(BytesIO(file_bytes))
    elif filename.endswith('.docx'):
        doc = Document(BytesIO(file_bytes))
        return '\n'.join([p.text for p in doc.paragraphs])
    elif filename.endswith('.txt'):
        return file_bytes.decode('utf-8', errors='ignore')
    return ""

def get_jobs_with_embeddings():
    conn = sqlite3.connect('jobs.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, description, url, embedding FROM jobs')
    rows = cursor.fetchall()
    conn.close()
    jobs = []
    for r in rows:
        emb = pickle.loads(r[4]) if r[4] else None
        jobs.append({'id': r[0], 'title': r[1], 'description': r[2], 'url': r[3], 'embedding': emb})
    return jobs

def save_feedback(job_id, feedback):
    conn = sqlite3.connect('jobs.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO feedback (user_id, job_id, feedback) VALUES (?, ?, ?)',
                   (current_user.id, job_id, feedback))
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        cv_file = request.files.get('cvfile')
        if not cv_file:
            return "No file uploaded", 400

        cv_text = extract_text_from_cv(cv_file)
        if not cv_text.strip():
            return "Could not extract text from CV", 400

        cv_embedding = model.encode([cv_text])[0]
        jobs = get_jobs_with_embeddings()

        results = []
        for job in jobs:
            try:
                if job['embedding'] is None:
                    continue
                job_text = clean_text(f"{job['title']} {job['description']}")
                job_emb_norm = job['embedding'] / (np.linalg.norm(job['embedding']) + 1e-10)
                sim_score = np.dot(cv_embedding, job_emb_norm)
                keyword_score = keyword_overlap_score(cv_text, job_text)
                hybrid_score = 0.8 * sim_score + 0.2 * keyword_score

                if hybrid_score > 0.3:
                    explanation = {
                        'semantic_similarity': round(sim_score, 3),
                        'keyword_overlap': round(keyword_score, 3),
                        'score': round(hybrid_score, 3)
                    }
                    results.append((job, explanation))
            except Exception as e:
                logging.warning(f"Job match error: {e}")

        results.sort(key=lambda x: x[1]['score'], reverse=True)
        top_results = results[:5]
        return render_template('index.html', results=top_results)

    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    data = request.get_json()
    job_id = data.get('job_id')
    fb = data.get('feedback')
    if job_id is None or fb not in [0, 1]:
        return jsonify({'error': 'Invalid feedback data'}), 400
    save_feedback(job_id, fb)
    return jsonify({'status': 'success'})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        conn = sqlite3.connect('jobs.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            return 'Username already exists'
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('jobs.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, password FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        if result and check_password_hash(result[1], password):
            user = User(result[0], username)
            login_user(user)
            return redirect(url_for('index'))
        return 'Invalid credentials'
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))  # Redirect to public main page after logout

if __name__ == '__main__':
    app.run(debug=True)
