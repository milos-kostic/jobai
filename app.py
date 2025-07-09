from flask import Flask, request, render_template, jsonify
from io import BytesIO
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np
import pickle

app = Flask(__name__)

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_cv(file_storage):
    filename = file_storage.filename.lower()
    file_bytes = file_storage.read()
    file_storage.seek(0)
    content = ""
    if filename.endswith('.pdf'):
        try:
            pdf_stream = BytesIO(file_bytes)
            content = extract_pdf_text(pdf_stream)
        except:
            content = ""
    elif filename.endswith('.docx'):
        try:
            doc = Document(BytesIO(file_bytes))
            content = '\n'.join([p.text for p in doc.paragraphs])
        except:
            content = ""
    elif filename.endswith('.txt'):
        try:
            content = file_bytes.decode('utf-8', errors='ignore')
        except:
            content = ""
    return content or ""

def get_jobs_with_embeddings():
    conn = sqlite3.connect('jobs.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, description, url, embedding FROM jobs')
    rows = cursor.fetchall()
    conn.close()
    jobs = []
    for r in rows:
        emb = pickle.loads(r[4]) if r[4] else None
        jobs.append({
            'id': r[0],
            'title': r[1],
            'description': r[2],
            'url': r[3],
            'embedding': emb
        })
    return jobs

def save_feedback(job_id, feedback):
    conn = sqlite3.connect('jobs.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO feedback (job_id, feedback) VALUES (?, ?)', (job_id, feedback))
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
            if job['embedding'] is None:
                continue
            score = np.dot(cv_embedding, job['embedding']) / (np.linalg.norm(cv_embedding)*np.linalg.norm(job['embedding']))
            results.append((job, score))

        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:5]

        return render_template('index.html', results=top_results)

    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    job_id = data.get('job_id')
    fb = data.get('feedback')
    if job_id is None or fb not in [0, 1]:
        return jsonify({'error': 'Invalid feedback data'}), 400

    save_feedback(job_id, fb)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
