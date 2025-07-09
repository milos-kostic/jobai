import sqlite3
from sentence_transformers import SentenceTransformer
import pickle

# Connect to DB
conn = sqlite3.connect('jobs.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    description TEXT,
    url TEXT,
    embedding BLOB
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    feedback INTEGER,  -- 1=like, 0=dislike
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

# Example jobs data
jobs_data = [
    {
        "title": "Java Developer",
        "description": "Looking for Java developer experienced in Spring Framework and REST APIs.",
        "url": "https://example.com/job/java-developer"
    },
    {
        "title": "Electronics Technician",
        "description": "Maintenance and repair of electronic equipment and surveillance systems.",
        "url": "https://example.com/job/electronics-tech"
    },
    {
        "title": "IT Administrator",
        "description": "Manage and maintain IT infrastructure, networks, and security.",
        "url": "https://example.com/job/it-admin"
    },
    {
        "title": "C++ Linux Programmer",
        "description": "Develop system-level software using C++ on Linux environment.",
        "url": "https://example.com/job/cpp-linux-dev"
    }
]

model = SentenceTransformer('all-MiniLM-L6-v2')

# Insert jobs and their embeddings
for job in jobs_data:
    cursor.execute('INSERT INTO jobs (title, description, url) VALUES (?, ?, ?)',
                   (job["title"], job["description"], job["url"]))
    job_id = cursor.lastrowid
    embedding = model.encode([job["description"]])[0]
    blob = pickle.dumps(embedding)
    cursor.execute('UPDATE jobs SET embedding=? WHERE id=?', (blob, job_id))

conn.commit()
conn.close()
print("Database setup complete with example jobs.")
