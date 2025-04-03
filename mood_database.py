# mood_database.py
import sqlite3
from datetime import datetime

conn = sqlite3.connect("mood_tracking.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS mood_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        face_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        emotion TEXT NOT NULL
    )
""")
conn.commit()

print("✅ SQLite Database connected and table created successfully!")

def save_mood_to_db(face_id, emotion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect("mood_tracking.db") as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO mood_log (face_id, timestamp, emotion) VALUES (?, ?, ?)", (face_id, timestamp, emotion))
        conn.commit()
    print(f"📝 Saved: Face ID {face_id}, Emotion: {emotion} at {timestamp}")
    check_stress_alert()

def fetch_last_moods():
    with sqlite3.connect("mood_tracking.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT face_id, emotion FROM mood_log ORDER BY timestamp DESC LIMIT 5")
        return cursor.fetchall()

def check_stress_alert():
    stress_emotions = {"sad", "angry", "fear"}
    last_moods = fetch_last_moods()

    stress_counts = {}
    for face_id, emotion in last_moods:
        if emotion in stress_emotions:
            stress_counts[face_id] = stress_counts.get(face_id, 0) + 1

    for face_id, count in stress_counts.items():
        if count >= 4:
            notify_hr(face_id)

def notify_hr(face_id):
    print(f"🚨 ALERT: Face ID {face_id} has shown prolonged signs of stress (sad, angry, or fear). Notify HR/Manager for immediate support.")

def close_connection():
    conn.close()
    print("🔒 Database connection closed!")
