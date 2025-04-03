import sqlite3
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, jsonify, request
import cv2
from keras.models import model_from_json
import numpy as np
import speech_recognition as sr
import librosa
import threading

app = Flask(__name__)

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=False)

json_file = open("expressiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("expressiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
task_assignment = {
    'angry': "Take deep breaths and relax.",
    'disgust': "Engage in a team discussion for clarity.",
    'fear': "Reassess task workload and provide support.",
    'happy': "Encourage collaboration and brainstorming.",
    'neutral': "Continue with assigned tasks normally.",
    'sad': "Assign lighter tasks or encourage social interaction.",
    'surprise': "Allow creative or unexpected tasks."
}

alert_sent = {}  # Dictionary to track alerts for each face

def save_mood_to_db(face_id, emotion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect("mood_tracking.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mood_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                face_id TEXT NOT NULL,
                emotion TEXT NOT NULL
            )
        """)
        cursor.execute("INSERT INTO mood_log (timestamp, face_id, emotion) VALUES (?, ?, ?)", (timestamp, face_id, emotion))
        conn.commit()
    check_stress_alert(face_id)

def check_stress_alert(face_id):
    global alert_sent
    stress_emotions = {"sad", "angry", "fear"}
    time_limit = datetime.now() - timedelta(seconds=15)
    with sqlite3.connect("mood_tracking.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT emotion FROM mood_log WHERE face_id = ? AND timestamp >= ?", (face_id, time_limit.strftime("%Y-%m-%d %H:%M:%S")))
        last_moods = [row[0] for row in cursor.fetchall()]

    stress_count = sum(1 for mood in last_moods if mood in stress_emotions)
    if stress_count >= 4 and not alert_sent.get(face_id, False):
        alert_sent[face_id] = True
        threading.Timer(15, reset_alert, [face_id]).start()
        send_stress_alert(face_id)

def reset_alert(face_id):
    global alert_sent
    alert_sent[face_id] = False

def send_stress_alert(face_id):
    print(f"🚨 HR ALERT: Employee with Face ID {face_id} is showing prolonged stress.")

def analyze_text_emotion(text):
    sentiment = sia.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return "happy"
    elif sentiment['compound'] <= -0.05:
        return "sad"
    else:
        emotion_result = emotion_classifier(text)[0]['label']
        return emotion_result

def analyze_speech_emotion(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    if np.mean(mfccs) > 0:
        return "happy"
    else:
        return "neutral"

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def generate_face_id(p, q, r, s):
    return f"{p}-{q}-{r}-{s}"

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get("text", "")
    face_id = data.get("face_id", "unknown")
    emotion = analyze_text_emotion(text)
    save_mood_to_db(face_id, emotion)
    return jsonify({"emotion": emotion})

@app.route('/analyze_speech', methods=['POST'])
def analyze_speech():
    audio_file = request.files['file']
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    face_id = request.form.get("face_id", "unknown")
    emotion = analyze_speech_emotion(audio_path)
    save_mood_to_db(face_id, emotion)
    return jsonify({"emotion": emotion})

detected_faces = []

@app.route('/get_emotion_task')
def get_emotion_task():
    return jsonify(detected_faces) if detected_faces else jsonify([{"emotion": "No face detected", "task": "N/A"}])

def generate_frames():
    global detected_faces
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_skip = 2
    frame_count = 0

    while True:
        success, im = webcam.read()
        if not success:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
        detected_faces.clear()

        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)

            pred = model.predict(img)
            detected_emotion = labels[pred.argmax()]
            assigned_task = task_assignment[detected_emotion]

            face_id = generate_face_id(p, q, r, s)
            detected_faces.append({'face_id': face_id, 'emotion': detected_emotion, 'task': assigned_task})
            save_mood_to_db(face_id, detected_emotion)

            cv2.putText(im, detected_emotion, (p, q-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', im)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
