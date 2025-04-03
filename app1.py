import sqlite3
import smtplib
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from datetime import datetime
from email.mime.text import MIMEText
from flask import Flask, render_template, Response, jsonify, request
import cv2
from keras.models import model_from_json
import numpy as np
import speech_recognition as sr
import librosa
from mood_database import save_mood_to_db

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

def save_mood_to_db(emotion):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect("mood_tracking.db") as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO mood_log (timestamp, emotion) VALUES (?, ?)", (timestamp, emotion))
        conn.commit()
    check_stress_alert()

def check_stress_alert():
    stress_emotions = {"sad", "angry", "fear"}
    with sqlite3.connect("mood_tracking.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT emotion FROM mood_log ORDER BY timestamp DESC LIMIT 5")
        last_moods = [row[0] for row in cursor.fetchall()]

    if len(last_moods) < 5:
        return

    stress_count = sum(1 for mood in last_moods if mood in stress_emotions)
    if stress_count >= 4:
        send_stress_alert()

def send_stress_alert():
    sender_email = os.getenv("EMAIL_USER")
    receiver_email = os.getenv("HR_EMAIL")
    password = os.getenv("EMAIL_PASS")

    if not sender_email or not password:
        print("⚠ Email credentials missing.")
        return

    subject = "🚨 Employee Stress Alert!"
    body = "An employee has shown signs of prolonged stress (sad, angry, or fear). Please check in."
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("📢 HR Alert Sent")
    except Exception as e:
        print(f"⚠ Email sending error: {e}")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

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
    webcam = cv2.VideoCapture(0)

    while True:
        success, im = webcam.read()
        if not success:
            break

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

            detected_faces.append({'emotion': detected_emotion, 'task': assigned_task})
            save_mood_to_db(detected_emotion)

            cv2.putText(im, detected_emotion, (p, q-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', im)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    webcam.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
