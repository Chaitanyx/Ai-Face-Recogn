import sqlite3
import smtplib
from email.mime.text import MIMEText
import os

conn = sqlite3.connect("mood_tracking.db")
cursor = conn.cursor()

negative_emotions = {"angry", "sad", "fear", "disgust"}

def check_prolonged_stress():
    cursor.execute("SELECT emotion FROM mood_log WHERE timestamp >= datetime('now', '-7 days')")
    recent_moods = [row[0] for row in cursor.fetchall()]

    if not recent_moods:
        return 

    negative_count = sum(1 for mood in recent_moods if mood in negative_emotions)
    stress_percentage = (negative_count / len(recent_moods)) * 100

    if stress_percentage > 60:  
        send_alert_to_hr(stress_percentage)

def send_alert_to_hr(stress_percentage):
    sender_email = os.getenv("EMAIL_USER")
    hr_email = os.getenv("HR_EMAIL")
    manager_email = os.getenv("MANAGER_EMAIL")  

    if not sender_email:
        print("⚠ Email credentials missing. Cannot send alert.")
        return

    subject = "🚨 Employee Stress Alert!"
    body = f"Warning: The system detected {stress_percentage:.2f}% negative emotions over the past week."

    recipients = [hr_email, manager_email]  

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_email, os.getenv("EMAIL_PASS"))
        server.sendmail(sender_email, recipients, msg.as_string())

    print("📢 HR & Manager Alert Sent")

