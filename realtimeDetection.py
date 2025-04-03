import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("expressiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("expressiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  
    return feature / 255.0  

task_assignment = {
    'angry': "Perform Deep breathing exercise, Try to use humor or listen to music.",
    'disgust': "Avoid negative chatter, Engage in a clear discussion, Think about your acheivements.",
    'fear': "calmly reassess your mind, try to seek a professional guide/trusted colleague to share your feelings.",
    'happy': "Encourage collaboration and brainstorming with colleagues. Share jokes with your coworkers",
    'neutral': "Continue with assigned tasks normally. Have a look at the 'wall of fame' or company acheivemnets",
    'sad': "Assign lighter tasks or encourage social interaction or else Take a small break talk to your family",
    'surprise': "Allow creative or unexpected tasks with respective rewards and appreciations or a small party "
}

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

webcam = cv2.VideoCapture(0)

while True:
    ret, im = webcam.read()
    if not ret:
        continue 

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

    detected_faces = []  

    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]  
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)  

            image = cv2.resize(image, (48, 48))  
            img = extract_features(image)

            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]  
            detected_faces.append(prediction_label)  

            cv2.putText(im, prediction_label, (p, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

    except cv2.error as e:
        print(f"OpenCV error: {e}")

    cv2.imshow("Emotion Detection", im)

    if detected_faces:
        print("\nDetected Emotions & Task Assignments:")
        for emotion in detected_faces:
            print(f"Emotion: {emotion} → Suggested Task: {task_assignment[emotion]}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


#Run in VS_Terminal- python realtimeDetection.py
