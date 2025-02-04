import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import tkinter as tk
from tkinter import Label, Button, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import threading

# Load the trained model
model = load_model('emotion_model.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emoji_dict = {
    'Angry': "😠", 'Disgust': "🤢", 'Fear': "😨", 'Happy': "😊",
    'Sad': "😢", 'Surprise': "😲", 'Neutral': "😐"
}

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Emotion tracking
detected_emotions = []


def capture_emotions():
    global detected_emotions
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(
                    bboxC.height * h)
                face_roi = frame[y:y + height, x:x + width]

                if face_roi.size == 0:
                    continue

                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (48, 48)) / 255.0
                reshaped_face = resized_face.reshape(1, 48, 48, 1)

                prediction = model.predict(reshaped_face, verbose=0)
                emotion_idx = np.argmax(prediction)
                emotion = emotion_labels[emotion_idx]
                confidence = np.max(prediction) * 100
                detected_emotions.append(emotion)

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion} ({confidence:.1f}%) {emoji_dict[emotion]}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def save_snapshot():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG files", "*.jpg"), ("All Files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, frame)
    cap.release()


def generate_report():
    global detected_emotions
    df = pd.DataFrame({'Emotions': detected_emotions})
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df.to_csv(file_path, index=False)
    detected_emotions = []


def plot_emotion_trends():
    plt.figure(figsize=(8, 4))
    plt.hist(detected_emotions, bins=len(emotion_labels), color='skyblue', edgecolor='black')
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    plt.title('Emotion Trends Over Time')
    plt.xticks(rotation=45)
    plt.show()


def start_emotion_detection():
    threading.Thread(target=capture_emotions).start()


# GUI Setup
root = tk.Tk()
root.title("Emotion Detection GUI")
root.geometry("400x400")

Label(root, text="Emotion Detection System", font=("Arial", 16)).pack()
Button(root, text="Start Detection", command=start_emotion_detection).pack(pady=5)
Button(root, text="Save Snapshot", command=save_snapshot).pack(pady=5)
Button(root, text="Generate Report", command=generate_report).pack(pady=5)
Button(root, text="Show Emotion Trends", command=plot_emotion_trends).pack(pady=5)
Button(root, text="Exit", command=root.quit).pack(pady=5)

root.mainloop()