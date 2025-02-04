import os
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load trained model
model = load_model('emotion_model.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Emotion colors for background change
emotion_colors = {
    "Happy": "#FFFF99",  # Light Yellow
    "Sad": "#A9CCE3",  # Light Blue
    "Angry": "#E74C3C",  # Red
    "Neutral": "#D5DBDB",  # Gray
    "Fear": "#7D3C98",  # Purple
    "Surprise": "#F4D03F",  # Yellow
    "Disgust": "#58D68D"  # Green
}

# Motivational messages
motivational_quotes = {
    "Happy": "Keep spreading positivity! 😊",
    "Sad": "Cheer up! Things will get better soon! 💙",
    "Angry": "Take a deep breath! Stay calm. 😌",
    "Neutral": "Hope you have a great day! ☀️",
    "Fear": "You are stronger than your fears! 💪",
    "Surprise": "Exciting times ahead! 🎉",
    "Disgust": "Stay positive! Good things are coming. 🌟"
}

# Emotion tracking
emotion_history = []
confidence_history = []
timestamps = []

# GUI Setup
window = tk.Tk()
window.title("Interactive Emotion Detection")
window.geometry("1000x700")
window.configure(bg="#D5DBDB")

# Video display label
video_label = tk.Label(window)
video_label.pack(pady=10)

# Emotion label
emotion_message = tk.StringVar()
emotion_message.set("No face detected")
emotion_label = tk.Label(window, textvariable=emotion_message, font=("Helvetica", 16, "bold"))
emotion_label.pack()

# Confidence level display
confidence_message = tk.StringVar()
confidence_message.set("Confidence: N/A")
confidence_label = tk.Label(window, textvariable=confidence_message, font=("Helvetica", 14))
confidence_label.pack()

# Stop button to exit
stop_button = ttk.Button(window, text="Stop", command=window.quit)
stop_button.pack(pady=10)

# Function to update frame
def update_frame():
    global emotion_history, confidence_history, timestamps

    ret, frame = cap.read()
    if not ret:
        window.after(10, update_frame)
        return

    # Convert frame for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    overall_emotion = "Neutral"
    confidence = 0.0

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            face_roi = frame[y:y+height, x:x+width]

            if face_roi.size == 0:
                continue

            try:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (48, 48))
                normalized_face = resized_face / 255.0
                reshaped_face = normalized_face.reshape(1, 48, 48, 1)

                prediction = model.predict(reshaped_face, verbose=0)
                emotion_idx = np.argmax(prediction)
                emotion = emotion_labels[emotion_idx]
                confidence = np.max(prediction) * 100  # Confidence percentage

                # Track detected emotions
                emotion_history.append(emotion)
                confidence_history.append(confidence)
                timestamps.append(time.time())

                # Choose dominant emotion
                overall_emotion = max(set(emotion_history[-10:]), key=emotion_history[-10:].count)

                # Change background color
                window.configure(bg=emotion_colors.get(overall_emotion, "#D5DBDB"))

                # Update displayed emotion and confidence
                emotion_message.set(motivational_quotes.get(overall_emotion, "Stay positive!"))
                confidence_message.set(f"Confidence: {confidence:.1f}%")

                # Draw face bounding box
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing face: {e}")

    # Convert frame for Tkinter
    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(display_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    window.after(10, update_frame)

# Graph function
def update_graph():
    global emotion_history, timestamps

    plt.clf()
    plt.title("Emotion Trend Over Time")
    plt.xlabel("Time")
    plt.ylabel("Emotion")
    plt.xticks(rotation=45)
    plt.yticks([])
    plt.grid(True)

    plt.plot(timestamps, emotion_history, marker="o", linestyle="-")
    canvas.draw()
    window.after(5000, update_graph)  # Update every 5 seconds

# Graph setup
fig, ax = plt.subplots(figsize=(5, 2))
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().pack()

# Start updates
update_frame()
update_graph()

window.mainloop()

cap.release()
cv2.destroyAllWindows()
