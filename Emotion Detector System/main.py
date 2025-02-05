import cv2
import numpy as np
import csv
from datetime import datetime
from tensorflow.keras.models import load_model
import mediapipe as mp

# Define Colors
RED = (0, 0, 255)  # Red for instructions

# Emotion Colors for Dynamic Text & Bounding Box
emotion_colors = {
    'Angry': (0, 0, 255),  # Red
    'Disgust': (0, 255, 255),  # Dark Green
    'Fear': (128, 0, 128),  # Purple
    'Happy': (15, 255, 80),  # 
    'Sad': (255, 0, 0),  # Blue
    'Surprise': (255, 165, 0),  # Orange
    'Neutral': (128, 128, 128)  # Gray
}

# Load Emotion Recognition Model
model = load_model('emotion_model.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7)

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Emotion Log
emotion_log = []


def log_emotion(emotion, confidence):
    """Save emotion to the log list"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emotion_log.append([timestamp, emotion, f"{confidence:.1f}%"])


def save_report():
    """Generate and save emotion report as CSV"""
    with open('emotion_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Emotion", "Confidence"])
        writer.writerows(emotion_log)
    print("Emotion report saved successfully!")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Process frame with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    current_emotion = "No face detected"
    emotion_color = (203, 192, 255)  # Default Light Pink
    max_confidence = 0

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, width, height = (int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h))
            x, y = max(x, 0), max(y, 0)
            width, height = min(width, w - x), min(height, h - y)

            face_roi = frame[y:y + height, x:x + width]
            if face_roi.size == 0:
                continue

            try:
                # Preprocess face for emotion model
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (48, 48))
                normalized_face = resized_face / 255.0
                reshaped_face = normalized_face.reshape(1, 48, 48, 1)

                # Predict emotion
                prediction = model.predict(reshaped_face, verbose=0)
                emotion_idx = np.argmax(prediction)
                emotion = emotion_labels[emotion_idx]
                confidence = np.max(prediction) * 100

                if confidence > max_confidence:
                    current_emotion = f"Your Emotion is: {emotion}"
                    max_confidence = confidence
                    emotion_color = emotion_colors.get(emotion, (203, 192, 255))  # Dynamic color

                # Draw face bounding box & emotion text with dynamic color
                cv2.rectangle(frame, (x, y), (x + width, y + height), emotion_color, 2)  # Dynamic Color
                cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            emotion_color, 2)

                # Log emotion
                log_emotion(emotion, confidence)
            except Exception as e:
                print(f"Processing error: {str(e)}")
                continue

    # Display dynamic emotion status with changing color
    cv2.putText(frame, current_emotion, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, emotion_color, 2)

    # Display instructions in red
    cv2.putText(frame, "Press 'S' to Snap | 'R' for Report | 'Enter' for Quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

    cv2.imshow('Emotion Detection', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == 13:
        break
    elif key & 0xFF == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved snapshot: {filename}")
    elif key & 0xFF == ord('r'):
        save_report()

cap.release()
cv2.destroyAllWindows()
