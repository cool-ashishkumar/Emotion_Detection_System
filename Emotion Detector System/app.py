
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained emotion model (update the model path if needed)
model = load_model('emotion_model.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 = short-range detection, 1 = full-range
    min_detection_confidence=0.5
)

# Start webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Default emotion text for display (if no face is detected)
    display_text = "No face detected"

    if results.detections:
        for detection in results.detections:
            # Get relative bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            # Clamp coordinates to frame boundaries
            x = max(x, 0)
            y = max(y, 0)
            width = min(width, w - x)
            height = min(height, h - y)

            # Extract face ROI and ensure it's not empty
            face_roi = frame[y:y+height, x:x+width]
            if face_roi.size == 0:
                continue

            try:
                # Preprocess face ROI: convert to grayscale, resize, normalize, and reshape
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (48, 48))
                normalized_face = resized_face / 255.0
                reshaped_face = normalized_face.reshape(1, 48, 48, 1)

                # Predict emotion
                prediction = model.predict(reshaped_face, verbose=0)
                emotion_idx = np.argmax(prediction)
                emotion = emotion_labels[emotion_idx]
                confidence = np.max(prediction) * 100

                # Update display text based on the predicted emotion
                if emotion == "Happy":
                    display_text = "Your Emotion is Happy"
                else:
                    display_text = f"Your Emotion is {emotion}"

                # Draw the bounding box around the detected face
                cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                # Display the emotion label and confidence near the face
                cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face ROI: {e}")
                continue

    # Display the overall message at the top of the frame
    cv2.putText(frame, display_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Show the frame with overlays
    cv2.imshow('Real-Time Emotion Detection (MediaPipe)', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1)  == 13:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
