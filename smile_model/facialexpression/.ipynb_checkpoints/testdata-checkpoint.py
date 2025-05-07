import cv2
import numpy as np
import pandas as pd
import time
from keras.models import load_model
from gaze_tracking import GazeTracking
from datetime import datetime

# Load models
emotion_model = load_model('model_file.h5')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gaze = GazeTracking()

# Constants
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
positive_emotions = ['Happy', 'Surprise']
negative_emotions = ['Angry', 'Disgust', 'Fear', 'Sad']

# Start video capture
cap = cv2.VideoCapture(0)

# Data storage
log_data = []
start_time = time.time()
blink_count = 0
smile_count = 0
frame_count = 0
gaze_on_screen_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Detect emotion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 3)

    emotion_label = "Neutral"
    smile_score = 0.0

    for (x, y, w, h) in faces:
        sub_face = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face, (48, 48))
        norm = resized / 255.0
        reshaped = np.reshape(norm, (1, 48, 48, 1))
        prediction = emotion_model.predict(reshaped)
        smile_score = float(prediction[0][3])  # 'Happy' index
        emotion_label = labels_dict[np.argmax(prediction)]

        if emotion_label == 'Happy':
            smile_count += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Gaze tracking
    gaze.refresh(frame)
    blink = gaze.is_blinking()
    if blink:
        blink_count += 1
    on_screen = gaze.is_center()
    if on_screen:
        gaze_on_screen_count += 1

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    frame_count += 1

    # Log data per frame
    log_data.append({
        "timestamp": timestamp,
        "emotion": emotion_label,
        "smile_score": round(smile_score, 3),
        "blink": int(blink),
        "gaze_on_screen": int(on_screen),
        "left_pupil": left_pupil,
        "right_pupil": right_pupil
    })

    # Display
    cv2.putText(frame, f"Emotion: {emotion_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Blink: {'Yes' if blink else 'No'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Gaze: {'Center' if on_screen else 'Off'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Gaze + Emotion Tracking", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

# Stop session
cap.release()
cv2.destroyAllWindows()

# Convert to DataFrame
df = pd.DataFrame(log_data)

# Save raw data
df.to_csv("session_log.csv", index=False)

# Time calculations
session_duration = (time.time() - start_time) / 60  # in minutes

# Derived metrics
smile_rate = smile_count / (session_duration * 60)
avg_smile_intensity = df['smile_score'].mean()
blink_rate = blink_count / session_duration
gaze_engagement = gaze_on_screen_count / frame_count
positive_count = df['emotion'].isin(positive_emotions).sum()
negative_count = df['emotion'].isin(negative_emotions).sum()
positivity_index = positive_count / (positive_count + negative_count + 1e-5)

# Display summary
print("\n--- Session Metrics ---")
print(f"Smile Rate: {smile_rate:.2f} smiles/sec")
print(f"Avg Smile Intensity: {avg_smile_intensity:.2f}")
print(f"Blink Rate: {blink_rate:.2f} blinks/min")
print(f"Gaze Engagement: {gaze_engagement:.2%}")
print(f"Positivity Index: {positivity_index:.2f}")
