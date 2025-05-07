import cv2
import math
import numpy as np
import mediapipe as mp
from fer import FER
from keras.models import load_model
from gaze_tracking import GazeTracking
from datetime import datetime
import time
import pandas as pd
import os

# ========== CONFIGURATION ==========
FRAME_SKIP = 2               # Process every 2nd frame
SHOW_DEBUG = True            # Set to False to disable visualizations
RESIZE_FACTOR = 0.5          # Downsample frames to half resolution
MIN_SMILE_THRESHOLD = 0.5    # Threshold for counting as a smile
# ===================================

# Load models
print("Loading models...")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)
fer_detector = FER(mtcnn=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model_file.h5')
emotion_model = load_model(model_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gaze = GazeTracking()

# Emotion labels
positive_labels = {'Happy', 'Surprise'}
negative_labels = {'Angry', 'Disgust', 'Fear', 'Sad'}
neutral_labels = {'Neutral'}

emotion_labels = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 
    3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

# Initialize tracking variables
frame_log = []
blink_count = 0
focused_frames = 0
total_frames = 0
start_time = time.time()

# Video setup
import os
video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'face_recording.avi'))

if not os.path.exists(video_path):
    print(f"[ERROR] Video file not found: {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] Could not open video file")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps if fps > 0 else 0
print(f"Video Info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f} seconds")

def rotation_matrix_to_angles(rotation_matrix):
    """Convert rotation matrix to Euler angles"""
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi

# Main processing loop
frame_counter = 0
print("Starting video processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video reached")
        break
    
    frame_counter += 1
    if frame_counter % FRAME_SKIP != 0:
        continue  # Skip this frame
    
    # Downsample frame for faster processing
    if RESIZE_FACTOR < 1.0:
        frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Face detection (faster than face mesh)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)  # Faster parameters
    
    # Only run expensive processing if faces detected
    if len(faces) > 0:
        # Face Mesh Processing
        results = face_mesh.process(image_rgb)
        
        # Smile Detection (FER)
        try:
            emotions = fer_detector.detect_emotions(frame)
            smile_score = emotions[0]["emotions"]["happy"] if emotions else 0.0
        except:
            smile_score = 0.0
        
        # Head Pose Estimation
        pitch, yaw = None, None
        if results.multi_face_landmarks:
            face_coord_image = []
            face_coord_real = np.array([
                [285, 528, 200], [285, 371, 152],
                [197, 574, 128], [173, 425, 108],
                [360, 574, 128], [391, 425, 108]
            ], dtype=np.float64)
            
            for face_landmarks in results.multi_face_landmarks:
                for idx in [1, 9, 57, 130, 287, 359]:  # Key landmarks only
                    lm = face_landmarks.landmark[idx]
                    face_coord_image.append([int(lm.x * w), int(lm.y * h)])
            
            if len(face_coord_image) == 6:  # Only process if we got all points
                face_coord_image = np.array(face_coord_image, dtype=np.float64)
                cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
                success, rot_vec, _ = cv2.solvePnP(
                    face_coord_real, face_coord_image, 
                    cam_matrix, dist_matrix
                )
                rot_matrix, _ = cv2.Rodrigues(rot_vec)
                angles = rotation_matrix_to_angles(rot_matrix)
                pitch, yaw = int(angles[0]), int(angles[1])
    
        # Gaze Tracking
        gaze.refresh(frame)
        gaze_on_screen = 0
        blink = 0
        
        if gaze.is_blinking():
            blink = 1
            blink_count += 1
        elif gaze.is_center():
            gaze_on_screen = 1
            focused_frames += 1
        
        # Emotion Detection
        detected_emotion = "Neutral"
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            resized = cv2.resize(roi_gray, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))
            prediction = emotion_model.predict(reshaped, verbose=0)  # Disable logging
            detected_emotion = emotion_labels[np.argmax(prediction)]
    else:
        # Default values when no face detected
        smile_score = 0.0
        detected_emotion = "Neutral"
        gaze_on_screen = 0
        blink = 0
        pitch, yaw = None, None
    
    total_frames += 1
    
    # Log frame data
    frame_log.append({
        'timestamp': timestamp,
        'emotion': detected_emotion,
        'smile_score': smile_score,
        'blink': blink,
        'gaze_on_screen': gaze_on_screen,
        'head_pitch': pitch if pitch is not None else '',
        'head_yaw': yaw if yaw is not None else ''
    })
    
    # Visualization (only if enabled)
    if SHOW_DEBUG:
        focus_percent = (focused_frames / total_frames) * 100 if total_frames else 0
        
        cv2.putText(frame, f"Emotion: {detected_emotion}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Smile: {smile_score:.2f}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(frame, f"Focus: {focus_percent:.1f}%", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        
        if pitch is not None and yaw is not None:
            cv2.putText(frame, f"Head: P{pitch}° Y{yaw}°", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        cv2.imshow("Emotion Analysis Dashboard", frame)
    
    # Early exit if 'q' pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save and analyze results
print("\nProcessing results...")
df = pd.DataFrame(frame_log)
df.to_csv("session_log.csv", index=False)

# Calculate metrics
session_time_min = (time.time() - start_time) / 60
total_frames_logged = len(df)

positive_emotions = df['emotion'].apply(lambda e: e in positive_labels).sum()
negative_emotions = df['emotion'].apply(lambda e: e in negative_labels).sum()
neutral_emotions = df['emotion'].apply(lambda e: e in neutral_labels).sum()
positivity_index = positive_emotions / max(1, (positive_emotions + negative_emotions))

smiling_frames = (df['smile_score'] > MIN_SMILE_THRESHOLD).sum()
smile_rate_per_min = smiling_frames / max(0.1, session_time_min)  # Avoid division by zero
avg_smile_intensity = df['smile_score'].mean()

blink_rate_per_min = blink_count / max(0.1, session_time_min)
focused_percent = df['gaze_on_screen'].mean() * 100

# Print comprehensive report
print("\n======= SESSION SUMMARY =======")
print(f"{'Total Duration:':<25} {session_time_min:.1f} minutes")
print(f"{'Frames Processed:':<25} {total_frames_logged}")
print(f"{'Processing Speed:':<25} {total_frames_logged/max(0.1, session_time_min):.1f} FPS")
print("\n----- Engagement Metrics -----")
print(f"{'Focus on Screen:':<25} {focused_percent:.1f}%")
print(f"{'Smile Rate:':<25} {smile_rate_per_min:.1f} smiles/min")
print(f"{'Avg Smile Intensity:':<25} {avg_smile_intensity:.2f}")
print(f"{'Blink Rate:':<25} {blink_rate_per_min:.1f} blinks/min")
print("\n----- Emotion Analysis -----")
print(f"{'Positivity Index:':<25} {positivity_index:.2f}")
print(f"{'Positive Emotions:':<25} {positive_emotions} ({positive_emotions/total_frames_logged:.1%})")
print(f"{'Neutral Emotions:':<25} {neutral_emotions} ({neutral_emotions/total_frames_logged:.1%})")
print(f"{'Negative Emotions:':<25} {negative_emotions} ({negative_emotions/total_frames_logged:.1%})")
print("==============================")
import json

# Prepare report data
report_data = {
    "total_duration_min": round(session_time_min, 2),
    "frames_processed": total_frames_logged,
    "processing_speed_fps": round(total_frames_logged / max(0.1, session_time_min), 2),
    "focus_on_screen_percent": round(focused_percent, 2),
    "smile_rate_per_min": round(smile_rate_per_min, 2),
    "avg_smile_intensity": round(avg_smile_intensity, 3),
    "blink_rate_per_min": round(blink_rate_per_min, 2),
    "positive_emotions_count": int(positive_emotions),
    "negative_emotions_count": int(negative_emotions),
    "neutral_emotions_count": int(neutral_emotions),
    "positivity_index": round(positivity_index, 3)
}

# Load existing report.json if exists
report_file = "report.json"
if os.path.exists(report_file):
    with open(report_file, 'r') as f:
        existing_reports = json.load(f)
else:
    existing_reports = []

# Append new session report
existing_reports.append(report_data)

# Write back to report.json
with open(report_file, 'w') as f:
    json.dump(existing_reports, f, indent=4)

print(f"\nReport saved to {report_file}")
