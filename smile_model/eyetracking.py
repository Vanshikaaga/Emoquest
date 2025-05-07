import cv2
import mediapipe as mp
from deepface import DeepFace
from gaze_tracking import GazeTracking
import time
import json

# Initialize OpenCV and GazeTracking
gaze = GazeTracking()
cap = cv2.VideoCapture(0)

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Data Logger List
data_log = []

# Function to calculate smile confidence
def calculate_smile_confidence(landmarks, img_width, img_height):
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    TOP_LIP = 13
    BOTTOM_LIP = 14

    left = landmarks[LEFT_MOUTH]
    right = landmarks[RIGHT_MOUTH]
    top = landmarks[TOP_LIP]
    bottom = landmarks[BOTTOM_LIP]

    # Convert normalized coordinates to pixel values
    left = (int(left.x * img_width), int(left.y * img_height))
    right = (int(right.x * img_width), int(right.y * img_height))
    top = (int(top.x * img_width), int(top.y * img_height))
    bottom = (int(bottom.x * img_width), int(bottom.y * img_height))

    mouth_width = ((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2) ** 0.5
    mouth_height = ((bottom[0] - top[0]) ** 2 + (bottom[1] - top[1]) ** 2) ** 0.5

    if mouth_height == 0:
        return 0
    smile_ratio = mouth_width / mouth_height

    smile_confidence = min(max((smile_ratio - 1.5) * 100, 0), 100)
    return smile_confidence

# Main loop
while cap.isOpened():
    _, frame = cap.read()
    gaze.refresh(frame)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_height, img_width = frame.shape[:2]
    
    # Process the gaze tracking
    new_frame = gaze.annotated_frame()
    text = ""
    
    if gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    # Initialize values
    smile_confidence = 0
    emotion_text = "Detecting..."
    timestamp = time.time()  # Capture timestamp for the current frame

    # Process the face landmarks and detect smile and emotion
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            smile_confidence = calculate_smile_confidence(face_landmarks.landmark, img_width, img_height)

            mp.solutions.drawing_utils.draw_landmarks(
                new_frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

        # Emotion detection with DeepFace
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 10 == 0:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                dominant_emotion = result[0]['dominant_emotion']
                emotion_text = dominant_emotion
            except Exception as e:
                print("Emotion detection error:", e)

    # Display Smile and Emotion on frame
    cv2.putText(new_frame, f"Smile: {int(smile_confidence)}%", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)
    cv2.putText(new_frame, f"Emotion: {emotion_text}", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 105, 180), 2)

    # Display Gaze Direction
    cv2.putText(new_frame, text, (60, 150), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)

    # Log data every 1 second (or any n frames you prefer)
    if int(timestamp) % 1 == 0:
        data_log.append({
            "timestamp": timestamp,
            "smile_confidence": smile_confidence,
            "emotion_class": emotion_text,
            "gaze_direction": text
        })

    # Show the final frame with all the information
    cv2.imshow("Real-Time Tracking", new_frame)

    # Exit if 'Esc' is pressed
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Optionally, save the data log to a file (JSON format)
with open('data_log.json', 'w') as f:
    json.dump(data_log, f, indent=4)
