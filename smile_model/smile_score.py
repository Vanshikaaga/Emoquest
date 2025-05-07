import math
import cv2
import mediapipe as mp
import numpy as np
from fer import FER

# Initialize models
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
detector = FER(mtcnn=True)  # mtcnn=True gives better results for face alignment

cap = cv2.VideoCapture(0)

def rotation_matrix_to_angles(rotation_matrix):
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], math.sqrt(rotation_matrix[0, 0] ** 2 +
                                                     rotation_matrix[1, 0] ** 2))
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180. / math.pi

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image = frame.copy()
    h, w, _ = image.shape

    # For FER smile detection
    emotions = detector.detect_emotions(image)
    smile_score = 0.0
    if emotions:
        smile_score = emotions[0]["emotions"]["happy"]  # Value between 0 and 1
        cv2.putText(image, f'Smile: {smile_score:.2f}', (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Head pose estimation
    face_coordination_in_real_world = np.array([
        [285, 528, 200],
        [285, 371, 152],
        [197, 574, 128],
        [173, 425, 108],
        [360, 574, 128],
        [391, 425, 108]
    ], dtype=np.float64)

    face_coordination_in_image = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 9, 57, 130, 287, 359]:
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_coordination_in_image.append([x, y])

            face_coordination_in_image = np.array(face_coordination_in_image,
                                                  dtype=np.float64)

            # Camera matrix
            focal_length = 1 * w
            cam_matrix = np.array([[focal_length, 0, w / 2],
                                   [0, focal_length, h / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rotation_vec, translation_vec = cv2.solvePnP(
                face_coordination_in_real_world, face_coordination_in_image,
                cam_matrix, dist_matrix)

            rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
            result = rotation_matrix_to_angles(rotation_matrix)

            for i, info in enumerate(zip(('pitch', 'yaw', 'roll'), result)):
                k, v = info
                text = f'{k}: {int(v)}'
                cv2.putText(image, text, (20, i * 30 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

    cv2.imshow('Head Pose + Smile Score', image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
