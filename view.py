import cv2

# Open the recorded video file
cap = cv2.VideoCapture('face_recording.avi')

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        cv2.imshow('Game Recording', frame)

        # Press 'q' to quit the video playback
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()
