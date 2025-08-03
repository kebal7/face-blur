import cv2
import face_recognition

# Load the image of the target face and encode it
target_image = face_recognition.load_image_file("target_face.jpg")
target_encoding = face_recognition.face_encodings(target_image)[0]

# Load video file or use webcam (0)
cap = cv2.VideoCapture(0)  # or use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing (but not too small)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert BGR (OpenCV format) to RGB (face_recognition format)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small)

    # Only encode if any faces were found
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with the target face
            match = face_recognition.compare_faces([target_encoding], face_encoding, tolerance=0.5)

            if match[0]:
                # Scale back to original size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Apply blur to the face region
                face_roi = frame[top:bottom, left:right]
                face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
                frame[top:bottom, left:right] = face_roi

    # Show the frame
    cv2.imshow("Target Face Blur", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

