import cv2
import face_recognition

# Load target face
target_image = face_recognition.load_image_file("target_face.png")
target_encoding = face_recognition.face_encodings(target_image)[0]

# Open video file or webcam
VIDEO_PATH = "video.mp4"  # Use 0 for webcam
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 24  # fallback to 24 FPS

# Define the codec and output file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use 'XVID' for .avi
out = cv2.VideoWriter("output_blurred.mp4", fourcc, fps, (frame_width, frame_height))

# Window
window_name = "Target Face Blur"
cv2.namedWindow(window_name)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Finished processing.")
        break

    # Resize and convert
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            match = face_recognition.compare_faces([target_encoding], face_encoding, tolerance=0.5)

            if match[0]:
                # Scale back
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Blur the face
                face_roi = frame[top:bottom, left:right]
                face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
                frame[top:bottom, left:right] = face_roi

    # Write frame to output video
    out.write(frame)

    # Show the frame
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        print("👋 Exited by user (q)")
        break

    # Detect window close
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        print("❌ Window closed")
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("📁 Saved as: output_blurred.mp4")

