import cv2
import face_recognition
from AttendanceProject import mark_attendance  # assuming mark_attendance() is defined there
import os
import numpy as np

# Ensure the folder for captured images exists
if not os.path.exists("Images_Attendance"):
    os.makedirs("Images_Attendance")

# Initialize webcam
cap = cv2.VideoCapture(0)
print("Press 's' to capture your face for attendance")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Webcam", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Capture face
        img_path = "Images_Attendance/captured_face.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Face captured and saved as {img_path}")
        break
    elif key == ord('q'):  # Quit
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Encode captured face
img = face_recognition.load_image_file(img_path)
captured_face_encoding = face_recognition.face_encodings(img)[0]

# Store known face encodings and names
known_face_encodings = [captured_face_encoding]
known_face_names = ["New User"]  # Replace with dynamic input if needed

# Start live recognition for attendance
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            mark_attendance(name)  # Log attendance in CSV

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q'
        break

cap.release()
cv2.destroyAllWindows()
