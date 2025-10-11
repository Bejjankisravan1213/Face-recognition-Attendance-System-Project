import cv2
import face_recognition
import os
import csv
from datetime import datetime

# Ensure folders exist
if not os.path.exists("Images_Attendance"):
    os.makedirs("Images_Attendance")

if not os.path.exists("Attendance.csv"):
    with open("Attendance.csv", "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Time"])

# Capture your face dynamically
cap = cv2.VideoCapture(0)
print("Press 's' to capture your face...")
name = input("Enter your name: ")

while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):  # capture face
        file_path = f"Images_Attendance/{name}.jpg"
        cv2.imwrite(file_path, frame)
        print(f"Face captured and saved as {file_path}")
        break
    elif key == ord('q'):  # quit
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Load captured image and encode
imgCaptured = face_recognition.load_image_file(file_path)
imgCapturedEncoding = face_recognition.face_encodings(imgCaptured)[0]

# Start live webcam recognition
cap = cv2.VideoCapture(0)
print("Starting live face recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces([imgCapturedEncoding], face_encoding)
        if True in matches:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Mark attendance
            with open("Attendance.csv", "r+", newline="") as f:
                existing = [row[0] for row in csv.reader(f)]
                if name not in existing:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv.writer(f).writerow([name, now])
                    print(f"Attendance marked for {name}")

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


