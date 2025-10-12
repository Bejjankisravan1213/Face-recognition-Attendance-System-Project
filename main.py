import cv2
import face_recognition
import os
import csv
from datetime import datetime

# === Ensure folders and files exist ===
if not os.path.exists("Images_Attendance"):
    os.makedirs("Images_Attendance")

if not os.path.exists("Attendance.csv"):
    with open("Attendance.csv", "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Time"])

# === Capture your face dynamically ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()

print("Press 's' to capture your face or 'q' to quit.")
name = input("Enter your name: ").strip()

if name == "":
    print("❌ Name cannot be empty.")
    cap.release()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from webcam.")
        break

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Save face
        file_path = f"Images_Attendance/{name}.jpg"
        cv2.imwrite(file_path, frame)
        print(f"✅ Face captured and saved as {file_path}")
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# === Load captured image and encode ===
try:
    imgCaptured = face_recognition.load_image_file(file_path)
    encodings = face_recognition.face_encodings(imgCaptured)
    if len(encodings) == 0:
        print("❌ No face detected in the saved image. Try again.")
        exit()
    imgCapturedEncoding = encodings[0]
except Exception as e:
    print(f"❌ Error processing captured image: {e}")
    exit()

# === Start live webcam recognition ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()

print("\n🎥 Starting live face recognition... Press 'q' to quit.\n")

attendance_marked = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces([imgCapturedEncoding], face_encoding)
        if True in matches:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if not attendance_marked:
                with open("Attendance.csv", "r+", newline="") as f:
                    existing = [row[0] for row in csv.reader(f)]
                    if name not in existing:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        csv.writer(f).writerow([name, now])
                        print(f"🟢 Attendance marked for {name} at {now}")
                        attendance_marked = True

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




