import cv2
import face_recognition
import os
import csv
from datetime import datetime
import numpy as np

# === Ensure folders and files exist ===
if not os.path.exists("Images_Attendance"):
    os.makedirs("Images_Attendance")

if not os.path.exists("Attendance.csv"):
    with open("Attendance.csv", "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Time"])

# === Helper function to detect blink ===
def is_blinking(landmarks):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]

    def eye_aspect_ratio(eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (A + B) / (2.0 * C)

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0
    return ear < 0.21  # threshold for blink

# === Load all known faces ===
known_encodings = []
known_names = []

for file in os.listdir("Images_Attendance"):
    if file.endswith(".jpg"):
        img = face_recognition.load_image_file(f"Images_Attendance/{file}")
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])

# === Start webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access webcam.")
    exit()

print("\n🎥 Blink once to capture your face...\n")

blink_count = 0
photo_captured = False
file_path = ""

# === Blink detection and photo capture ===
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    if frame is None or frame.size == 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    face_locations = face_recognition.face_locations(rgb)
    face_landmarks_list = face_recognition.face_landmarks(rgb)

    for face_location, landmarks in zip(face_locations, face_landmarks_list):
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        if is_blinking(landmarks):
            blink_count += 1
            cv2.putText(frame, "Blink Detected!", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            blink_count = 0

        # Capture after one blink
        if blink_count == 1 and not photo_captured:
            print("🟢 Blink detected! Capturing your face...")
            photo_captured = True
            file_path = "Images_Attendance/temp_capture.jpg"
            cv2.imwrite(file_path, frame)
            break

    cv2.imshow("Blink to Capture", frame)
    if photo_captured or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === Process captured image ===
if not photo_captured:
    print("❌ No photo captured. Exiting.")
    exit()

captured_img = face_recognition.load_image_file(file_path)
captured_encoding_list = face_recognition.face_encodings(captured_img)
if len(captured_encoding_list) == 0:
    print("❌ No face detected. Try again.")
    exit()
captured_encoding = captured_encoding_list[0]

# === Check if face is already known ===
matches = face_recognition.compare_faces(known_encodings, captured_encoding)
name = "Unknown"

if True in matches:
    match_index = matches.index(True)
    name = known_names[match_index]
    print(f"✅ Face recognized: {name}")
else:
    name = input("Enter your name to save: ").strip()
    if name:
        save_path = f"Images_Attendance/{name}.jpg"
        cv2.imwrite(save_path, cv2.imread(file_path))
        known_encodings.append(captured_encoding)
        known_names.append(name)
        print(f"✅ Saved new face as {name}")

# === Mark attendance ===
if name != "Unknown":
    with open("Attendance.csv", "r+", newline="") as f:
        existing = [row[0] for row in csv.reader(f)]
        if name not in existing:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv.writer(f).writerow([name, now])
            print(f"🕒 Attendance marked for {name} at {now}")

# === Live recognition window ===
cap = cv2.VideoCapture(0)
print("\n🎥 Starting live recognition... Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    if frame is None or frame.size == 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Live Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








