import cv2
import face_recognition
import os
import csv
from datetime import datetime

# Create Images_Attendance folder if it doesn't exist
if not os.path.exists("Images_Attendance"):
    os.makedirs("Images_Attendance")

# Ask user for their name
name = input("Enter your name: ").strip()
if name == "":
    print("Name cannot be empty!")
    exit()

# Capture face from webcam
cap = cv2.VideoCapture(0)
print("Press 's' to capture your face...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam")
        break
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Press 's' to save face
        img_path = f"Images_Attendance/{name}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Image saved as {img_path}")
        break
    elif key == ord('q'):  # Quit program
        print("Quitting...")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Encode captured face
try:
    imgCaptured = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(imgCaptured)
    if len(encodings) == 0:
        print("No face detected! Please try again.")
        exit()
    imgCapturedEncoding = encodings[0]
except Exception as e:
    print(f"Error loading face: {e}")
    exit()

# Load all known faces from Images_Attendance folder
known_face_encodings = []
known_face_names = []

for file in os.listdir("Images_Attendance"):
    if file.endswith(".jpg") or file.endswith(".png"):
        path = f"Images_Attendance/{file}"
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)
        if len(encoding) > 0:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(file)[0])

print("Known faces loaded:", known_face_names)

# Initialize webcam for live recognition
cap = cv2.VideoCapture(0)
print("Press 'q' to quit live recognition...")

# Open CSV file for attendance
csv_file = "Attendance.csv"
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name_detected = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name_detected = known_face_names[best_match_index]

        # Draw rectangle and label
        top, right, bottom, left = [v*4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name_detected, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Mark attendance if recognized
        if name_detected != "Unknown":
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            with open(csv_file, "r+", newline="") as f:
                existing_lines = f.readlines()
                names_in_file = [line.split(",")[0] for line in existing_lines]
                if name_detected not in names_in_file:
                    f.write(f"{name_detected},{date_str},{time_str}\n")
                    print(f"Attendance marked for {name_detected}")

    cv2.imshow("Live Recognition", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting live recognition...")
        break

cap.release()
cv2.destroyAllWindows()
