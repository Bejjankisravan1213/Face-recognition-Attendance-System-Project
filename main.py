import cv2
import face_recognition
import dlib
import numpy as np
import os
import pickle
from datetime import datetime

# ---------- CONFIG ----------
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
ENCODING_FILE = "encodings.pkl"
SAVE_FOLDER = "Captured_Images"

# ---------- SETUP ----------
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

if not os.path.exists(ENCODING_FILE):
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump({}, f)

# Load dlib's face landmark predictor
predictor = dlib.shape_predictor(MODEL_PATH)

# ---------- UTILS ----------
def is_blinking(landmarks):
    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]

    def eye_aspect_ratio(eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        return (A + B) / (2.0 * C)

    left_ratio = eye_aspect_ratio(left_eye)
    right_ratio = eye_aspect_ratio(right_eye)
    ear = (left_ratio + right_ratio) / 2.0
    return ear < 0.25  # smaller = eyes closed


def load_encodings():
    with open(ENCODING_FILE, "rb") as f:
        return pickle.load(f)


def save_encoding(name, encoding):
    data = load_encodings()
    data[name] = encoding
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(data, f)


# ---------- STEP 1: BLINK TO CAPTURE ----------
print("🎥 Blink once to capture your face...")

cap = cv2.VideoCapture(0)
blink_count = 0
photo_captured = False
file_path = ""

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.convertScaleAbs(rgb)  # Ensure 8-bit image
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    try:
        face_locations = face_recognition.face_locations(rgb)
        face_landmarks_list = face_recognition.face_landmarks(rgb)
    except Exception as e:
        print("⚠️ Skipping frame due to image format issue:", e)
        continue

    for face_location, landmarks in zip(face_locations, face_landmarks_list):
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)

        if is_blinking(landmarks):
            blink_count += 1
            cv2.putText(frame, "Blink Detected!", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            blink_count = 0

        if blink_count == 1 and not photo_captured:
            print("🟢 Blink detected! Capturing your face...")
            photo_captured = True
            file_path = os.path.join(SAVE_FOLDER, "temp_capture.jpg")
            cv2.imwrite(file_path, frame)
            break

    cv2.imshow("Blink to Capture", frame)
    if photo_captured or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------- STEP 2: ENCODE AND SAVE ----------
if not file_path:
    print("❌ No face captured. Exiting.")
    exit()

img = face_recognition.load_image_file(file_path)
encodings = face_recognition.face_encodings(img)
if len(encodings) == 0:
    print("❌ No face found in captured image.")
    exit()

encoding = encodings[0]

data = load_encodings()
matches = face_recognition.compare_faces(list(data.values()), encoding)
name = None

if True in matches:
    match_index = matches.index(True)
    name = list(data.keys())[match_index]
    print(f"✅ Welcome back, {name}!")
else:
    name = input("Enter your name: ")
    save_encoding(name, encoding)
    print(f"✅ Face saved as {name}")

# ---------- STEP 3: ATTENDANCE MARK ----------
attendance_file = "Attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Time\n")

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(attendance_file, "a") as f:
    f.write(f"{name},{now}\n")

print(f"🕒 Attendance marked for {name} at {now}")
print("✅ All done!")










