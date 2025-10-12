import cv2
import face_recognition
import os
import pickle
import numpy as np

# ---------- CONFIG ----------
ENCODING_FILE = "encodings.pkl"
SAVE_FOLDER = "Captured_Images"

# ---------- SETUP ----------
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

if not os.path.exists(ENCODING_FILE):
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump({}, f)

# ---------- UTILITIES ----------
def load_encodings():
    with open(ENCODING_FILE, "rb") as f:
        return pickle.load(f)

def save_encoding(name, encoding):
    data = load_encodings()
    data[name] = encoding
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(data, f)

# ---------- STEP 1: CAPTURE FACE ----------
cap = cv2.VideoCapture(0)
print("🎥 Position your face in front of the camera and press 's' to capture...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        img_path = os.path.join(SAVE_FOLDER, "temp.jpg")
        cv2.imwrite(img_path, frame)
        print("🟢 Face captured!")
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# ---------- STEP 2: READ IMAGE AND CONVERT TO RGB 8-BIT ----------
# Use OpenCV to read image
image_bgr = cv2.imread(img_path)
if image_bgr is None or image_bgr.size == 0:
    print("❌ Failed to read captured image.")
    exit()

# Convert BGR -> RGB and ensure dtype uint8
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)

# ---------- STEP 3: ENCODE FACE ----------
encodings = face_recognition.face_encodings(image_rgb)
if len(encodings) == 0:
    print("❌ No face detected. Please try again.")
    exit()

encoding = encodings[0]

# ---------- STEP 4: CHECK IF FACE EXISTS ----------
data = load_encodings()
matches = face_recognition.compare_faces(list(data.values()), encoding)
name = None

if True in matches:
    match_index = matches.index(True)
    name = list(data.keys())[match_index]
    print(f"✅ Face recognized: {name}")
else:
    name = input("Enter your name: ").strip()
    save_encoding(name, encoding)
    cv2.imwrite(os.path.join(SAVE_FOLDER, f"{name}.jpg"), image_bgr)
    print(f"✅ Face saved as: {name}")

















