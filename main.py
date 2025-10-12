import cv2
import face_recognition
import os
import pickle
import numpy as np

ENCODING_FILE = "encodings.pkl"
SAVE_FOLDER = "Captured_Images"

# Setup folders and encoding file
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

if not os.path.exists(ENCODING_FILE):
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump({}, f)

def load_encodings():
    with open(ENCODING_FILE, "rb") as f:
        return pickle.load(f)

def save_encoding(name, encoding):
    data = load_encodings()
    data[name] = encoding
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(data, f)

# Capture face
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("🎥 Position your face in front of the camera and press 's' to capture...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        img_path = os.path.join(SAVE_FOLDER, "temp.jpg")
        # Save using imwrite and absolute path
        cv2.imwrite(img_path, frame)
        print("🟢 Face captured!")
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Read image safely
image_bgr = cv2.imread(img_path)
if image_bgr is None:
    print("❌ Failed to read captured image. Make sure path is correct and has no spaces.")
    exit()

# Convert to RGB and ensure uint8
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_rgb = np.array(image_rgb, dtype=np.uint8)

# Check that image is not empty
if image_rgb.size == 0:
    print("❌ Image is empty. Cannot process.")
    exit()

# Encode face
encodings = face_recognition.face_encodings(image_rgb)
if len(encodings) == 0:
    print("❌ No face detected in the captured image. Try again.")
    exit()

encoding = encodings[0]

# Check if face exists
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
    # Save permanent copy of the face
    cv2.imwrite(os.path.join(SAVE_FOLDER, f"{name}.jpg"), image_bgr)
    print(f"✅ Face saved as: {name}")


















