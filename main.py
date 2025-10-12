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

# ---------- UTILS ----------
def load_encodings():
    with open(ENCODING_FILE, "rb") as f:
        return pickle.load(f)

def save_encoding(name, encoding):
    data = load_encodings()
    data[name] = encoding
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(data, f)

# ---------- STEP 1: CAPTURE FACE ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("🎥 Position your face in front of the camera. Press 's' to capture your face.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if ret:
            # Convert to RGB 8-bit, ensure memory layout and dtype for face_recognition
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb = np.ascontiguousarray(image_rgb, dtype=np.uint8)
            print("Shape:", image_rgb.shape)
            print("Dtype:", image_rgb.dtype)
            print("🟢 Face captured!")
            break
        else:
            print("Error capturing frame, try again.")
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# ---------- STEP 2: ENCODE FACE ----------
encodings = face_recognition.face_encodings(image_rgb)
if len(encodings) == 0:
    print("❌ No face detected. Please try again.")
    exit()

encoding = encodings[0]

# ---------- STEP 3: CHECK IF FACE EXISTS ----------
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
    # Save the face image for record
    cv2.imwrite(os.path.join(SAVE_FOLDER, f"{name}.jpg"), frame)
    print(f"✅ Face saved as: {name}")

# ---------- OPTIONAL: DISPLAY RECOGNIZED FACE ----------
cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
cv2.imshow("Recognized Face", frame)
cv2.waitKey(3000)
cv2.destroyAllWindows()


















