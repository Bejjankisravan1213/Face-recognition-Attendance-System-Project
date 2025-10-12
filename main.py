import cv2
import face_recognition
import os
import pickle

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

# ---------- STEP 2: ENCODE FACE ----------
image = face_recognition.load_image_file(img_path)
encodings = face_recognition.face_encodings(image)

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
    print(f"✅ Face saved as: {name}")















