import cv2
import face_recognition
import os

# Create directory to store captured images if it doesn't exist
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
    if key == ord('s'):  # Press 's' to capture face
        img_path = "Images_Attendance/captured_face.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Face captured and saved as {img_path}")
        break
    elif key == ord('q'):  # Press 'q' to quit
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Encode the captured face
img = face_recognition.load_image_file(img_path)
face_encoding = face_recognition.face_encodings(img)[0]
known_face_encodings = [face_encoding]
known_face_names = ["New User"]  # Replace with the user's name or prompt for input


