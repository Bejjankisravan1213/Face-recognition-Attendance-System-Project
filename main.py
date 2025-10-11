import cv2
import face_recognition
import os

# Create Images_Attendance folder if not exists
if not os.path.exists("Images_Attendance"):
    os.makedirs("Images_Attendance")

# Capture image from webcam
cap = cv2.VideoCapture(0)
print("Press 's' to capture your face...")

while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):  # press 's' to save face
        cv2.imwrite("Images_Attendance/captured_face.jpg", frame)
        print("Face captured!")
        break
    elif key == ord('q'):  # quit
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Encode captured face
imgCaptured = face_recognition.load_image_file("Images_Attendance/captured_face.jpg")
imgCapturedEncoding = face_recognition.face_encodings(imgCaptured)[0]
knownEncodings = [imgCapturedEncoding]
knownNames = ["You"]  # replace "You" with your name if you like
