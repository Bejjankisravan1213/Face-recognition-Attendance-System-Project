import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Folder to save captured faces
path = 'Images_Attendance'
if not os.path.exists(path):
    os.makedirs(path)

# Function to mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        lines = f.readlines()
        namesList = [line.split(',')[0] for line in lines]
        if name not in namesList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'{name},{dtString}\n')
            print(f"Attendance marked for {name}")

# Capture face dynamically from webcam
cap = cv2.VideoCapture(0)
print("Press 's' to capture your face for attendance")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow('Webcam', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Press 's' to capture
        imgCaptured = frame
        imgPath = os.path.join(path, 'captured_face.jpg')
        cv2.imwrite(imgPath, imgCaptured)
        print("Face captured for attendance!")
        break
    elif key == ord('q'):  # Press 'q' to quit
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# Encode captured face
imgCaptured = face_recognition.load_image_file(imgPath)
encodeCaptured = face_recognition.face_encodings(imgCaptured)[0]

encodeListKnown = [encodeCaptured]
classNames = ["STUDENT"]  # Change to your name or get from input
print("Encoding complete, starting live attendance...")

# Start live face recognition
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        continue

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)
            y1, x2, y2, x1 = [coord * 4 for coord in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

