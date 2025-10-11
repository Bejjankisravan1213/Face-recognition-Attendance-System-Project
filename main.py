import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# Folder to store known faces
path = 'Images_Attendance'
if not os.path.exists(path):
    os.makedirs(path)

# Load existing images
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if len(encode) > 0:
            encodeList.append(encode[0])
    return encodeList

# Function to mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)
        lines = f.readlines()
        nameList = [line.split(',')[0] for line in lines]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S,%d/%m/%Y')
            f.write(f'\n{name},{dtString}')

# Encode all known faces
encodeListKnown = findEncodings(images)
print("Encoding Complete:", classNames)

# Start webcam
cap = cv2.VideoCapture(0)
print("Starting camera... Press 'q' to exit.")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        name = "Unknown"

        if encodeListKnown:
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()

        y1, x2, y2, x1 = [v * 4 for v in faceLoc]

        if name == "Unknown":
            color = (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, "NEW FACE DETECTED", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow('Webcam', img)
            cv2.waitKey(1000)

            new_name = input("Enter name for this new face: ").strip()
            if new_name:
                img_path = os.path.join(path, f"{new_name}.jpg")
                cv2.imwrite(img_path, img[y1:y2, x1:x2])  # Save cropped face
                classNames.append(new_name)
                images.append(cv2.imread(img_path))
                encodeListKnown = findEncodings(images)
                print(f"✅ Added new face: {new_name}")
                markAttendance(new_name)
        else:
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
