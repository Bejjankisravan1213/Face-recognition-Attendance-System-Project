import cv2
import face_recognition
import numpy as np

# Load reference image
reference_image = face_recognition.load_image_file('Images_Attendance/reference.jpg')
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

# Name for the reference
reference_name = "Person_Name"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encode
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare face with reference
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        face_distance = face_recognition.face_distance([reference_encoding], face_encoding)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = reference_name
        else:
            name = "Unknown"

        # Scale back up face locations
        top, right, bottom, left = face_location
        top, right, bottom, left = top*4, right*4, bottom*4, left*4

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, f"{name}", (left+6, bottom-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Show the webcam
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
