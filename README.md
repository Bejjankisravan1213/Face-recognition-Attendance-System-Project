# 🤖 Face Recognition Attendance System

> An AI-powered attendance system that uses real-time face detection via webcam to automatically register and recognize individuals — no manual input needed.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![face_recognition](https://img.shields.io/badge/face__recognition-1.3.0-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📌 About the Project

This project automates the attendance marking process using computer vision and facial recognition. When a person stands in front of the webcam, the system either recognizes their face (if already registered) or prompts them to register as a new user. The attendance is automatically saved with a timestamp.

This eliminates the need for manual roll calls, sign-in sheets, or ID card scanning — making it ideal for classrooms, offices, and events.

---

## ✨ Features

- 📷 **Live webcam capture** — captures face directly from the camera with a single keypress
- 🧠 **Face encoding & recognition** — converts faces into unique numerical encodings using `face_recognition`
- 🆕 **Auto-registration** — new faces are registered on first capture and saved locally
- ✅ **Smart matching** — compares live face against all stored encodings to identify the person
- 💾 **Persistent storage** — face encodings saved in a `.pkl` file, images in `Captured_Images/` folder
- 🖥️ **Visual feedback** — displays the recognized name directly on the video frame

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core language |
| OpenCV (`cv2`) | Webcam access, image display, video processing |
| face_recognition | Face detection and encoding (built on dlib) |
| NumPy | Image array manipulation |
| Pickle | Storing and loading face encodings |
| OS | File and directory management |

---

## 📂 Project Structure

```
Face-recognition-Attendance-System-Project/
│
├── AttendanceProject.py     # Main application logic
├── main.py                  # Entry point (same logic, alternate run file)
├── Attendance.csv           # Auto-generated attendance log
├── encodings.pkl            # Stored face encodings (auto-created on first run)
├── Captured_Images/         # Saved face photos (auto-created on first run)
└── README.md
```

---

## ⚙️ How It Works

```
1. Camera opens → user positions their face
2. User presses 's' to capture the frame
3. Face encoding is extracted from the image
4. System checks encoding against all saved encodings
   ├── Match found  → Displays recognized name on screen ✅
   └── No match     → Asks for name → Saves new face + encoding 🆕
5. Result displayed on webcam window for 3 seconds
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or above
- A working webcam
- Windows OS (uses `cv2.CAP_DSHOW` for DirectShow)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Bejjankisravan1213/Face-recognition-Attendance-System-Project.git
cd Face-recognition-Attendance-System-Project

# 2. Install required libraries
pip install opencv-python face_recognition numpy

# Note: face_recognition requires cmake and dlib
# On Windows, install cmake first:
# pip install cmake
# pip install dlib
# pip install face_recognition
```

### Run the Project

```bash
python AttendanceProject.py
```

### Controls

| Key | Action |
|-----|--------|
| `s` | Capture face from webcam |
| `q` | Quit the application |

---

## 📋 Sample Output

```
🎥 Position your face in front of the camera. Press 's' to capture your face.
🟢 Face captured!
✅ Recognized face: Sravan Kumar
```

Or for a new user:
```
🆕 No existing data found.
Enter your name: Sravan Kumar
✅ Face registered as: Sravan Kumar
```

---

## 🔮 Future Improvements

- [ ] Add CSV-based attendance logging with timestamps
- [ ] Build a web dashboard to view attendance records
- [ ] Support for multiple faces in a single frame
- [ ] Add a GUI using Tkinter or PyQt
- [ ] Email alerts for attendance confirmation

---

## 👤 Author

**Sravan Kumar Bejjanki**
- GitHub: [@Bejjankisravan1213](https://github.com/Bejjankisravan1213)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project useful, please consider giving it a star!
