from datetime import datetime

def mark_attendance(name):
    with open("Attendance.csv", "r+") as f:
        lines = f.readlines()
        names = [line.split(',')[0] for line in lines]
        if name not in names:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            f.writelines(f"{name},{timestamp}\n")
            print(f"Attendance marked for {name}")


