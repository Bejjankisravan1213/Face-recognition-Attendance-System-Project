from datetime import datetime
import os

def mark_attendance(name):
    file_path = "Attendance.csv"
    # Create CSV if it does not exist
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("Name,Time\n")

    with open(file_path, "r+") as f:
        lines = f.readlines()
        names = [line.split(',')[0] for line in lines[1:]]  # skip header
        if name not in names:
            now = datetime.now()
            time_string = now.strftime("%Y-%m-%d %H:%M:%S")
            f.writelines(f"{name},{time_string}\n")
            print(f"Attendance marked for {name}")





