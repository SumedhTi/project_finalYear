import cv2
import os

BASE = os.path.dirname(os.path.abspath(__file__))

os.chdir(BASE)

video_path = os.path.join(BASE, "sample.mp4")
output_folder = os.path.join(BASE, "frames")

print("CWD:", os.getcwd())
print("Video exists:", os.path.exists(video_path))

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

frame_interval = 4
frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        frame_path = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_count += 1

    frame_count += 1

cap.release()
cv2.destroyAllWindows()