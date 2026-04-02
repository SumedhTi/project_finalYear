import cv2
import os

BASE = os.path.dirname(os.path.abspath(__file__))


frames_folder = os.path.join(BASE, "frames")
output_video = os.path.join(BASE, "output.mp4")

frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".jpg")])

first_frame_path = os.path.join(frames_folder, frame_files[0])
frame = cv2.imread(first_frame_path)

height, width, _ = frame.shape

fps = 10

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for file in frame_files:
    frame_path = os.path.join(frames_folder, file)
    frame = cv2.imread(frame_path)
    out.write(frame)

out.release()
cv2.destroyAllWindows()