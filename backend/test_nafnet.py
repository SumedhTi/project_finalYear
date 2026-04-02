import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_VIDEO = os.path.join(BASE_DIR, "input.mp4")
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
OUTPUT_FRAMES_DIR = os.path.join(BASE_DIR, "output_frames")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "output_video.mp4")

MODEL_PATH = os.path.join(BASE_DIR, "experiments/pretrained_models/NAFNet-GoPro-width64.pth")

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

sys.path.append(BASE_DIR)

from basicsr.models.archs.NAFNet_arch import NAFNet


def extract_frames(video_path, output_folder, target_fps=10):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_interval = max(int(original_fps / target_fps), 1)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved:05d}.png")
            cv2.imwrite(filename, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Extracted {saved} frames")


def load_model():
    model = NAFNet(
        img_channel=3,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1]
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def process_frames(model, input_folder, output_folder):
    files = sorted(os.listdir(input_folder))

    for file in tqdm(files):
        path = os.path.join(input_folder, file)
        img = cv2.imread(path)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            output = model(img_input)

        output = output.squeeze().permute(1, 2, 0).numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(output_folder, file), output)


def frames_to_video(frames_folder, output_video, fps=10):
    files = sorted(os.listdir(frames_folder))

    first_frame = cv2.imread(os.path.join(frames_folder, files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for file in files:
        frame = cv2.imread(os.path.join(frames_folder, file))
        out.write(frame)

    out.release()
    print("Video saved:", output_video)


if __name__ == "__main__":
    print("Step 1: Extracting frames...")
    extract_frames(INPUT_VIDEO, FRAMES_DIR, target_fps=10)

    print("Step 2: Loading model...")
    model = load_model()

    print("Step 3: Processing frames...")
    process_frames(model, FRAMES_DIR, OUTPUT_FRAMES_DIR)

    print("Step 4: Rebuilding video...")
    frames_to_video(OUTPUT_FRAMES_DIR, OUTPUT_VIDEO, fps=10)

    print("DONE")