import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import pytesseract
from PIL import Image
import io
import base64
import os
import uuid
import tempfile
import shutil
from pathlib import Path
from typing import Optional


from inference import run_pipeline, load_model
from compressmain import compress_image_bytes_to_jpeg
from lowLight import load_lol_model, enhance_image


# Create uploads folder if it doesn't exist
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

app = FastAPI()
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'

# Enable CORS for your React/Vite frontend (usually port 5173 or 5174)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models as module-level globals so Uvicorn/ASGI can use them too
try:
    deblur_model = load_model("experiments/pretrained_models/NAFNet-GoPro-width64.pth")
    denoise_model = load_model("experiments/pretrained_models/NAFNet-SIDD-width64.pth")
    lol_model = load_lol_model()
except Exception as e:
    deblur_model = None
    denoise_model = None
    lol_model = None
    print(f"Warning: Could not load models at startup: {e}")

def file_to_numpy(file_bytes):
    """Helper to convert uploaded bytes to a NumPy array (RGB)"""
    image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    return np.array(image)


def decode_video_to_frames(video_bytes, max_frames=300):
    """Extract all video frames from bytes."""
    temp_dir = Path(tempfile.mkdtemp())
    video_path = temp_dir / "upload_video.mp4"
    video_path.write_bytes(video_bytes)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Unable to open uploaded video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(frames) >= max_frames:
            break

        frames.append(frame)

    cap.release()
    return frames, fps, temp_dir


def encode_frames_to_video(frames, fps, output_path=None):
    """Write frames to MP4 bytes and return bytes."""
    if len(frames) == 0:
        raise ValueError("No frames to encode")

    h, w = frames[0].shape[:2]
    temp_file = Path(output_path) if output_path else Path(tempfile.mkstemp(suffix='.mp4')[1])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # type: ignore
    writer = cv2.VideoWriter(str(temp_file), fourcc, fps, (w, h))

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()

    video_data = temp_file.read_bytes()
    try:
        temp_file.unlink(missing_ok=True)
    except Exception:
        pass

    return video_data


def process_video_frames(frames, mode, quality=70, intensity=3.0, frame_step=3):
    """Apply enhancement/compression on every `frame_step`-th frame and keep the rest unchanged."""
    processed_frames = []

    for idx, frame in enumerate(frames):
        if idx % frame_step != 0:
            # Keep unprocessed frame to save CPU
            processed_frames.append(frame)
            continue

        # Video frames from OpenCV are BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if mode in ("Deblur", "Denoise", "Both"):
            enhance_mode = "Deblur" if mode == "Deblur" else "Denoise" if mode == "Denoise" else "Both"
            enhanced = run_pipeline(rgb_frame, deblur_model, denoise_model, 720, enhance_mode)
            bgr_frame = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        elif mode == "Compress":
            # lossily compress and decompress to keep 3-channel frame
            _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            decompress = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            bgr_frame = decompress
        elif mode == "LowLight":
            # Convert to PIL for low light enhancement
            pil_img = Image.fromarray(rgb_frame)
            enhanced_pil = enhance_image(lol_model, pil_img, intensity)
            enhanced_rgb = np.array(enhanced_pil)
            bgr_frame = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        else:
            bgr_frame = frame

        processed_frames.append(bgr_frame)

    return processed_frames


@app.post("/process-video")
async def process_video(file: UploadFile = File(...), mode: str = "Both", quality: int = 70, intensity: float = 3.0, frame_step: int = 3):
    """Endpoint for video uploads: split to frames, process sampled frames, reassemble."""
    try:
        contents = await file.read()
        frames, fps, temp_dir = decode_video_to_frames(contents)
        if not frames:
            raise HTTPException(status_code=400, detail="Video contains no frames")

        processed_frames = process_video_frames(frames, mode, quality, intensity, frame_step)
        video_bytes = encode_frames_to_video(processed_frames, fps)

        # Clean up temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)

        b64 = base64.b64encode(video_bytes).decode('utf-8')
        return {"video_url": f"data:video/mp4;base64,{b64}", "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


@app.post("/ocr")
async def get_ocr_text(file: UploadFile = File(...)):
    """Endpoint specifically for Tesseract text recognition"""
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        
        # Perform OCR
        text = pytesseract.image_to_string(pil_img)
        
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

@app.post("/process-image")
async def process_image(file: UploadFile = File(...), mode: str = "Deblur"):
    """Endpoint for heavy processing using inference.py or video->frame->video path."""
    is_video = (file.content_type and file.content_type.startswith("video")) or file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    if is_video:
        # Reuse video pipeline with every-3rd-frame processing by default
        return await process_video(file, mode=mode, quality=70, frame_step=3)

    contents = await file.read()
    try:
        image_np = file_to_numpy(contents)

        # Run your custom pipeline logic
        processed_np = run_pipeline(image_np, deblur_model, denoise_model, 720, mode)

        # Convert the resulting NumPy array back to base64 for the frontend
        # OpenCV uses BGR by default, so we convert back to BGR for encoding
        success, buffer = cv2.imencode('.jpg', cv2.cvtColor(processed_np, cv2.COLOR_RGB2BGR))
        if not success:
            raise ValueError("Could not encode image")
            
        img_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        return {
            "image_url": f"data:image/jpeg;base64,{img_str}",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/compress")
async def compress_image(file: UploadFile = File(...), quality: int = 70):
    """Endpoint for image compression using JPEG or video->frame->video path."""
    is_video = (file.content_type and file.content_type.startswith("video")) or file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    if is_video:
        return await process_video(file, mode="Compress", quality=quality, frame_step=3)

    contents = await file.read()

    try:
        # Compress using JPEG
        compressed_bytes = compress_image_bytes_to_jpeg(contents, quality)
        
        # Encode to base64
        img_str = base64.b64encode(compressed_bytes).decode('utf-8')
        
        return {
            "image_url": f"data:image/jpeg;base64,{img_str}",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")


@app.post("/lowlight")
async def enhance_lowlight(file: UploadFile = File(...), intensity: float = 3.0):
    """Endpoint for low light enhancement using LOL model or video->frame->video path."""
    is_video = (file.content_type and file.content_type.startswith("video")) or file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    if is_video:
        return await process_video(file, mode="LowLight", intensity=intensity, frame_step=3)

    contents = await file.read()

    try:
        # Enhance low light
        pil_img = Image.open(io.BytesIO(contents))
        enhanced_pil = enhance_image(lol_model, pil_img, intensity)
        
        # Convert back to bytes
        buffer = io.BytesIO()
        enhanced_pil.save(buffer, format='JPEG')
        enhanced_bytes = buffer.getvalue()
        
        # Encode to base64
        img_str = base64.b64encode(enhanced_bytes).decode('utf-8')
        
        return {
            "image_url": f"data:image/jpeg;base64,{img_str}",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Low light enhancement failed: {str(e)}")


if __name__ == "__main__":
    if deblur_model is None or denoise_model is None:
        deblur_model = load_model("models/NAFNet-GoPro-width64.pth")
        denoise_model = load_model("models/NAFNet-SIDD-width64.pth")
    lol_model = load_lol_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
