# media-app

A lightweight multimedia processing project with a Python backend for image/video enhancement and a frontend UI built with Vite + React.

## Project structure

- `backend/` - Python modules for image/video handling, inference, compression, low-light enhancement, and super-resolution (NAFNet, BasicSR). 
- `frontend/` - React + Vite web app UI for uploading media, invoking backend processing, and previewing output.
- `IO/` - input/output directories for artefacts and temporary media files.

## Quick start

1. Backend: create and activate Python venv
   ```bash
   cd backend
   python -m venv .venv
   .venv/Scripts/activate   # Windows
   pip install -r requirements.txt
   ```
2. Run backend service (example):
   ```bash
   python main.py
   ```
3. Frontend: install and run
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

Open the dev frontend URL (usually `http://localhost:5173`) and use the UI to process media.

## Features

- Image and video conversion utilities (`Img_to_Video.py`, `Video_to_img.py`).
- AI-based enhancement backends (`lowLight.py`, `inference.py`, `NAFNet` models).
- Compression and Huffman coding tools.

## Notes

- Model weights are stored under `models/`.
- `basicsr/` includes existing SR architecture and loaders.
- This repository was set up primarily for rapid experimentation and demo workflows.

## License

Add your license and contributions policy here.
