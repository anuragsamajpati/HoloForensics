# HoloForensics

An advanced AI-powered forensic analysis system that reconstructs 3D crime scenes from multi-camera footage, performs intelligent video inpainting, and enables natural-language case querying with legal‑grade confidence scoring.

## Highlights

- Developed an advanced AI-powered forensic analysis system that reconstructs 3D crime scenes from multi-camera footage, performs intelligent video inpainting, and enables natural language case querying with legal-grade confidence scoring.
- Leveraged NeRF / 3D Gaussian Splatting, YOLOv8, ByteTrack + FastReID, and physics-informed Transformers to achieve multi-view object tracking, real-time 3D trajectory smoothing, and predictive motion modeling for forensic applications.
- Contributed to the development of an LLM-integrated reasoning pipeline using RAG with FAISS, Llama-3 / GPT, and vector embeddings, enabling semantic Q&A, evidence retrieval, and contextual analysis of reconstructed scenes.
- Built a full-stack forensic web application using Django, Next.js, Three.js, and FastAPI, supporting 3D visualization, timeline reconstruction, event detection, and forensic Q&A dashboards with integrated MLOps monitoring.

### Technologies Utilized

Python, PyTorch, TensorFlow, OpenCV, COLMAP, YOLOv8, NeRF / 3DGS, E2FGVI, FAISS, RAG, Llama-3 / GPT, Django, Next.js, Three.js, PostgreSQL, MinIO, Redis, Docker, GCP, FastAPI

## Quick Start (Local)

1. Create and activate a virtual environment
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
2. Install Python dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Django web app
   ```bash
   cd holoforensics/holoforensics_web
   python manage.py runserver
   ```
4. Open the dashboard
   - Dashboard URL: http://127.0.0.1:8000/

## Open the Full Website / Dashboard

- Local development URL: http://127.0.0.1:8000/

## Repository Notes

- Large binary artifacts (videos, models, zips) are tracked via **Git LFS**.
- Generated outputs (e.g., `Analysed Data/`) and local environment files are ignored via `.gitignore`.

## Contributing

See `CONTRIBUTING.md` for guidelines, branching, and PR process. A GitHub Actions workflow runs basic CI on every push and PR.
