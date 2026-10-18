# SHAS — Suspicious Human Activity Detection and Alert System

SHAS is a monolithic Flask application for reviewing video and running suspicious-activity detection. The Flask app serves its own HTML templates and static assets; there is no separate frontend build. Detection uses OpenCV, PyTorch, YOLO models, and an action-recognition model. The app can also send Telegram alerts when configured.

## Features

- Flask login/registration screens and detection dashboard
- Video upload and multipart video-stream endpoints
- Weapon, accident, explosion, person, and action-recognition modules
- Optional Telegram notifications

## Project layout

- `app.py` — Flask application and routes; WSGI entry point is `app:app`
- `stream_engine.py`, `detect*.py`, `A_Recognition.py`, `models/`, `utils/` — inference and model code
- `templates/`, `static/` — Flask-served UI
- `test/`, `testweapon/`, `testaccident/`, `testexplosion/`, `testperson/` — sample input videos
- `Labels/`, `TRAINING/`, `train.py` — training data metadata and training utilities

## Requirements and model files

Use Python 3.10 or 3.11 on a Linux host with enough memory and disk for CPU PyTorch and OpenCV. Install `requirements.txt` in a virtual environment. The model weights (`*.pt`, `*.h5`) are intentionally excluded from Git because they are large binary artifacts. The current code expects `kni_axe.pt`, `ACCIDENT.pt`, `fire.pt`, `yolov5s.pt`, and `model_new.h5` in the project root; provision the matching files there before starting the app. Keep them in private object storage or a model registry, and do not commit them.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` for local development and set credentials as needed. Never commit `.env`.

- `TELEGRAM_BOT_TOKEN` — optional Telegram bot token
- `TELEGRAM_CHAT_ID` — optional destination chat/channel ID

If either Telegram variable is unset, Telegram alerts are disabled. The current application uses a local SQLite database (`user_data.db`) for registration and login; it is ignored by Git and is not a production persistent database. Configure durable database storage and secure password handling before exposing registration/login publicly.

## Run locally

```bash
python app.py
```

Open `http://127.0.0.1:5000`. For a production WSGI server, install the requirements and run:

```bash
gunicorn --workers 1 --threads 8 --timeout 120 app:app
```

Use a single worker because the stream controller and loaded models are process-local. Configure a reverse proxy and TLS, restrict access, and provide persistent storage for uploads and the database if those features are needed.

## Deployment architecture

This is a monolithic Flask app, not a static frontend. Vercel is not a suitable host for the complete application: `/video_feed` returns a long-running multipart stream, inference loads local model files and native computer-vision dependencies, and camera access/background work require a persistent process. Deploy the application to a VM or container host that supports long-lived Python processes, persistent model files, and sufficient CPU/GPU resources. Vercel may host a separately built frontend in a future split architecture, but this repository has no such frontend today. No Vercel configuration is included.

The deployment host must provide the model weights outside Git and set Telegram environment variables if alerts are desired. Validate the stream, uploads, database persistence, resource use, and access controls on that host. Sample videos and trained models do not make a server-side deployment compatible with browser camera access.

## Security and limitations

- Rotate the Telegram bot token that was previously embedded in `stream_engine.py`; the code now reads it from the environment.
- The existing login implementation stores plaintext passwords and is for demonstration only. Do not expose it to untrusted users until password hashing, session authentication, CSRF protection, and persistent database storage are implemented.
- Restrict upload size and file types at the reverse proxy/application layer before public deployment.
- Do not use detection output as the sole basis for safety-critical decisions.
