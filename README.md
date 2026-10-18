# SHAS — Suspicious Human Activity Detection and Alert System

SHAS is a single Flask application that serves its own HTML templates and static assets and performs video inference in a persistent Python process. It includes suspicious-activity recognition, weapon, accident, fire/explosion, and multiple-person detection, plus optional Telegram alerts.

## Project components

- `app.py` — Flask routes and WSGI entry point (`app:app`), including upload and health endpoints
- `stream_engine.py`, `A_Recognition.py`, `detect*.py`, `models/`, `model/`, `utils/` — inference code
- `templates/`, `static/` — Flask-served UI
- `requirements.txt` — general development/training dependencies
- `requirements-render.txt` — pinned Python 3.9 CPU serving stack
- `render.yaml` — Render web service, persistent disk, environment, and health-check configuration
- `scripts/prepare_model_assets.py` — optionally provisions model files from a private tar.gz URL
- `model_manifest.json` — expected model sizes and SHA-256 hashes used during provisioning and readiness checks

## Models and persistent data

Five files are required for all dashboard detection modules: `kni_axe.pt`, `ACCIDENT.pt`, `fire.pt`, `yolov5s.pt`, and `model_new.h5`. They are intentionally not stored in Git. Provision the original files in the directory set by `SHAS_MODEL_DIR`; do not replace them with dummy models. `person.pt` is also retained locally but is not referenced by the dashboard's current model configuration.

The Render Blueprint mounts a 10 GB persistent disk at `/var/data`, storing models in `/var/data/models`, uploads in `/var/data/uploads`, and SQLite data in `/var/data/user_data.db`. Set `SHAS_MODEL_ARCHIVE_URL` to a private `.tar.gz` archive containing the five model files (archive entries may be at the root or in subdirectories). If the URL requires a bearer token, set `SHAS_MODEL_ARCHIVE_TOKEN`. The startup helper only extracts the allow-listed files and verifies exact byte sizes and SHA-256 hashes against `model_manifest.json`. `/health/models` reports ready only after integrity checks and successful loading of every YOLO and action-recognition model. Without model files, it returns HTTP 503 and detection is unavailable.

The original weights in the local working copy are preserved and remain excluded from Git. Create a private archive for transfer without adding it to the repository:

```powershell
tar -czf shas-models.tar.gz kni_axe.pt ACCIDENT.pt fire.pt yolov5s.pt model_new.h5
```

Upload that archive to private object storage and configure its short-lived, read-only URL only in Render's environment settings. Keep archive credentials and signed URLs private. Alternatively, transfer the files to `/var/data/models` using an authorized Render shell/session.

## Local development

Python 3.9 is required for the pinned legacy TensorFlow action model. For a local setup, install the project dependencies appropriate to your OS and activate a virtual environment. The general `requirements.txt` supports development and training; the Render requirements target Linux CPU serving.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Open `http://127.0.0.1:5000`. `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are optional; alerts are disabled if they are not set. The app accepts `SHAS_DATA_DIR`, `SHAS_MODEL_DIR`, `SHAS_UPLOAD_DIR`, `SHAS_DATABASE_PATH`, and `SHAS_MAX_UPLOAD_BYTES` to place writable data outside the source directory. In production, set `SHAS_ENV=production` and provide a random private `SHAS_SECRET_KEY` of at least 32 random bytes through the host's environment settings.

## Deploy the backend on Render

This app is not a Vercel serverless application. Its multipart `/video_feed` endpoint is long-running, OpenCV/PyTorch/TensorFlow load native libraries and local model files, and inference runs inside the server process. Vercel previously rejected the generated Python function bundle at approximately 5,933.56 MB against a 500 MB limit. Do not omit model weights to get that deployment through.

Connect this GitHub repository to Render and create a Blueprint from `render.yaml` (or create a web service with the same settings). The configuration selects Python 3.9.18, CPU-only Torch, one Gunicorn worker, four request threads, a 4 CPU / 8 GB plan, Singapore region, and a 10 GB persistent disk. The existing templates and static files stay with Flask, so there is no separate Vercel frontend or cross-origin API configuration in this repository.

Before expecting full detection, provision all five required model files on the persistent disk or configure `SHAS_MODEL_ARCHIVE_URL`. Set optional Telegram values in the provider dashboard, never in Git. Then verify:

- `https://<service>.onrender.com/` — Flask UI
- `https://<service>.onrender.com/healthz` — process health
- `https://<service>.onrender.com/health/models` — model-file readiness (200 only when all five are present)

The service runs inference on CPU; video inference may be slower than a GPU workstation. A Render service cannot access a camera attached to the user's browser or local computer. Uploaded videos can be processed by the server. A persistent disk attaches to one service instance, so the configured single process/instance avoids relying on ephemeral local storage or in-memory coordination across replicas.

## Security

- SQLite and uploaded media are stored on the persistent disk. Back up this disk/database according to your retention needs.
- Registration stores Werkzeug scrypt password hashes. Existing plaintext passwords are upgraded in place after the owner's first successful login, preserving the current SQLite table and records.
- Dashboard, upload, stop-stream, and video-stream endpoints require an authenticated session. State-changing requests require a session-bound CSRF token. Session cookies are HttpOnly, SameSite=Lax, and Secure in production.
- Usernames, passwords, contact fields, file extensions, video signatures, upload names, and configured upload size are validated. Uploaded files receive collision-resistant names and detection sources are restricted to the upload and sample-video directories.
- The Render production Blueprint requires `SHAS_SECRET_KEY`; set it as a private environment variable before the first deploy. Never place it, Telegram credentials, archive URLs, or tokens in Git.
- Telegram bot tokens were present in earlier repository history and have been removed from the current source. Rotate previously exposed tokens; this cleanup does not rewrite Git history.
- Webcam capture works only when a camera is physically accessible to the Python host; remote browser webcam access is not implemented.
- Limit access to uploaded videos and detection output. Configure a production domain/TLS and review request-size limits before public use.
- Do not use model results as the sole basis for safety-critical decisions.
