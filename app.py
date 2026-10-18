import hmac
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import timedelta
from functools import wraps
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, Response, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from stream_engine import (
    MODEL_CONFIGS,
    generate_frames,
    get_action_model_and_classes,
    get_new_stream_token,
    get_yolo_model,
    stop_all_streams,
)
from scripts.prepare_model_assets import verify_model_files

app = Flask(__name__)
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv('SHAS_DATA_DIR', str(ROOT))).expanduser().resolve()
UPLOAD_FOLDER = Path(os.getenv('SHAS_UPLOAD_DIR', str(DATA_DIR / 'uploads'))).expanduser().resolve()
DATABASE_PATH = Path(os.getenv('SHAS_DATABASE_PATH', str(DATA_DIR / 'user_data.db'))).expanduser().resolve()
MODEL_DIR = Path(os.getenv('SHAS_MODEL_DIR', str(ROOT))).expanduser().resolve()
MODEL_INTEGRITY_ERRORS = verify_model_files(MODEL_DIR)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['DATABASE_PATH'] = str(DATABASE_PATH)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('SHAS_MAX_UPLOAD_BYTES', str(512 * 1024 * 1024)))
IS_PRODUCTION = os.getenv('SHAS_ENV', '').lower() == 'production'
SECRET_KEY = os.getenv('SHAS_SECRET_KEY', '')
if IS_PRODUCTION and not SECRET_KEY:
    raise RuntimeError('SHAS_SECRET_KEY must be set for production sessions and CSRF protection')
app.config.update(
    SECRET_KEY=SECRET_KEY or secrets.token_hex(32),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)

ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.m4v', '.mov', '.avi', '.webm', '.mkv', '.mpeg', '.mpg'}


def db_connect():
    return sqlite3.connect(app.config['DATABASE_PATH'], timeout=30)


@contextmanager
def db_connection():
    connection = db_connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db():
    with db_connection() as connection:
        connection.execute(
            'CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)'
        )


init_db()


@app.context_processor
def template_configuration():
    return {
        'telegram_enabled': bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID')),
        'csrf_token': csrf_token,
    }


def csrf_token():
    token = session.get('_csrf_token')
    if token is None:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token


@app.before_request
def protect_state_changing_requests():
    if request.method not in {'GET', 'HEAD', 'OPTIONS', 'TRACE'}:
        expected = session.get('_csrf_token', '')
        supplied = request.form.get('csrf_token', '') or request.headers.get('X-CSRFToken', '')
        if not expected or not supplied or not hmac.compare_digest(expected, supplied):
            abort(400, description='Invalid or missing CSRF token')


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('username'):
            if request.path in {'/upload_video', '/stop_stream'}:
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('index'))
        return view(*args, **kwargs)
    return wrapped


@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok', 'service': 'shas'})


@app.route('/health/models')
def model_health():
    required = ('kni_axe.pt', 'ACCIDENT.pt', 'fire.pt', 'yolov5s.pt', 'model_new.h5')
    models = {name: name not in MODEL_INTEGRITY_ERRORS for name in required}
    if MODEL_INTEGRITY_ERRORS:
        return jsonify({'status': 'models_missing_or_invalid', 'models': models}), 503
    loaded = {}
    try:
        for module in MODEL_CONFIGS:
            get_yolo_model(module)
            loaded[module] = True
        action_model, classes = get_action_model_and_classes()
        if action_model is None or not classes:
            raise RuntimeError('Action model or class mapping is unavailable')
        loaded['action'] = True
    except Exception as error:
        failed_module = next((name for name in MODEL_CONFIGS if name not in loaded), 'action')
        return jsonify({
            'status': 'model_load_failed',
            'models': models,
            'loaded': loaded,
            'failed_module': failed_module,
            'error_type': type(error).__name__,
        }), 503
    return jsonify({'status': 'ready', 'models': models, 'loaded': loaded}), 200


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({'error': 'Upload exceeds the configured size limit'}), 413

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        password = request.form.get('password', '')
        with db_connection() as connection:
            row = connection.execute(
                'SELECT rowid, name, password FROM user WHERE name = ? ORDER BY rowid LIMIT 1',
                (name,),
            ).fetchone()
            if row is None:
                valid = False
            elif (row[2] or '').startswith(('scrypt:', 'pbkdf2:')):
                try:
                    valid = check_password_hash(row[2], password)
                except (ValueError, TypeError):
                    valid = False
            else:
                # Upgrade a legacy plaintext password only after a successful login.
                valid = hmac.compare_digest(str(row[2]), password)
                if valid:
                    connection.execute(
                        'UPDATE user SET password = ? WHERE rowid = ?',
                        (generate_password_hash(password), row[0]),
                    )
        if not valid:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided, Try Again')
        session.clear()
        session['username'] = row[1]
        session.permanent = True
        return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        password = request.form.get('password', '')
        mobile = request.form.get('phone', '').strip()
        email = request.form.get('email', '').strip()
        if not re.fullmatch(r'[A-Za-z0-9_.-]{3,64}', name):
            return render_template('index.html', msg='Username must be 3–64 letters, digits, dots, underscores, or hyphens')
        if len(password) < 12 or len(password) > 128:
            return render_template('index.html', msg='Password must be between 12 and 128 characters')
        if len(email) > 254 or (email and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email)):
            return render_template('index.html', msg='Enter a valid email address')
        if len(mobile) > 32 or (mobile and not re.fullmatch(r'[0-9+() .-]+', mobile)):
            return render_template('index.html', msg='Enter a valid phone number')
        with db_connection() as connection:
            exists = connection.execute('SELECT 1 FROM user WHERE name = ? LIMIT 1', (name,)).fetchone()
            if exists:
                return render_template('index.html', msg='That username is already registered')
            connection.execute(
                'INSERT INTO user(name, password, mobile, email) VALUES (?, ?, ?, ?)',
                (name, generate_password_hash(password), mobile, email),
            )
        return render_template('index.html', msg='Successfully Registered. Please sign in.')

    return render_template('index.html')


@app.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route that yields multipart JPEG frames."""
    module = request.args.get('module', 'weapon')
    source = request.args.get('source', '0')
    token = get_new_stream_token()
    return Response(
        generate_frames(module, source, token),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stop_stream', methods=['POST'])
@login_required
def stop_stream():
    """Stops any currently active detection stream and releases resources."""
    stop_all_streams()
    return jsonify({'status': 'stopped', 'message': 'All detection streams stopped and resources released'})


@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    """Handles video file uploads from dashboard forms."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    filename = secure_filename(file.filename)
    extension = Path(filename).suffix.lower()
    if not filename or extension not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({'error': 'Unsupported video file type'}), 415
    header = file.stream.read(12)
    file.stream.seek(0)
    signatures = {
        '.mp4': header[4:8] == b'ftyp',
        '.m4v': header[4:8] == b'ftyp',
        '.mov': header[4:8] == b'ftyp',
        '.avi': header[:4] == b'RIFF' and header[8:12] == b'AVI ',
        '.webm': header.startswith(b'\x1aE\xdf\xa3'),
        '.mkv': header.startswith(b'\x1aE\xdf\xa3'),
        '.mpeg': header.startswith(b'\x00\x00\x01'),
        '.mpg': header.startswith(b'\x00\x00\x01'),
    }
    if not signatures.get(extension, False):
        return jsonify({'error': 'File content does not match a supported video format'}), 415
    filename = f"{secrets.token_hex(8)}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    relative_path = f"uploads/{filename}"
    return jsonify({
        'status': 'success',
        'filename': filename,
        'path': relative_path
    })


# Legacy form POST routes - now seamlessly start detection inside the dashboard without subprocess windows
@app.route('/video', methods=['GET', 'POST'])
@login_required
def video():
    if request.method == 'POST':
        filename = request.form.get('filename', '')
        source = ('test/' + filename) if filename else 'test/test1.mp4'
        return render_template('userlog.html', initial_module='analyse', initial_source=source)
    return render_template('userlog.html')


@app.route('/Live1', methods=['GET', 'POST'])
@login_required
def Live1():
    if request.method == 'POST':
        filename = request.form.get('filename1', '')
        source = ('testweapon/' + filename) if filename else '0'
        return render_template('userlog.html', initial_module='weapon', initial_source=source)
    return render_template('userlog.html')


@app.route('/Live2', methods=['GET', 'POST'])
@login_required
def Live2():
    if request.method == 'POST':
        filename = request.form.get('filename2', '')
        source = ('testaccident/' + filename) if filename else '0'
        return render_template('userlog.html', initial_module='accident', initial_source=source)
    return render_template('userlog.html')


@app.route('/Live3', methods=['GET', 'POST'])
@login_required
def Live3():
    if request.method == 'POST':
        filename = request.form.get('filename3', '')
        source = ('testexplosion/' + filename) if filename else '0'
        return render_template('userlog.html', initial_module='explosion', initial_source=source)
    return render_template('userlog.html')


@app.route('/Live4', methods=['GET', 'POST'])
@login_required
def Live4():
    if request.method == 'POST':
        filename = request.form.get('filename4', '')
        source = ('testperson/' + filename) if filename else '0'
        return render_template('userlog.html', initial_module='person', initial_source=source)
    return render_template('userlog.html')


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    stop_all_streams()
    session.clear()
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=False, use_reloader=False)
