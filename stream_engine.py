import os
import sys
import time
import threading
from pathlib import Path
import cv2
import numpy as np
import torch
import telepot

# Fix PosixPath on Windows
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Telegram Bot Setup. Credentials are supplied by the deployment environment.
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHANNEL_ID = os.getenv('TELEGRAM_CHAT_ID', '')
try:
    telegram_bot = telepot.Bot(BOT_TOKEN) if BOT_TOKEN and CHANNEL_ID else None
except Exception as e:
    telegram_bot = None
    print(f"[Telegram Bot Init Error]: {e}")

_last_alert_time = {}
_alert_lock = threading.Lock()

def send_telegram_alert(message, cooldown=8.0):
    """Sends Telegram alert in a non-blocking background thread with rate-limiting."""
    if telegram_bot is None:
        return
    now = time.time()
    with _alert_lock:
        if message in _last_alert_time and (now - _last_alert_time[message]) < cooldown:
            return
        _last_alert_time[message] = now

    def _send():
        try:
            telegram_bot.sendMessage(CHANNEL_ID, message)
            print(f"[Telegram Alert Sent]: {message}")
        except Exception as err:
            print(f"[Telegram Alert Warning]: {err}")

    t = threading.Thread(target=_send, daemon=True)
    t.start()


# YOLO Model Cache
_YOLO_MODELS = {}
_YOLO_LOCK = threading.Lock()
_DEVICE = select_device('cpu')

MODEL_CONFIGS = {
    'weapon': {
        'weights': str(ROOT / 'kni_axe.pt'),
        'title': 'YOLO Weapon Detection',
        'classes': None, # detect all classes in model (knife, pistol)
        'conf_thres': 0.25,
        'iou_thres': 0.45,
    },
    'accident': {
        'weights': str(ROOT / 'ACCIDENT.pt'),
        'title': 'YOLO Accident Detection',
        'classes': None,
        'conf_thres': 0.25,
        'iou_thres': 0.45,
    },
    'explosion': {
        'weights': str(ROOT / 'fire.pt'),
        'title': 'YOLO Explosion Detection',
        'classes': None,
        'conf_thres': 0.25,
        'iou_thres': 0.45,
    },
    'person': {
        'weights': str(ROOT / 'yolov5s.pt'),
        'title': 'Multiple Person Detection',
        'classes': [0], # COCO person class
        'conf_thres': 0.25,
        'iou_thres': 0.45,
    }
}

def get_yolo_model(module_key):
    """Lazily loads and caches YOLO models in memory."""
    with _YOLO_LOCK:
        if module_key in _YOLO_MODELS:
            return _YOLO_MODELS[module_key]
        config = MODEL_CONFIGS.get(module_key)
        if not config:
            raise ValueError(f"Unknown module key: {module_key}")
        weights_path = config['weights']
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")
        print(f"[StreamEngine] Loading model weights for {module_key}: {weights_path}")
        model = DetectMultiBackend(weights_path, device=_DEVICE, dnn=False, fp16=False)
        _YOLO_MODELS[module_key] = model
        return model


# Action Recognition Model
_ACTION_MODEL = None
_ACTION_CLASSES = None
_ACTION_LOCK = threading.Lock()

def get_action_model_and_classes():
    """Lazily imports action recognition model and classes."""
    global _ACTION_MODEL, _ACTION_CLASSES
    with _ACTION_LOCK:
        if _ACTION_MODEL is None:
            from A_Recognition import model as act_m
            _ACTION_MODEL = act_m
            classes = []
            classes_path = ROOT / 'output' / 'classes.txt'
            if classes_path.exists():
                with open(classes_path, 'r') as fp:
                    for line in fp:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            classes.append(parts[1])
            _ACTION_CLASSES = classes
        return _ACTION_MODEL, _ACTION_CLASSES


# Active stream controller
_active_stream_token = 0
_stream_token_lock = threading.Lock()

def get_new_stream_token():
    global _active_stream_token
    with _stream_token_lock:
        _active_stream_token += 1
        return _active_stream_token

def stop_all_streams():
    global _active_stream_token
    with _stream_token_lock:
        _active_stream_token += 1
        return _active_stream_token

def is_stream_active(token):
    with _stream_token_lock:
        return token == _active_stream_token


def make_status_frame(title, message, subtext="", frame_size=(640, 480), is_error=False):
    """Creates a stylized information or error frame."""
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    bg_color = (25, 20, 20) if is_error else (20, 30, 20)
    frame[:] = bg_color

    # Border
    border_color = (0, 0, 220) if is_error else (0, 180, 0)
    cv2.rectangle(frame, (10, 10), (frame_size[0] - 10, frame_size[1] - 10), border_color, 2)

    # Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, title, (30, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Message
    msg_color = (120, 120, 255) if is_error else (100, 255, 100)
    cv2.putText(frame, message, (30, 120), font, 0.7, msg_color, 2, cv2.LINE_AA)

    # Subtext
    if subtext:
        cv2.putText(frame, subtext, (30, 160), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes() if ret else b''


def resolve_video_source(source_str):
    """Resolves source string to webcam int index or verified file path."""
    source_clean = (source_str or '').strip()
    if source_clean in ('0', 'webcam', 'live', ''):
        return 0, 'Webcam (Live Camera 0)'

    # Candidate file checks
    candidate_paths = [
        source_clean,
        str(ROOT / source_clean),
        str(ROOT / 'uploads' / source_clean),
        str(ROOT / 'test' / source_clean),
        str(ROOT / 'testweapon' / source_clean),
        str(ROOT / 'testaccident' / source_clean),
        str(ROOT / 'testexplosion' / source_clean),
        str(ROOT / 'testperson' / source_clean),
    ]

    for p in candidate_paths:
        if os.path.isfile(p):
            return p, os.path.basename(p)

    return None, source_clean


def stream_yolo_detection(module_key, source_str, token):
    """Generates JPEG frame stream for YOLO-based modules (weapon, accident, explosion, person)."""
    resolved_src, display_name = resolve_video_source(source_str)

    if resolved_src is None:
        err_frame = make_status_frame("Source Error", f"File Not Found: {display_name}", "Please check the filename or select a valid video.", is_error=True)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + err_frame + b'\r\n')
        return

    config = MODEL_CONFIGS[module_key]
    module_title = config['title']

    try:
        model = get_yolo_model(module_key)
        names = model.names
        classes_filter = config.get('classes')
        conf_thres = config.get('conf_thres', 0.25)
        iou_thres = config.get('iou_thres', 0.45)
    except Exception as e:
        err_frame = make_status_frame(f"{module_title} Error", f"Model Load Failed: {str(e)[:40]}", str(e), is_error=True)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + err_frame + b'\r\n')
        return

    # Open video capture
    is_webcam = (resolved_src == 0)
    if is_webcam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(resolved_src)

    if not cap.isOpened():
        err_msg = "Could not open Webcam 0" if is_webcam else f"Cannot decode video: {display_name}"
        err_frame = make_status_frame("Camera / Video Error", err_msg, "Ensure camera is plugged in or video is supported.", is_error=True)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + err_frame + b'\r\n')
        return

    try:
        frame_idx = 0
        while is_stream_active(token):
            ret, im0 = cap.read()
            if not ret:
                if is_webcam:
                    time.sleep(0.05)
                    continue
                else:
                    # Video completed: loop back to beginning for continuous inspection
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, im0 = cap.read()
                    if not ret:
                        break

            frame_idx += 1
            h, w = im0.shape[:2]

            # Preprocess for YOLO
            im = letterbox(im0, 640, stride=model.stride, auto=model.pt)[0]
            im = im.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(_DEVICE).float() / 255.0
            im = im[None]

            # Inference
            pred = model(im)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes_filter)

            annotator = Annotator(im0, line_width=2, example=str(names))
            person_count = 0
            detected_classes = []

            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in det:
                        c = int(cls)
                        c_name = names[c]
                        conf_val = float(conf)

                        if module_key == 'person':
                            if c_name == 'person':
                                person_count += 1
                                label = f"{person_count}) person {conf_val * 100:.1f}%"
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                detected_classes.append(c_name)
                        else:
                            label = f"{c_name} {conf_val * 100:.1f}%"
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            detected_classes.append(c_name)

            im0 = annotator.result()

            # HUD Overlay
            hud_bg_color = (35, 30, 30)
            cv2.rectangle(im0, (0, 0), (w, 42), hud_bg_color, -1)
            cv2.line(im0, (0, 42), (w, 42), (0, 200, 255), 1)

            cv2.putText(im0, f"Module: {module_title}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            if module_key == 'person':
                count_str = f"Persons Detected: {person_count}"
                cv2.putText(im0, count_str, (w - 260, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 255, 50) if person_count > 0 else (200, 200, 200), 2, cv2.LINE_AA)
                if person_count > 0:
                    send_telegram_alert(f"{person_count} person(s) detected via {module_title}")
            else:
                det_summary = f"Detections: {len(detected_classes)}"
                det_color = (0, 70, 255) if detected_classes else (180, 180, 180)
                cv2.putText(im0, det_summary, (w - 200, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, det_color, 2, cv2.LINE_AA)
                for cl in set(detected_classes):
                    send_telegram_alert(f"{cl} detected via {module_title}")

            # Encode JPEG
            ret, buffer = cv2.imencode('.jpg', im0, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # Maintain smooth playback (~25 fps)
            if not is_webcam:
                time.sleep(0.03)

    finally:
        cap.release()
        print(f"[StreamEngine] Released capture for {module_key} ({display_name})")


def stream_action_recognition(source_str, token):
    """Generates JPEG frame stream for Action Recognition (Analyse button)."""
    resolved_src, display_name = resolve_video_source(source_str)

    if resolved_src is None:
        err_frame = make_status_frame("Action Recognition Error", f"File Not Found: {display_name}", "Please check the filename or select a valid video.", is_error=True)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + err_frame + b'\r\n')
        return

    try:
        model, classes = get_action_model_and_classes()
    except Exception as e:
        err_frame = make_status_frame("Action Recognition Error", f"Model Load Failed: {str(e)[:40]}", str(e), is_error=True)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + err_frame + b'\r\n')
        return

    is_webcam = (resolved_src == 0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if is_webcam else cv2.VideoCapture(resolved_src)

    if not cap.isOpened():
        err_msg = "Could not open Webcam 0" if is_webcam else f"Cannot decode video: {display_name}"
        err_frame = make_status_frame("Action Recognition Error", err_msg, "Ensure video file is valid or camera is connected.", is_error=True)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + err_frame + b'\r\n')
        return

    nb_frames = 25
    img_size = 224

    try:
        # Pre-read 25 frames for model prediction
        predict_frames = []
        all_cached_frames = []
        while len(predict_frames) < nb_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            all_cached_frames.append(frame)
            predict_frames.append(cv2.resize(frame, (img_size, img_size)))

        if len(predict_frames) == 0:
            err_frame = make_status_frame("Action Recognition Error", "No Frames in Video", "Video appears to be empty.", is_error=True)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + err_frame + b'\r\n')
            return

        # Pad with last frame if video has fewer than 25 frames
        while len(predict_frames) < nb_frames:
            predict_frames.append(predict_frames[-1])

        X = np.array(predict_frames) / 255.0
        X = np.reshape(X, (1, nb_frames, img_size, img_size, 3))

        pred = model.predict(X)
        preds = pred.argmax(axis=1)
        action_idx = preds[0]
        action_name = classes[action_idx] if action_idx < len(classes) else f"Action #{action_idx}"
        confidence = float(np.max(pred))

        send_telegram_alert(f"{action_name} detected via Action Recognition")

        # Rewind video to stream from start
        if not is_webcam:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while is_stream_active(token):
            ret, frame = cap.read()
            if not ret:
                if is_webcam:
                    time.sleep(0.05)
                    continue
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break

            h, w = frame.shape[:2]

            # Overlay HUD
            hud_color = (30, 90, 30) if action_name.lower() == 'normal' else (20, 20, 160)
            cv2.rectangle(frame, (0, 0), (w, 46), hud_color, -1)
            cv2.line(frame, (0, 46), (w, 46), (255, 255, 255), 1)

            cv2.putText(frame, "Action Recognition: 3D ResNet-50", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            pred_text = f"Action: {action_name} ({confidence * 100:.1f}%)"
            cv2.putText(frame, pred_text, (w - 360, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            if not is_webcam:
                time.sleep(0.035)

    finally:
        cap.release()
        print(f"[StreamEngine] Released Action Recognition capture ({display_name})")


def generate_frames(module_key, source_str, token):
    """Router for stream generators based on module key."""
    mod = (module_key or '').lower().strip()
    if mod in ('analyse', 'action', 'video'):
        return stream_action_recognition(source_str, token)
    elif mod in ('weapon', 'live1'):
        return stream_yolo_detection('weapon', source_str, token)
    elif mod in ('accident', 'live2'):
        return stream_yolo_detection('accident', source_str, token)
    elif mod in ('explosion', 'live3'):
        return stream_yolo_detection('explosion', source_str, token)
    elif mod in ('person', 'live4'):
        return stream_yolo_detection('person', source_str, token)
    else:
        def _err():
            err_frame = make_status_frame("Unknown Module", f"Module '{module_key}' is not recognized.", "Choose Analyse, Weapon, Accident, Explosion, or Person.", is_error=True)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + err_frame + b'\r\n')
        return _err()
