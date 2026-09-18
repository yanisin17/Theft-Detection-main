import cv2
import asyncio
import base64
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

load_dotenv()

# RTSP 拉流选项：走 TCP + 5 秒 socket 超时（stimeout/timeout 分别兼容新旧版 ffmpeg，
# 不识别的选项会被忽略）。必须在创建第一个 VideoCapture 之前设置。
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|timeout;5000000",
)

from ultralytics import YOLO

# ============================================================
# 项目A行为识别引擎导入（21种行为 + XGBoost + MediaPipe）
# ============================================================
import sys
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from detection_engine.alert_policy import should_run_aggregate_detector, should_trigger_theft_alert
from detection_engine import detector_config

BEHAVIOR_ENGINE_AVAILABLE = False
try:
    from detection_engine.models.behavior.video_behavior import VideoBehaviorDetector
    from detection_engine.models.behavior.image_behavior import ImageBehaviorDetector
    from detection_engine.models.detection import TheftDetector
    BEHAVIOR_ENGINE_AVAILABLE = True
    print("Behavior engine loaded: 21 behavior types + XGBoost + MediaPipe.")
except Exception as e:
    print(f"Behavior engine not loaded: {e}. Falling back to basic YOLO only.")

# 实时行为检测器（MediaPipe 33点 + 伸手→回缩藏匿状态机，实时流可靠告警）
REALTIME_ENGINE_AVAILABLE = False
try:
    from detection_engine.realtime_behavior import RealtimeBehaviorDetector
    REALTIME_ENGINE_AVAILABLE = True
except Exception as e:
    print(f"Realtime behavior engine not loaded: {e}")
import numpy as np
import time
from datetime import datetime
import json
import threading
import uuid
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import requests
from pydantic import BaseModel
import sqlite3
import pickle
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("face_recognition not installed. Face ID disabled.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not installed. System resource monitor will run in simulation mode.")

if not os.path.exists("alerts"):
    os.makedirs("alerts")
EVIDENCE_DIR = os.path.abspath("evidence_videos")
if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

app = FastAPI()

app.mount("/alerts", StaticFiles(directory="alerts"), name="alerts")
app.mount("/evidence_videos", StaticFiles(directory=EVIDENCE_DIR), name="evidence_videos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Setup ---
DB_NAME = "theft_detection.db"

def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS alerts
                     (id TEXT PRIMARY KEY, message TEXT, timestamp TEXT, image_path TEXT,
                      confidence REAL, behavior_type TEXT, track_id INTEGER)''')
        c.execute('''CREATE TABLE IF NOT EXISTS faces
                     (id TEXT PRIMARY KEY, name TEXT, type TEXT, encoding BLOB)''')
        # 轻量迁移：老数据库补列（已存在则忽略）
        for col, decl in [("confidence", "REAL"), ("behavior_type", "TEXT"), ("track_id", "INTEGER"), ("video_path", "TEXT")]:
            try:
                c.execute(f"ALTER TABLE alerts ADD COLUMN {col} {decl}")
                print(f"DB migrated: alerts.{col} added.")
            except sqlite3.OperationalError:
                pass  # 列已存在
        conn.commit()
        conn.close()
        print("Database initialized.")
    except Exception as e:
        print(f"Database error: {e}")

init_db()

# --- Settings & Models ---
SETTINGS_FILE = "settings.json"

class SettingsModel(BaseModel):
    emailEnabled: bool = False
    smtpServer: str = "smtp.gmail.com"
    smtpPort: str = "587"
    senderEmail: str = ""
    senderPassword: str = ""
    receiverEmail: str = ""
    telegramEnabled: bool = False
    telegramBotToken: str = ""
    telegramChatId: str = ""
    roiPoints: list[list[int]] = []
    showHeatmap: bool = False


try:
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            settings_data = json.load(f)
            current_settings = SettingsModel(**settings_data)
            roi_points = current_settings.roiPoints
    else:
        current_settings = SettingsModel()
except Exception as e:
    current_settings = SettingsModel()


# --- Heatmap Logic ---
def update_heatmap(cam_data, center_x, center_y, frame_shape):
    if cam_data.get("heatmap_accumulator") is None or cam_data["heatmap_accumulator"].shape[:2] != frame_shape[:2]:
        cam_data["heatmap_accumulator"] = np.zeros(frame_shape[:2], dtype=np.float32)
    try:
        cam_data["heatmap_accumulator"][center_y, center_x] += 1
    except: pass

def get_heatmap_overlay(cam_data, frame):
    if cam_data.get("heatmap_accumulator") is None: return frame
    msg_max = np.max(cam_data["heatmap_accumulator"])
    if msg_max == 0: return frame
    
    norm_heatmap = cam_data["heatmap_accumulator"] / msg_max
    norm_heatmap = (norm_heatmap * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
    result = cv2.addWeighted(frame, 0.7, color_map, 0.3, 0)
    return result

# --- Face ID Logic ---
known_face_encodings = []
known_face_names = []
known_face_types = [] # 'blacklist' or 'whitelist'
faces_lock = threading.Lock()

def load_known_faces():
    global known_face_encodings, known_face_names, known_face_types
    if not FACE_REC_AVAILABLE: return
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT name, type, encoding FROM faces")
        rows = c.fetchall()
        
        temp_encodings = []
        temp_names = []
        temp_types = []
        for row in rows:
            name, f_type, encoding_blob = row
            encoding = pickle.loads(encoding_blob)
            temp_encodings.append(encoding)
            temp_names.append(name)
            temp_types.append(f_type)
        conn.close()
        
        with faces_lock:
            known_face_encodings = temp_encodings
            known_face_names = temp_names
            known_face_types = temp_types
            
        print(f"Loaded {len(known_face_names)} faces.")
    except Exception as e:
        print(f"Error loading faces: {e}")

load_known_faces()

# --- API Endpoints ---


@app.post("/faces/register")
async def register_face(file: UploadFile = File(...), name: str = Form(...), type: str = Form("blacklist")):
    if not FACE_REC_AVAILABLE: return {"status": "error", "message": "Face Rec not available"}
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        image = face_recognition.load_image_file(temp_filename)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            encoding = encodings[0]
            encoding_blob = pickle.dumps(encoding)
            face_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("INSERT INTO faces VALUES (?,?,?,?)", (face_id, name, type, encoding_blob))
            conn.commit()
            conn.close()
            
            load_known_faces() # Reload
            return {"status": "success", "message": f"Face registered: {name}"}
        else:
            return {"status": "error", "message": "No face found in image"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/faces")
async def get_faces():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, name, type FROM faces")
        rows = c.fetchall()
        conn.close()
        return [{"id": r[0], "name": r[1], "type": r[2]} for r in rows]
    except Exception as e:
        return {"error": str(e)}



# --- API Endpoints ---

@app.get("/settings")
async def get_settings():
    return current_settings

@app.post("/settings")
async def save_settings(settings: SettingsModel):
    global current_settings, roi_points
    current_settings = settings
    roi_points = settings.roiPoints # Update global ROI
    
    from dotenv import set_key
    env_path = ".env"
    if not os.path.exists(env_path):
        open(env_path, 'a').close()
        
    if settings.senderPassword and settings.senderPassword != "********":
        set_key(env_path, "SMTP_PASSWORD", settings.senderPassword)
    if settings.telegramBotToken and settings.telegramBotToken != "********":
        set_key(env_path, "TELEGRAM_BOT_TOKEN", settings.telegramBotToken)
        
    safe_settings = settings.dict()
    # Mask sensitive data in JSON
    safe_settings["senderPassword"] = ""
    safe_settings["telegramBotToken"] = ""
    
    with open(SETTINGS_FILE, "w") as f:
        json.dump(safe_settings, f, indent=4)
        
    load_dotenv(override=True)
    return {"status": "success", "message": "Settings saved"}

@app.post("/roi")
async def save_roi(data: dict):
    global roi_points, current_settings
    if "points" in data:
        roi_points = data["points"]
        current_settings.roiPoints = roi_points
        with open(SETTINGS_FILE, "w") as f:
            json.dump(current_settings.dict(), f, indent=4)
        print(f"ROI Updated: {roi_points}")
        return {"status": "success"}
    return {"status": "error"}

@app.get("/roi")
async def get_roi():
    return {"points": roi_points}


@app.post("/settings/test")
async def test_settings(settings: SettingsModel):
    original_settings = current_settings.copy()
    
    if settings.emailEnabled:
        try:
            msg = MIMEMultipart()
            msg['From'] = settings.senderEmail
            msg['To'] = settings.receiverEmail
            msg['Subject'] = "Theft Detection - Test Email"
            msg.attach(MIMEText("This is a test email from your Theft Detection System.", 'plain'))
            server = smtplib.SMTP(settings.smtpServer, int(settings.smtpPort))
            server.starttls()
            server.login(settings.senderEmail, settings.senderPassword)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            return {"status": "error", "message": f"Email Test Failed: {str(e)}"}

    if settings.telegramEnabled:
        try:
            url = f"https://api.telegram.org/bot{settings.telegramBotToken}/sendMessage"
            data = {"chat_id": settings.telegramChatId, "text": "Theft Detection - Test Message"}
            resp = requests.post(url, data=data)
            if resp.status_code != 200:
                 return {"status": "error", "message": f"Telegram Test Failed: {resp.text}"}
        except Exception as e:
            return {"status": "error", "message": f"Telegram Test Failed: {str(e)}"}
            
    return {"status": "success", "message": "All enabled tests sent successfully!"}



@app.delete("/faces/{face_id}")
async def delete_face(face_id: str):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        conn.commit()
        conn.close()
        load_known_faces() # Reload
        return {"status": "success", "message": "Face deleted successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/history")
async def get_history():
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 100")
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/evidence/stream/{alert_id}")
async def stream_evidence(alert_id: str):
    """MJPEG over HTTP multipart 推流：浏览器 <img src> 直接播放证据视频"""
    frames_dir = f"{EVIDENCE_DIR}/{alert_id}"
    if not os.path.isdir(frames_dir):
        raise HTTPException(status_code=404, detail="证据视频不存在")
    jpg_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if not jpg_files:
        raise HTTPException(status_code=404, detail="证据帧为空")

    # 从 DB 取实际 FPS
    fps = 30.0
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT timestamp FROM alerts WHERE id = ?", (alert_id,))
        row = c.fetchone()
        conn.close()
        # FPS 存在 cam_data 里，但这里用固定间隔推，浏览器按帧间隔算速度
    except Exception:
        pass

    frame_interval = 1.0 / fps  # 每帧间隔秒数

    def generate():
        # 播放一遍然后停（不是无限循环）
        for fname in jpg_files:
            fpath = os.path.join(frames_dir, fname)
            with open(fpath, "rb") as f:
                jpg_data = f.read()
            length_hdr = f"Content-Length: {len(jpg_data)}\r\n\r\n".encode()
            header = b"Content-Type: image/jpeg\r\n" + length_hdr
            yield b"--frame\r\n" + header + jpg_data + b"\r\n"
            time.sleep(frame_interval)
        # 结束标记
        yield b"--frame--\r\n"

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

def _find_evidence_mp4(image_path):
    """根据告警截图文件名 alert_{cam_id}_{ts}.jpg 定位对应的归档 mp4。

    归档 mp4 由 _encode_evidence_video 生成，命名 evidence_{cam_id}_{ts}.mp4，
    其中 cam_id/ts 与同一告警的截图文件名完全一致。
    解析失败或文件不存在时返回 None——宁可不删，也不能误删其他告警的证据。
    """
    try:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        # cam_id 是 UUID（不含下划线），ts 形如 20260915_164734_123456（含下划线）
        _, cam_id, ts = stem.split("_", 2)
        mp4_path = os.path.join(EVIDENCE_DIR, f"evidence_{cam_id}_{ts}.mp4")
        return mp4_path if os.path.exists(mp4_path) else None
    except Exception:
        return None

@app.delete("/history/{alert_id}")
async def delete_alert(alert_id: str):
    """删除指定告警记录（同时清理图片、视频帧目录、归档 mp4）"""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT image_path, video_path FROM alerts WHERE id = ?", (alert_id,))
        row = c.fetchone()
        if row is None:
            conn.close()
            raise HTTPException(status_code=404, detail="告警记录不存在")

        image_path = row["image_path"]
        video_path = row["video_path"]
        c.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
        conn.commit()
        conn.close()

        # 删图片（兼容相对/绝对路径）
        if image_path:
            for p in [image_path, os.path.abspath(image_path)]:
                if os.path.exists(p):
                    try: os.remove(p); break
                    except Exception: pass

        # 删视频帧目录（兼容相对/绝对路径）
        if video_path:
            import shutil
            for p in [video_path, os.path.abspath(video_path), os.path.join(EVIDENCE_DIR, os.path.basename(video_path))]:
                if os.path.isdir(p):
                    try: shutil.rmtree(p); break
                    except Exception: pass

        # 删归档 mp4：只删与该告警同刻生成的 evidence_{cam_id}_{ts}.mp4，
        # 不允许全目录遍历删除（否则会清空所有告警的证据视频）
        mp4_path = _find_evidence_mp4(image_path)
        if mp4_path:
            try: os.remove(mp4_path)
            except Exception: pass

        return {"message": "告警记录已删除"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Video Logic ---

# Global variables
roi_points = []
roi_entry_times = {}
LOITERING_THRESHOLD = 5.0
last_alert_time = 0
ALERT_COOLDOWN = 3.0
latest_frame = None
alert_payload = None # Initialize
lock = threading.Lock()
clients = []

if not os.path.exists("alerts"):
    os.makedirs("alerts")

# --- Threaded Camera Stream ---
def _expand_env_source(src):
    """支持 ${VAR} 环境变量占位符，避免摄像头账号密码明文写进 cameras.json"""
    import re
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), str(src))

class ThreadedCamera:
    """每路摄像头独立采集线程：只保留最新帧 + 断线自动重连 + 运行统计"""

    RECONNECT_BASE_DELAY = 2.0     # 首次重连等待秒数（之后指数退避）
    RECONNECT_MAX_DELAY = 30.0
    MAX_CONSECUTIVE_FAILURES = 15  # 连续读取失败达到该次数触发重连

    def __init__(self, src):
        self.src = _expand_env_source(src)
        self.cap = None
        self.running = True
        self.lock = threading.Lock()

        self.frame = None
        self.frame_ts = 0.0   # 最近一帧的采集时间（墙钟秒）
        self._captured = 0    # 采集线程累计成功帧数
        self._consumed = 0    # read() 最近消费到的帧序号

        # 每路摄像头的实际运行状态（第一阶段采集链路监控）
        self.stats = {
            "width": 0, "height": 0, "fps": 0.0,
            "frames_read": 0, "frames_dropped": 0,
            "read_latency_ms": 0.0, "reconnects": 0,
            "last_frame_ts": 0.0, "status": "connecting",
        }

        self._open()
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        if self.cap is not None:
            self.thread.start()

    def _open(self):
        try:
            src_val = int(self.src)
            is_index = True
        except ValueError:
            src_val = self.src
            is_index = False

        if is_index and os.name == 'nt':
            self.cap = cv2.VideoCapture(src_val, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src_val)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            try:
                # 小缓冲：尽量让推理线程拿到最新帧，避免排队处理旧帧
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            # 记录实际输出参数（RTSP 下 set 分辨率并不保证生效）
            with self.lock:
                self.stats.update({
                    "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0,
                    "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0,
                    "fps": float(self.cap.get(cv2.CAP_PROP_FPS)) or 0.0,
                    "status": "active",
                })
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self._captured += 1
                    self.frame = frame
                    self.frame_ts = time.time()
                    self.stats["frames_read"] += 1
                    self.stats["last_frame_ts"] = self.frame_ts
        else:
            with self.lock:
                self.stats["status"] = "error"

    def update(self):
        failures = 0
        delay = self.RECONNECT_BASE_DELAY
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            t0 = time.time()
            ret, frame = self.cap.read()
            latency_ms = (time.time() - t0) * 1000.0
            if ret:
                failures = 0
                delay = self.RECONNECT_BASE_DELAY
                with self.lock:
                    self._captured += 1
                    self.frame = frame
                    self.frame_ts = time.time()
                    self.stats["frames_read"] += 1
                    self.stats["read_latency_ms"] = round(latency_ms, 2)
                    self.stats["last_frame_ts"] = self.frame_ts
                    self.stats["status"] = "active"
                time.sleep(0.005)
            else:
                failures += 1
                with self.lock:
                    self.stats["status"] = "reconnecting"
                if failures >= self.MAX_CONSECUTIVE_FAILURES:
                    # 断线重连：释放旧句柄，指数退避后重建连接
                    with self.lock:
                        self.stats["reconnects"] += 1
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    time.sleep(delay)
                    if not self.running:
                        break
                    delay = min(delay * 1.5, self.RECONNECT_MAX_DELAY)
                    self._open()
                    failures = 0
                else:
                    time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.frame is not None:
                # 统计两次消费之间被新帧覆盖丢弃的帧数
                dropped = self._captured - self._consumed - 1
                if dropped > 0:
                    self.stats["frames_dropped"] += dropped
                self._consumed = self._captured
                return True, self.frame.copy()
            return False, None

    def timestamp(self):
        with self.lock:
            return self.frame_ts

    def get_stats(self):
        with self.lock:
            return dict(self.stats)

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        self.running = False
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        if self.cap is not None:
            self.cap.release()

# --- Camera Management ---
class CameraManager:
    def __init__(self):
        self.cameras = {}
        # 可重入锁：save_cameras() 内部也会获取 self.lock，
        # 若调用方持锁调用会死锁（threading.Lock 不可重入）
        self.lock = threading.RLock()
        self.load_cameras()

    def load_cameras(self):
        file_path = "cameras.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for cam in data:
                        self.add_camera_internal(cam["id"], cam["source"], cam["name"], cam.get("roi_points", []))
                print(f"Loaded {len(self.cameras)} cameras from cameras.json.")
                return
            except Exception as e:
                print(f"Error loading cameras.json: {e}")

        # Fallback to default webcam if no file exists
        self.add_camera_internal("0", "0", "Kamera 1", [])
        self.save_cameras()

    def save_cameras(self):
        file_path = "cameras.json"
        try:
            data = []
            with self.lock:
                for cam_id, cam_data in self.cameras.items():
                    data.append({
                        "id": cam_id,
                        "name": cam_data["name"],
                        "source": cam_data["source"],
                        "roi_points": cam_data.get("roi_points", [])
                    })
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving cameras.json: {e}")

    def add_camera_internal(self, cam_id, source, name, roi_points):
        threaded_cap = ThreadedCamera(source)
        self.cameras[cam_id] = {
            "cap": threaded_cap,
            "name": name,
            "source": source,
            "status": threaded_cap.get_stats().get("status", "error"),
            "roi_points": roi_points,
            "heatmap_accumulator": None,
            "roi_entry_times": {},
            "last_alert_time": 0
        }

    def add_camera(self, source, name):
        cam_id = str(uuid.uuid4())
        threaded_cap = ThreadedCamera(source)
        if threaded_cap.isOpened():
            with self.lock:
                self.cameras[cam_id] = {
                    "cap": threaded_cap,
                    "name": name,
                    "source": source,
                    "status": "active",
                    "roi_points": [],
                    "heatmap_accumulator": None,
                    "roi_entry_times": {},
                    "last_alert_time": 0
                }
            self.save_cameras()
            print(f"Kamera eklendi: {name} ({source}) ID: {cam_id}")
            return {"id": cam_id, "status": "connected"}
        else:
            print(f"Kamera açılamadı: {source}")
            return {"id": None, "status": "failed"}

    def remove_camera(self, cam_id):
        with self.lock:
            if cam_id in self.cameras:
                self.cameras[cam_id]["cap"].release()
                del self.cameras[cam_id]
                status = True
            else:
                status = False
        if status:
            self.save_cameras()
        return status

    def get_active_cameras(self):
        with self.lock:
            result = []
            for k, v in self.cameras.items():
                stats = v["cap"].get_stats() if v.get("cap") else {}
                result.append({
                    "id": k,
                    "name": v["name"],
                    "source": v["source"],
                    "status": stats.get("status", "error"),
                    "roi_points": v.get("roi_points", []),
                    # 每路摄像头实际分辨率/FPS/读取延迟/丢帧/重连统计
                    "stats": stats,
                })
            return result

camera_manager = CameraManager()

# --- API Endpoints for Cameras ---
class CameraInput(BaseModel):
    name: str
    source: str

@app.post("/cameras")
async def add_new_camera(cam: CameraInput):
    result = camera_manager.add_camera(cam.source, cam.name)
    if result["id"]:
        with camera_manager.lock:
            cam_data = camera_manager.cameras.get(result["id"])
            cam_details = {
                "id": result["id"],
                "name": cam_data["name"] if cam_data else cam.name,
                "source": cam_data["source"] if cam_data else cam.source,
                "status": "active"
            } if cam_data else None
        return {"message": "Camera added", "camera": cam_details}
    else:
        raise HTTPException(status_code=400, detail="Failed to open camera")

@app.get("/stats")
def get_stats():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT substr(timestamp, 1, 8), count(*) FROM alerts GROUP BY substr(timestamp, 1, 8)")
    data = dict(c.fetchall())
    conn.close()
    
    stats = []
    from datetime import timedelta
    today = datetime.now()
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        key = d.strftime("%Y%m%d")
        stats.append(data.get(key, 0))

    cpu_load = 0
    ram_load = 0
    if PSUTIL_AVAILABLE:
        try:
            cpu_load = psutil.cpu_percent()
            ram_load = psutil.virtual_memory().percent
        except:
            import random
            cpu_load = random.randint(15, 30)
            ram_load = random.randint(40, 50)
    else:
        import random
        cpu_load = random.randint(15, 30)
        ram_load = random.randint(40, 50)
        
    return {
        "weekly_data": stats,
        "cpu_load": cpu_load,
        "ram_load": ram_load
    }

@app.get("/cameras")
async def list_cameras():
    return camera_manager.get_active_cameras()

@app.delete("/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    if camera_manager.remove_camera(camera_id):
        return {"message": "Camera removed"}
    raise HTTPException(status_code=404, detail="Camera not found")

@app.post("/cameras/{camera_id}/roi")
async def save_camera_roi(camera_id: str, data: dict):
    if "points" in data:
        points = data["points"]
        found = False
        with camera_manager.lock:
            if camera_id in camera_manager.cameras:
                camera_manager.cameras[camera_id]["roi_points"] = points
                found = True
        if found:
            # 锁外落盘：save_cameras() 内部会再次获取同一把锁，
            # 避免在持锁期间做文件 IO
            camera_manager.save_cameras()
            return {"status": "success", "roi_points": points}
        raise HTTPException(status_code=404, detail="Camera not found")
    raise HTTPException(status_code=400, detail="Invalid data")

@app.get("/cameras/{camera_id}/roi")
async def get_camera_roi(camera_id: str):
    with camera_manager.lock:
        if camera_id in camera_manager.cameras:
            return {"points": camera_manager.cameras[camera_id].get("roi_points", [])}
    raise HTTPException(status_code=404, detail="Camera not found")



# --- 项目A行为引擎实时检测参数 ---
# 报警阈值统一收口到 detector_config（可被 detector_tuning.json 覆盖），
# 避免此处与 detector_config.THEFT_ALERT_THRESHOLD 两处不一致
BEHAVIOR_ALERT_THRESHOLD = detector_config.THEFT_ALERT_THRESHOLD
BEHAVIOR_CHECK_INTERVAL = 6      # 聚合行为引擎(21种规则)的采样间隔，兜底通路保持低频即可
REALTIME_CHECK_INTERVAL = 1      # 实时状态机(伸手→回缩)采样间隔：必须每帧分析。
                                 # 旧值同 BEHAVIOR_CHECK_INTERVAL=6，多摄像头串行推理下
                                 # 同一人的实际分析间隔高达 2~4 秒，而"伸手→回缩"
                                 # 动作 1~2 秒内完成，序列根本凑不齐（漏报主因之一）
PERSON_GRACE_SECONDS = 10.0      # track_id 消失超过该秒数后清理状态（再次出现可重新报警）

# 行为分档阈值：不是所有行为都同等重要
BEHAVIOR_TIERS = {
    # 高风险：直接关联盗窃动作，0.7 就报警
    "HIGH": {"rapid_item_concealment", "item_concealment", "item_grabbing",
             "suspicious_item_handling", "single_arm_hiding", "body_shielding",
             "concealment_gesture", "distraction_behavior", "suspected_tag_removal",
             "group_theft_coordination", "ML Detected Theft"},
    # 中风险：可能是盗窃也可能是正常动作，0.85 才报警
    "MEDIUM": {"suspicious_crouching", "unusual_elbow_position", "abnormal_arm_position",
               "unusual_reaching", "repetitive_position_adjustment"},
    # 低风险：正常购物行为，只在帧上标注但不报警
    "LOW": {"looking_around", "abnormal_head_movement", "covering_product_area",
            "reaching_motion", "Normal Shopping", "Checking Product"},
}

# 查询某行为类型对应的报警阈值
def _get_behavior_alert_threshold(btype: str) -> float:
    if btype in BEHAVIOR_TIERS["HIGH"]:
        return BEHAVIOR_ALERT_THRESHOLD        # 0.7
    elif btype in BEHAVIOR_TIERS["MEDIUM"]:
        return 0.85
    else:
        return 1.0  # 低风险永远不触发告警
# 正常购物行为不触发报警
NORMAL_BEHAVIORS = {"Normal Shopping", "Checking Product", "normal_shopping", "checking_product"}

# --- 行为权重表（来自项目 A 的 initial_behavior_weights）---
BEHAVIOR_WEIGHTS = {
    "covering_product_area": 0.5,
    "unusual_elbow_position": 0.5,
    "repetitive_position_adjustment": 0.5,
    "suspected_tag_removal": 0.4,
    "suspicious_item_handling": 0.7,
    "rapid_item_concealment": 0.8,
    "abnormal_arm_position": 0.6,
    "suspicious_crouching": 0.6,
    "unusual_reaching": 0.7,
    "body_shielding": 0.7,
    "abnormal_head_movement": 0.5,
    "single_arm_hiding": 0.7,
    "concealment_gesture": 0.7,
    "distraction_behavior": 0.8,
    "group_theft_coordination": 0.8,
    "item_grabbing": 0.8,
    "item_concealment": 0.8,
    "looking_around": 0.4,
    "reaching_motion": 0.5,
}

# --- State Tracker for Concealment ---
class PersonState:
    def __init__(self, track_id):
        self.track_id = track_id
        self.state = "NEUTRAL" # NEUTRAL, REACHING, HOLDING, SUSPICIOUS
        self.last_reach_time = 0
        self.holding_object = False
        self.holding_hand = None
        self.last_holding_time = 0
        self.face_checked = False
        self.face_check_time = 0
        # --- 按人去重 + MAX CONFIDENCE 追踪 ---
        self.last_seen = time.time()
        self.alert_fired = False        # 该人是否已报过警（跨帧只报一次）
        self.alert_id = None            # 对应 alerts 表行 id，用于后续 UPDATE 峰值置信度
        self.max_confidence = 0.0       # 该人本次出现期间的最高偷盗概率
        self.top_behavior = None        # 最高偷盗概率对应的主行为类型
        # --- 完整累加系统状态 ---
        self.behavior_frame_counts = {} # 行为类型 → 连续出现帧数（用于连续性加分）
        self.last_behaviors = set()     # 上一帧检测到的行为集合（用于连续性检测）
        self.rule_prob = 0.0            # 当前帧规则引擎加权概率
        self.continuity_bonus = 0.0     # 连续性加分
        self.sequence_bonus = 0.0       # 序列检测加分
        self.theft_probability = 0.0    # 最终偷盗概率
        self.sequence_detected = False  # 是否检测到抓取→隐藏完整序列
        self.grab_phase = None          # 'left'/'right'，用于序列检测
        # --- 佐证确认：单次藏匿事件先挂起，窗口内二次事件或手中持物才报警 ---
        self.conceal_event_times = []   # 报警级藏匿事件的时间戳列表（自动裁剪窗口外旧事件）
        # 其它报警类型也按人去重
        self.blacklist_alert_fired = False
        self.roi_alert_fired = False
        self.loiter_alert_fired = False

person_states = {} # {(cam_id, track_id): PersonState}

# --- 证据视频：前5秒+后5秒 ---
from collections import deque as _deque  # 延迟导入避免首屏慢
EVIDENCE_FPS = 30                # 默认值，运行时会被摄像头实际FPS覆盖
PRE_FRAMES = 150                 # 前 5 秒缓冲（30fps × 5s）
POST_FRAMES = 150                # 后 5 秒
# EVIDENCE_DIR 已在文件顶部定义（绝对路径）

# 每个摄像头一个环形缓冲区，存 JPEG 压缩帧 bytes（约 50KB/帧 × 150 ≈ 7.5MB/摄像头）
frame_buffers: dict[str, _deque] = {}

# 正在"冻结前5秒 + 采集后5秒"的告警队列
# key=cam_id，value=dict(alert_id, pre_frames:list, post_remaining:int, ts, cam_name)
evidence_pending: dict[str, dict] = {}

def prune_stale_person_states(current_time):
    """清理消失超过 PERSON_GRACE_SECONDS 的人物状态，使其再次出现时可以重新报警"""
    stale = [k for k, st in person_states.items() if current_time - st.last_seen > PERSON_GRACE_SECONDS]
    for k in stale:
        del person_states[k]
    if stale:
        print(f"Pruned {len(stale)} stale person tracks.")

# --- Helper Functions for Pose ---
def check_reaching(keypoints, roi_poly):
    if len(keypoints) < 11: return False, None
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    reaching_hand = None
    
    if left_wrist[0] > 0 and left_wrist[1] > 0 and len(roi_poly) >= 3:
        if cv2.pointPolygonTest(np.array(roi_poly), (int(left_wrist[0]), int(left_wrist[1])), False) >= 0:
            reaching_hand = "LEFT"

    if right_wrist[0] > 0 and right_wrist[1] > 0 and len(roi_poly) >= 3:
        if cv2.pointPolygonTest(np.array(roi_poly), (int(right_wrist[0]), int(right_wrist[1])), False) >= 0:
            reaching_hand = "RIGHT"
            
    return (reaching_hand is not None), reaching_hand

def check_object_in_hand(keypoints, object_boxes, hand="LEFT"):
    # Check if any object box is close to the specified wrist
    if len(keypoints) < 11: return False
    wrist = keypoints[9] if hand == "LEFT" else keypoints[10]
    
    if wrist[0] == 0: return False
    
    for obj in object_boxes:
        box = obj['bbox'] if isinstance(obj, dict) else obj
        # Box: x1, y1, x2, y2
        # Check distance from wrist to box center
        box_cx = (box[0] + box[2]) / 2
        box_cy = (box[1] + box[3]) / 2

        dist = np.sqrt((wrist[0] - box_cx)**2 + (wrist[1] - box_cy)**2)

        # If wrist is CLOSE to object center (e.g. < 100px) OR wrist is INSIDE box
        if dist < 120: # Threshold
            return True
        if box[0] < wrist[0] < box[2] and box[1] < wrist[1] < box[3]:
            return True

    return False

def check_concealment(keypoints, reaching_hand):
    if len(keypoints) < 13: return False
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    target_wrist = keypoints[9] if reaching_hand == "LEFT" else keypoints[10]
    
    if target_wrist[0] == 0 or left_hip[0] == 0 or right_hip[0] == 0: return False
    
    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    hip_center_y = (left_hip[1] + right_hip[1]) / 2
    
    dist_x = target_wrist[0] - hip_center_x
    dist_y = target_wrist[1] - hip_center_y
    distance = np.sqrt(dist_x**2 + dist_y**2)
    
    hip_width = np.abs(left_hip[0] - right_hip[0])
    threshold = max(hip_width * 1.5, 100) 
    
    return distance < threshold

def check_bending(keypoints):
    if len(keypoints) < 12: return False
    l_shoulder = keypoints[5]
    l_hip = keypoints[11]
    if l_shoulder[1] == 0 or l_hip[1] == 0: return False
    vertical_dist = l_hip[1] - l_shoulder[1]
    return vertical_dist < 50

# --- Updated Video Loop ---
def video_loop():
    global latest_frame, current_settings, alert_payload, known_face_encodings, known_face_names, known_face_types, person_states
    
    print("Video Loop Başlatılıyor...") 
    model_obj = None # Fallback or specialized
    model_is_specialized = False
    
    try:
        print("Loading Pose Model...")
        model_pose = YOLO('yolov8n-pose.pt')

        # P0 修复：多路摄像头必须各用独立的 YOLO 实例做 track。
        # persist=True 的 tracker 是有状态的（卡尔曼滤波 + 轨迹历史），
        # 多路摄像头的帧交替喂给同一实例时，预测完全失准导致
        # track ID 频繁跳变；而行为状态机按 cam_id:track_id 键控，
        # ID 一跳变，"伸手"与"回缩"就被拆到两个 key 上，序列永远凑不齐。
        pose_model_path = 'yolov8n-pose.pt'
        pose_trackers = {}  # cam_id -> 独立 YOLO 实例（各自独立的 tracker 状态）

        def _get_pose_tracker(cid):
            m = pose_trackers.get(cid)
            if m is None:
                m = YOLO(pose_model_path)
                pose_trackers[cid] = m
            return m

        
        print("Loading Theft Detection Model...")
        try:
            # Try to load specialized model first
            model_obj = YOLO('shoplifting.pt')
            model_is_specialized = True
            print("Özel Hırsızlık Modeli Yüklendi! (shoplifting.pt)")
        except:
            print("Özel model bulunamadı, standart nesne takibine (models/yolov11n.pt) geçiliyor...")
            try:
                model_obj = YOLO('models/yolov11n.pt')
            except Exception as e:
                print(f"Standart Model de yüklenemedi: {e}")
                model_obj = None

        print("Modeller hazır.")

        # 项目A行为识别引擎初始化（21种行为 + XGBoost + MediaPipe，用于离线文件分析）
        behavior_detector = None
        if BEHAVIOR_ENGINE_AVAILABLE:
            try:
                behavior_detector = VideoBehaviorDetector()
                print("VideoBehaviorDetector loaded - used for offline file analysis.")
            except Exception as e:
                print(f"VideoBehaviorDetector init failed: {e}")
                behavior_detector = None

        # 实时行为检测器（MediaPipe 33点，伸手拿取→回缩藏匿状态机）
        realtime_detector = None
        if REALTIME_ENGINE_AVAILABLE:
            try:
                realtime_detector = RealtimeBehaviorDetector()
                print("RealtimeBehaviorDetector loaded - live Taking/Concealment alerts enabled.")
            except Exception as e:
                print(f"RealtimeBehaviorDetector init failed: {e}")
                realtime_detector = None
    except Exception as e:
        print(f"CRITICAL MODEL ERROR: {e}")
        with open("error_log.txt", "a") as f:
             f.write(f"{datetime.now()}: CRITICAL LOAD ERROR: {e}\n")
        return

    frame_count = 0
    no_signal_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(no_signal_frame, "SINYAL YOK", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    while True:
        try:
            with camera_manager.lock:
                current_cams = list(camera_manager.cameras.items())

            frames_payload = [] 
            
            # Optimization: Run Object Det every 5 frames
            run_obj_det = (frame_count % 5 == 0) and (model_obj is not None)
            
            for cam_id, cam_data in current_cams:
                cap = cam_data["cap"]
                name = cam_data["name"]
                current_time = time.time()
                
                # Fetch specific camera ROI
                cam_roi = cam_data.get("roi_points", [])
                
                ret = False
                if cap.isOpened():
                    ret, frame = cap.read()
                if not ret:
                    frame = no_signal_frame.copy()

                # 本帧的采集时间戳（ThreadedCamera 记录），随画面与 WS 消息一起下发
                frame_ts = cap.timestamp() or time.time()

                if ret:
                    # 推理与人物裁剪始终用未标注的原始帧；标注只画在展示帧上
                    clean = frame.copy()

                    # 1. POSE INFERENCE (Every Frame for tracking)
                    # 每路摄像头使用独立 tracker 实例，消除多路串流导致的 ID 跳变
                    results_pose = _get_pose_tracker(cam_id).track(clean, persist=True, verbose=False, classes=[0])

                    # 先渲染 YOLO 骨架/检测框，再叠加所有自定义标注，
                    # 修复 results_pose[0].plot() 覆盖行为文字/热力图/告警框的问题
                    if results_pose[0].keypoints is not None:
                        frame = results_pose[0].plot()
                    frame = get_heatmap_overlay(cam_data, frame)

                    # 帧时间戳（右上角）：便于确认画面不是旧帧
                    ts_text = datetime.fromtimestamp(frame_ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    ts_x, ts_y = frame.shape[1] - 252, 22
                    cv2.putText(frame, ts_text, (ts_x + 1, ts_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(frame, ts_text, (ts_x, ts_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # 2. THEFT / OBJECT INFERENCE
                    detected_objects = []
                    suspicious_activity_detected = False
                    
                    if run_obj_det:
                        if model_is_specialized:
                            results_obj = model_obj(clean, verbose=False, conf=0.4)
                            if len(results_obj) > 0:
                                boxes = results_obj[0].boxes.xyxy.cpu().numpy().astype(int)
                                clss = results_obj[0].boxes.cls.cpu().numpy().astype(int)
                                confs = results_obj[0].boxes.conf.cpu().numpy()
                                
                                for b, c, conf in zip(boxes, clss, confs):
                                    class_name = model_obj.names[c].lower()
                                    if "shoplift" in class_name or "suspicious" in class_name or "theft" in class_name or "fight" in class_name:
                                        label = f"{class_name.upper()} {conf:.2f}"
                                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
                                        cv2.putText(frame, label, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                        suspicious_activity_detected = True
                                        
                                        if current_time - cam_data["last_alert_time"] > ALERT_COOLDOWN:
                                            trigger_alert(cam_id, name, f"CRIMINAL ACTIVITY: {class_name}", frame)
                                            cam_data["last_alert_time"] = current_time
                                    else:
                                         cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
                        else:
                            # Fallback Logic - Target classes for stealable items
                            TARGET_CLASSES = [24, 25, 26, 28, 39, 40, 41, 42, 43, 67, 73, 74, 75, 76, 77, 78, 79]
                            results_obj = model_obj(clean, verbose=False, conf=0.3)
                            if len(results_obj) > 0:
                                 boxes_obj = results_obj[0].boxes.xyxy.cpu().numpy().astype(int)
                                 cls_obj = results_obj[0].boxes.cls.cpu().numpy().astype(int)
                                 conf_obj = results_obj[0].boxes.conf.cpu().numpy()

                                 for b, c, conf in zip(boxes_obj, cls_obj, conf_obj):
                                     if c in TARGET_CLASSES:
                                         # 保留类别名和置信度，供裁剪图坐标系的物体关联使用
                                         detected_objects.append({
                                             'bbox': b,
                                             'class': model_obj.names[c],
                                             'conf': float(conf),
                                         })
                                         label = f"ITEM: {model_obj.names[c]} {conf:.2f}"
                                         cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 165, 255), 2)
                                         cv2.putText(frame, label, (b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    
                    if run_obj_det:
                        cam_data["last_objects"] = detected_objects
                        cam_data["last_objects_ts"] = current_time
                    else:
                        # 复用上一轮物体结果，但超过 1 秒的旧框直接丢弃，
                        # 避免过期物体框参与"手中持物/藏匿"判断
                        if current_time - cam_data.get("last_objects_ts", 0.0) > 1.0:
                            detected_objects = []
                        else:
                            detected_objects = cam_data.get("last_objects", [])

                    if results_pose[0].boxes.id is not None:
                        boxes = results_pose[0].boxes.xyxy.cpu().numpy().astype(int)
                        track_ids = results_pose[0].boxes.id.cpu().numpy().astype(int)
                        
                        try:
                            keypoints_all = results_pose[0].keypoints.xy.cpu().numpy()
                        except:
                            keypoints_all = []

                        for i, track_id in enumerate(track_ids):
                            box = boxes[i]
                            kpts = keypoints_all[i] if len(keypoints_all) > i else []
                            
                            # Multi-camera safe tracking key
                            state_key = (cam_id, track_id)
                            if state_key not in person_states:
                                person_states[state_key] = PersonState(track_id)
                            p_state = person_states[state_key]
                            p_state.last_seen = current_time
                            
                            is_bending = False
                            is_reaching = False
                            
                            # --- FACE REC ---
                            if FACE_REC_AVAILABLE and (not p_state.face_checked or (current_time - p_state.face_check_time > 2.0)):
                                p_state.face_check_time = current_time
                                fx1, fy1, fx2, fy2 = max(0, box[0]), max(0, box[1]), min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                                face_img = clean[fy1:fy2, fx1:fx2]
                                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                face_locs = face_recognition.face_locations(rgb_face)
                                if face_locs:
                                    encodings = face_recognition.face_encodings(rgb_face, face_locs)
                                    if encodings:
                                        with faces_lock:
                                            matches = face_recognition.compare_faces(known_face_encodings, encodings[0], tolerance=0.5)
                                        if True in matches:
                                            match_index = matches.index(True)
                                            match_name = known_face_names[match_index]
                                            match_type = known_face_types[match_index]
                                            if match_type == "blacklist":
                                                cv2.putText(frame, f"BLACKLIST: {match_name}", (box[0], box[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                                                # 同一人只报一次黑名单
                                                if not p_state.blacklist_alert_fired:
                                                    trigger_alert(cam_id, name, f"BLACKLIST FACE: {match_name}", frame)
                                                    p_state.blacklist_alert_fired = True
                                            else:
                                                cv2.putText(frame, f"VIP: {match_name}", (box[0], box[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                                p_state.face_checked = True

                            # --- POSE & THEFT LOGIC ---
                            is_bending = check_bending(kpts)
                            
                            if not model_is_specialized:
                                left_has_obj = check_object_in_hand(kpts, detected_objects, "LEFT")
                                right_has_obj = check_object_in_hand(kpts, detected_objects, "RIGHT")
                                current_holding = left_has_obj or right_has_obj
                                holding_hand = "LEFT" if left_has_obj else "RIGHT" if right_has_obj else None
    
                                if current_holding:
                                    p_state.holding_object = True
                                    p_state.last_holding_time = current_time
                                    p_state.holding_hand = holding_hand
                                    cv2.putText(frame, f"HOLDING ({holding_hand})", (box[0], box[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                                
                                if p_state.holding_object and not current_holding:
                                    time_since_hold = current_time - p_state.last_holding_time
                                    if time_since_hold < 3.0:
                                         hand_to_check = p_state.holding_hand
                                         if hand_to_check and check_concealment(kpts, hand_to_check):
                                              # 仅画面提示；盗窃是否成立由项目A的21种行为引擎（Rapid Item Concealment /
                                              # Single Arm Hiding / Hiding Item 等）按 track_id 统一报警
                                              cv2.putText(frame, "CONCEALMENT GESTURE", (box[0], box[1]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                                    else:
                                        if time_since_hold > 3.0:
                                            p_state.holding_object = False
                                            p_state.holding_hand = None

                            # --- 完整累加系统：规则引擎加权 × 连续性加分 × 序列检测 → 偷盗概率 ---
                            run_behavior = (behavior_detector is not None) and (
                                frame_count % BEHAVIOR_CHECK_INTERVAL == 0 or is_bending
                            )
                            realtime_confirmed = False

                            if realtime_detector is not None and frame_count % REALTIME_CHECK_INTERVAL == 0:
                                try:
                                    pad = 60
                                    sx1 = max(0, box[0] - pad)
                                    sy1 = max(0, box[1] - pad)
                                    sx2 = min(frame.shape[1], box[2] + pad)
                                    sy2 = min(frame.shape[0], box[3] + pad)
                                    person_frame = clean[sy1:sy2, sx1:sx2].copy()
                                    realtime_event = realtime_detector.analyze(
                                        person_frame, f'{cam_id}:{int(track_id)}'
                                    )

                                    if realtime_event is not None:
                                        event_type = realtime_event.get('type', '')
                                        # 藏匿报警事件返回 alert_score(0.45~0.85 报警分)，
                                        # 伸手提示仍返回 confidence；两者都兼容
                                        event_confidence = float(
                                            realtime_event.get('alert_score',
                                                               realtime_event.get('confidence', 0.0))
                                        )
                                        event_alertable = should_trigger_theft_alert(
                                            event_type, event_confidence, sequence_confirmed=True
                                        )
                                        event_color = (0, 0, 255) if event_alertable else (0, 165, 255)
                                        cv2.putText(
                                            frame, f'{event_type}: {event_confidence:.0%}',
                                            (box[0], box[3] + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            event_color, 2
                                        )

                                        if event_alertable:
                                            realtime_confirmed = True
                                            p_state.theft_probability = max(
                                                p_state.theft_probability, event_confidence
                                            )
                                            p_state.top_behavior = 'rapid_item_concealment'
                                            if event_confidence > p_state.max_confidence:
                                                p_state.max_confidence = event_confidence

                                            # --- 佐证确认：单次藏匿事件先挂起 ---
                                            # 窗口内第二次藏匿事件 或 藏匿时手中持有物品 才真正报警；
                                            # REQUIRE_CORROBORATION=False 时保持旧行为(单次即报)
                                            p_state.conceal_event_times.append(current_time)
                                            p_state.conceal_event_times = [
                                                t for t in p_state.conceal_event_times
                                                if current_time - t <= detector_config.CORROBORATION_WINDOW
                                            ]
                                            corroborated = (
                                                not detector_config.REQUIRE_CORROBORATION
                                                or len(p_state.conceal_event_times) >= 2
                                                or p_state.holding_object
                                            )

                                            if corroborated and not p_state.alert_fired:
                                                corr_source = 'holding' if p_state.holding_object else 'repeat'
                                                alert_id = trigger_alert(
                                                    cam_id, name,
                                                    f'THEFT: Rapid Item Concealment | probability={event_confidence:.2f} | sequence=Y | corroborated={corr_source}',
                                                    frame, confidence=event_confidence,
                                                    behavior_type='rapid_item_concealment',
                                                    track_id=int(track_id)
                                                )
                                                p_state.alert_fired = True
                                                p_state.alert_id = alert_id
                                                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                                            elif corroborated and p_state.alert_id:
                                                update_alert_peak(
                                                    p_state.alert_id, p_state.top_behavior, p_state.max_confidence
                                                )
                                            elif not p_state.alert_fired:
                                                # 已达报警级但缺佐证：橙色挂起提示，等窗口内二次事件
                                                cv2.putText(frame, 'PENDING CONFIRM (concealment)',
                                                            (box[0], box[3] + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                            (0, 165, 255), 2)
                                except Exception as e:
                                    print(f'RealtimeBehaviorDetector error (track {track_id}): {e}')
                            if run_behavior and should_run_aggregate_detector(realtime_confirmed):
                                try:
                                    pad = 60
                                    sx1 = max(0, box[0] - pad)
                                    sy1 = max(0, box[1] - pad)
                                    sx2 = min(frame.shape[1], box[2] + pad)
                                    sy2 = min(frame.shape[0], box[3] + pad)
                                    sub_frame = clean[sy1:sy2, sx1:sx2].copy()

                                    # 关键修复：检测框统一平移到裁剪图坐标系（减去裁剪偏移量），
                                    # 否则裁剪图内的手腕/人体位置无法与整帧坐标的物体框正确关联，
                                    # 导致伸手/拿取/藏匿判断错位。只传物体（人本身由
                                    # MediaPipe 关键点表示），并使用列表格式（引擎对
                                    # ultralytics Results 的属性访问从未生效过）。
                                    shifted_detections = []
                                    for obj in detected_objects:
                                        b = obj['bbox'] if isinstance(obj, dict) else obj
                                        shifted_detections.append({
                                            'class': obj.get('class', 'object') if isinstance(obj, dict) else 'object',
                                            'confidence': obj.get('conf', 0.5) if isinstance(obj, dict) else 0.5,
                                            'bbox': [int(b[0] - sx1), int(b[1] - sy1),
                                                     int(b[2] - sx1), int(b[3] - sy1)],
                                        })

                                    # 调用项目 A 的行为引擎（21种行为 + 规则引擎）
                                    all_behaviors = behavior_detector.detect_behaviors_in_image(sub_frame, shifted_detections)

                                    # 过滤 + 加权求和
                                    current_behaviors = {}  # btype → (conf, weight)
                                    for b in all_behaviors:
                                        conf = float(b.get("confidence", 0))
                                        btype = str(b.get("type", "")).lower()
                                        btype_original = str(b.get("type", ""))
                                        if conf > 0.3 and "Normal" not in btype_original and "Checking" not in btype_original:
                                            weight = BEHAVIOR_WEIGHTS.get(btype, 0.5)
                                            current_behaviors[btype] = (conf, weight)

                                    # === 第 1 层：规则引擎加权概率 ===
                                    # rule_prob = Σ (每种行为置信度 × 权重)，封顶 1.0
                                    rule_prob = 0.0
                                    for btype, (conf, weight) in current_behaviors.items():
                                        rule_prob += conf * weight
                                    p_state.rule_prob = min(1.0, rule_prob)

                                    # === 第 2 层：连续性加分 ===
                                    # 同一种行为连续出现 ≥3 帧 → 每多一帧 +0.05，封顶 0.15
                                    continuity_bonus = 0.0
                                    for btype in current_behaviors:
                                        if btype in p_state.last_behaviors:
                                            p_state.behavior_frame_counts[btype] = p_state.behavior_frame_counts.get(btype, 1) + 1
                                            count = p_state.behavior_frame_counts[btype]
                                            if count >= 3:
                                                bonus = min(0.15, 0.05 * (count - 2))
                                                continuity_bonus = max(continuity_bonus, bonus)
                                        else:
                                            p_state.behavior_frame_counts[btype] = 1
                                    # 消失的行为重置计数
                                    for btype in list(p_state.behavior_frame_counts.keys()):
                                        if btype not in current_behaviors:
                                            del p_state.behavior_frame_counts[btype]
                                    p_state.last_behaviors = set(current_behaviors.keys())
                                    p_state.continuity_bonus = continuity_bonus

                                    # === 第 3 层：序列检测（抓取 → 隐藏） ===
                                    # 检测左手/右手 reach → grab → concealment 的完整时序
                                    sequence_bonus = 0.0
                                    seq_detected = False
                                    bset = set(current_behaviors.keys())
                                    # 右手序列
                                    if "item_grabbing" in bset or "rapid_item_concealment" in bset or "item_concealment" in bset:
                                        if "single_arm_hiding" in bset or "concealment_gesture" in bset or "body_shielding" in bset:
                                            seq_detected = True
                                    # 辅助判断：伸手 + 隐藏同时出现
                                    if "reaching_motion" in bset and ("item_concealment" in bset or "rapid_item_concealment" in bset):
                                        seq_detected = True
                                    # 身体遮挡 + 可疑商品处理
                                    if "body_shielding" in bset and ("suspicious_item_handling" in bset or "item_grabbing" in bset):
                                        seq_detected = True

                                    if seq_detected:
                                        sequence_bonus = 0.3  # 序列检测直接 +0.3
                                        p_state.sequence_detected = True
                                    p_state.sequence_bonus = sequence_bonus

                                    # === 融合：最终偷盗概率 ===
                                    # ML 模型不可用 → rule_prob × 0.7 + 连续性 + 序列
                                    # ML 可用时改为 ml_prob × 0.6 + rule_prob × 0.4
                                    theft_prob = (
                                        p_state.rule_prob * 0.7          # 规则引擎贡献
                                        + p_state.continuity_bonus       # 连续性加分
                                        + p_state.sequence_bonus         # 序列加分
                                    )
                                    p_state.theft_probability = min(1.0, theft_prob)

                                    # === 确定主行为（展示用） ===
                                    top_btype = max(current_behaviors,
                                                    key=lambda t: current_behaviors[t][0] * current_behaviors[t][1],
                                                    default=None)
                                    if top_btype:
                                        p_state.top_behavior = top_btype
                                    else:
                                        top_btype = p_state.top_behavior

                                    # 中文标签
                                    cn_label = top_btype
                                    if hasattr(behavior_detector, 'behavior_type_map') and top_btype:
                                        cn_label = behavior_detector.behavior_type_map.get(top_btype, top_btype)

                                    # 更新峰值
                                    if p_state.theft_probability > p_state.max_confidence:
                                        p_state.max_confidence = p_state.theft_probability

                                    # === 帧上标注 ===
                                    # 聚合引擎的序列检测命中(抓取→藏匿组合)时，行为语义等同于
                                    # rapid_item_concealment，作为实时状态机之外的兜底报警通路
                                    agg_behavior = ('rapid_item_concealment'
                                                    if p_state.sequence_detected else p_state.top_behavior)
                                    # 主标签：偷盗概率
                                    label_color = (0, 0, 255) if should_trigger_theft_alert(
                                        agg_behavior, p_state.theft_probability,
                                        sequence_confirmed=p_state.sequence_detected
                                    ) else (0, 165, 255)
                                    label = f"偷盗概率: {p_state.theft_probability:.0%}"
                                    cv2.putText(frame, label, (box[0], box[3] + 22),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)

                                    # 参与计算的行为列表（小字）
                                    if current_behaviors:
                                        behavior_labels = " + ".join(
                                            f"{behavior_detector.behavior_type_map.get(b, b) if hasattr(behavior_detector,'behavior_type_map') else b}:{c:.0%}"
                                            for b, (c, w) in sorted(current_behaviors.items(),
                                                                     key=lambda x: -x[1][0]*x[1][1])[:4]
                                        )
                                        cv2.putText(frame, behavior_labels, (box[0], box[3] + 44),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

                                    # === 最终判定：序列确认 + 偷盗概率达标 → 报警 ===
                                    # (序列检测本身已要求"抓取→藏匿"两类行为同时出现，自带佐证语义)
                                    if (not p_state.alert_fired) and should_trigger_theft_alert(
                                        agg_behavior, p_state.theft_probability,
                                        sequence_confirmed=p_state.sequence_detected
                                    ):
                                        alert_id = trigger_alert(
                                            cam_id, name,
                                            f"THEFT: {cn_label} | 概率={p_state.theft_probability:.2f} | "
                                            f"rule={p_state.rule_prob:.2f} continuity={p_state.continuity_bonus:.2f} sequence={'Y' if p_state.sequence_detected else 'N'}",
                                            frame,
                                            confidence=p_state.theft_probability,
                                            behavior_type=p_state.top_behavior,
                                            track_id=int(track_id)
                                        )
                                        p_state.alert_fired = True
                                        p_state.alert_id = alert_id
                                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                                    elif p_state.alert_fired and p_state.alert_id:
                                        update_alert_peak(p_state.alert_id, p_state.top_behavior, p_state.max_confidence)
                                except Exception as e:
                                    print(f"VideoBehaviorDetector error (track {track_id}): {e}")

                            # 已报警人员持续标注（即使本次没跑引擎也显示）
                            if p_state.alert_fired and p_state.top_behavior:
                                cv2.putText(frame, f"REPORTED {p_state.max_confidence:.0%}",
                                            (box[0], box[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                            # --- ROI LOGIC ---
                            is_reaching, _ = check_reaching(kpts, cam_roi)
                            if is_reaching:
                                cv2.putText(frame, "RESTRICTED AREA ENT!", (box[0], box[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                # 同一人闯入只报一次
                                if not p_state.roi_alert_fired:
                                     trigger_alert(cam_id, name, "RESTRICTED AREA INTRUSION", frame)
                                     p_state.roi_alert_fired = True

                            if is_bending:
                                cv2.putText(frame, "BENDING", (box[0], box[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                            
                            # --- LOITERING ---
                            center_x = int((box[0] + box[2]) / 2)
                            center_y = int((box[1] + box[3]) / 2)
                            update_heatmap(cam_data, center_x, center_y, frame.shape)
                            
                            is_inside_roi = False
                            if len(cam_roi) >= 3:
                                if cv2.pointPolygonTest(np.array(cam_roi), (center_x, center_y), False) >= 0:
                                    is_inside_roi = True
                            
                            if is_inside_roi:
                                if track_id not in cam_data["roi_entry_times"]:
                                    cam_data["roi_entry_times"][track_id] = time.time()
                                duration = time.time() - cam_data["roi_entry_times"][track_id]
                                cv2.putText(frame, f"{duration:.1f}s", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                                if duration > LOITERING_THRESHOLD:
                                     # 同一人徘徊只报一次
                                     if not p_state.loiter_alert_fired:
                                         trigger_alert(cam_id, name, "LOITERING SUSPICION", frame)
                                         p_state.loiter_alert_fired = True
                            else:
                                if track_id in cam_data["roi_entry_times"]:
                                    del cam_data["roi_entry_times"][track_id]

                    if len(cam_roi) > 0:
                        cv2.polylines(frame, [np.array(cam_roi)], isClosed=True, color=(0, 255, 255), thickness=2)

                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                frames_payload.append({
                    "camera_id": cam_id,
                    "name": name,
                    "data": jpg_as_text,
                    # 实际视频宽高：前端 ROI 画布按此尺寸建立坐标系，点击坐标不再偏移
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "ts": frame_ts
                })

                # --- 证据视频：环形缓冲区追加 + 后5秒采集 ---
                buf = frame_buffers.setdefault(cam_id, _deque(maxlen=PRE_FRAMES))
                buf.append(buffer.tobytes())

                # 获取摄像头实际 FPS（首次取，之后缓存在 cam_data）
                if "fps" not in cam_data or cam_data["fps"] is None:
                    try:
                        cap_fps = cap.cap.get(cv2.CAP_PROP_FPS)
                        cam_data["fps"] = cap_fps if cap_fps and cap_fps > 1 else 30.0
                    except Exception:
                        cam_data["fps"] = 30.0
                use_fps = cam_data["fps"]

                # 如果该摄像头正在采集"后5秒"帧，追加并倒计时
                pending = evidence_pending.get(cam_id)
                if pending is not None:
                    pending["post_frames"].append(buffer.tobytes())
                    pending["post_remaining"] -= 1
                    if pending["post_remaining"] <= 0:
                        # 凑齐前5秒 + 后5秒，后台线程编码（用摄像头实际FPS）
                        threading.Thread(
                            target=_encode_evidence_video,
                            args=(pending["alert_id"], cam_id, pending["cam_name"],
                                  pending["ts"], pending["pre_frames"], pending["post_frames"],
                                  frame.shape[1], frame.shape[0], use_fps),
                            daemon=True
                        ).start()
                        del evidence_pending[cam_id]
            
            frame_count += 1

            # 定期清理消失超过宽限期的 track 状态（同一人离开后再回来可重新报警）
            if frame_count % 300 == 0:
                prune_stale_person_states(time.time())
                if realtime_detector is not None:
                    realtime_detector.prune()
                # 清理已删除摄像头的独立 pose tracker，防止实例泄漏
                active_cam_ids = {cid for cid, _ in current_cams}
                for stale_cam in set(pose_trackers) - active_cam_ids:
                    pose_trackers.pop(stale_cam, None)

            if frames_payload:
                with lock:
                    latest_frame = {
                        "type": "multi_frame",
                        "cameras": frames_payload,
                        "alert": alert_payload,
                        "audio": "siren" if alert_payload else None
                    }
                    # Clear alert_payload after packing into frame to avoid duplicate siren loops
                    alert_payload = None
            
            time.sleep(0.04) 

        except Exception as e:
            print(f"Loop Error: {e}")
            with open("error_log.txt", "a") as f:
                f.write(f"{datetime.now()}: Loop Runtime Error: {e}\n")
            time.sleep(1)


def trigger_alert(cam_id, cam_name, message, frame, confidence=None, behavior_type=None, track_id=None):
    """触发一次报警并写入数据库。返回 alert_id（供同一人后续回写 MAX CONFIDENCE + 证据视频关联）"""
    global alert_payload
    try:
        print(f"ALERT: {message}")
        # 文件名精确到微秒，避免同一秒内多个告警互相覆盖证据文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"alerts/alert_{cam_id}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        # --- 证据视频：冻结该摄像头的前5秒环形缓冲，启动后5秒采集 ---
        alert_id_for_video = None
        pending_old = evidence_pending.get(cam_id)
        if pending_old is None:
            buf = frame_buffers.get(cam_id)
            pre = list(buf) if buf else []   # 冻结前5秒 JPEG bytes
            evidence_pending[cam_id] = {
                "pre_frames": pre,
                "post_frames": [],
                "post_remaining": POST_FRAMES,
                "alert_id": None,   # INSERT 成功后回填
                "ts": timestamp,
                "cam_name": cam_name,
            }
        else:
            print(f"[Evidence] 该摄像头已有证据视频在采集，跳过本次")

        # database
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        alert_id = str(uuid.uuid4())
        c.execute(
            "INSERT INTO alerts (id, message, timestamp, image_path, confidence, behavior_type, track_id) VALUES (?,?,?,?,?,?,?)",
            (alert_id, message, timestamp, filename, confidence, behavior_type, track_id)
        )
        # 如果已有证据视频在等待，回填 alert_id
        if cam_id in evidence_pending:
            evidence_pending[cam_id]["alert_id"] = alert_id
        conn.commit()
        conn.close()

        with lock:
            alert_payload = {
                "id": alert_id,
                "message": message,
                "timestamp": timestamp,
                "image_path": filename,
                "camera_id": cam_id,
                "confidence": confidence,
                "behavior_type": behavior_type
            }
            
        # Send Email/Telegram if enabled (Settings)
        # We can implement a fire-and-forget thread for this to not block loop
        threading.Thread(target=send_notifications, args=(message, filename)).start()

        return alert_id
    except Exception as e:
        print(f"Alert Error: {e}")
        return None

def update_alert_peak(alert_id, behavior_type, confidence):
    """同一 track_id 已报过警后，仅把更高的峰值置信度回写到同一历史记录，不产生新报警"""
    if not alert_id:
        return
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            "UPDATE alerts SET confidence = ?, behavior_type = ? WHERE id = ? AND (confidence IS NULL OR confidence < ?)",
            (confidence, behavior_type, alert_id, confidence)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Update alert peak error: {e}")

def _encode_evidence_video(alert_id, cam_id, cam_name, ts, pre_frames, post_frames, width, height, fps=30.0):
    """后台线程：把前5秒 + 后5秒 JPEG bytes 存到目录（供 MJPEG 推流），同时生成 mp4v 归档"""
    try:
        # 合并帧（pre 在前，post 在后）
        all_frames = pre_frames + post_frames
        if not all_frames:
            print(f"[Evidence] alert={alert_id} 无帧可编码，跳过")
            return

        # 方案A：存 JPEG 帧到 evidence_videos/{alert_id}/ 目录（供浏览器 MJPEG 推流播放）
        frames_dir = f"{EVIDENCE_DIR}/{alert_id}"
        os.makedirs(frames_dir, exist_ok=True)
        for idx, jpg_bytes in enumerate(all_frames):
            frame_path = f"{frames_dir}/frame_{idx:05d}.jpg"
            with open(frame_path, "wb") as f:
                f.write(jpg_bytes)

        # 方案B：生成 mp4v 归档文件（可选，方便下载查看）
        out_path = f"{EVIDENCE_DIR}/evidence_{cam_id}_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if writer.isOpened():
            for jpg_bytes in all_frames:
                arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height))
                    writer.write(frame)
            writer.release()
        else:
            print(f"[Evidence] VideoWriter 打开失败，仅存 JPEG 帧")

        size_kb = os.path.getsize(out_path) / 1024 if os.path.exists(out_path) else 0
        print(f"[Evidence] 编码完成: frames={len(all_frames)}, fps={fps:.1f}, mp4={size_kb:.0f}KB, dir={frames_dir}")

        # 回写 DB：video_path 存帧目录路径（前端 MJPEG 推流端点用）
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("UPDATE alerts SET video_path = ? WHERE id = ?", (frames_dir, alert_id))
            conn.commit()
            conn.close()
        except Exception as db_e:
            print(f"[Evidence] video_path 回写 DB 失败: {db_e}")

    except Exception as e:
        print(f"[Evidence] 编码异常: {e}")

def send_notifications(message, image_path):
    try:
        if current_settings.emailEnabled:
            sender_email = os.getenv("SENDER_EMAIL", current_settings.senderEmail)
            sender_password = os.getenv("SMTP_PASSWORD", current_settings.senderPassword)
            if sender_email and sender_password:
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = current_settings.receiverEmail
                msg['Subject'] = "Theft Guard AI - Security Alert"
                
                body = f"ALERT: {message}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                msg.attach(MIMEText(body, 'plain'))
                
                try:
                    with open(image_path, 'rb') as f:
                        img_data = f.read()
                        image = MIMEImage(img_data, name=os.path.basename(image_path))
                        msg.attach(image)
                except Exception as img_e:
                    print(f"Could not attach image: {img_e}")
                
                server = smtplib.SMTP(current_settings.smtpServer, int(current_settings.smtpPort))
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
                server.quit()
                print("Email notification sent.")

        if current_settings.telegramEnabled:
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN", current_settings.telegramBotToken)
            chat_id = current_settings.telegramChatId
            if bot_token and chat_id:
                url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                with open(image_path, 'rb') as photo:
                    data = {"chat_id": chat_id, "caption": f"🚨 THEFT GUARD ALERT 🚨\n\n{message}"}
                    files = {"photo": photo}
                    resp = requests.post(url, data=data, files=files)
                if resp.status_code == 200:
                    print("Telegram notification sent.")
                else:
                    print(f"Telegram Error: {resp.text}")
    except Exception as e:
        print(f"Notification Error: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            # Send latest frame
            message_to_send = None
            with lock:
                if latest_frame:
                    message_to_send = json.dumps(latest_frame)
            
            if message_to_send:
                await websocket.send_text(message_to_send)

            await asyncio.sleep(0.04) 
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=video_loop, daemon=True)
    t.start()
    yield

app.router.lifespan_context = lifespan

# ============================================================
# 项目A行为识别引擎 - 文件上传分析接口（图片/视频）
# ============================================================

_theft_detector = None
_image_behavior_detector = None

def get_behavior_detectors():
    """懒加载并返回项目A的检测器实例（用于文件上传分析）"""
    global _theft_detector, _image_behavior_detector
    if _theft_detector is None and BEHAVIOR_ENGINE_AVAILABLE:
        try:
            _theft_detector = TheftDetector(model_path="models/yolov11n.pt")
            _image_behavior_detector = ImageBehaviorDetector()
            print("TheftDetector + ImageBehaviorDetector loaded for file analysis.")
        except Exception as e:
            print(f"File analysis detectors init failed: {e}")
            raise HTTPException(status_code=500, detail=f"Behavior engine init failed: {e}")
    return _theft_detector, _image_behavior_detector

@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """分析上传的图片（走实时行为检测器 + YOLOv8-pose）"""
    if not REALTIME_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Realtime behavior engine not available")

    temp_path = f"alerts/upload_{uuid.uuid4()}.jpg"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        frame = cv2.imread(temp_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="无法读取图片")

        # 加载 YOLOv8-pose 检测人
        pose = YOLO('yolov8n-pose.pt')
        rd = RealtimeBehaviorDetector()

        results = pose(frame, verbose=False, classes=[0])
        behaviors = []
        annotated = frame.copy()
        h, w = frame.shape[:2]

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            for i, box in enumerate(boxes):
                pad = 60
                sx1, sy1 = max(0, box[0]-pad), max(0, box[1]-pad)
                sx2, sy2 = min(w, box[2]+pad), min(h, box[3]+pad)
                sub = frame[sy1:sy2, sx1:sx2].copy()

                out = rd.analyze(sub, f"img-{i}")
                if out:
                    # 藏匿事件带 alert_score，伸手提示带 confidence，两者兼容读取
                    out_conf = float(out.get('alert_score', out.get('confidence', 0.0)))
                    behaviors.append({
                        "type": out["type"],
                        "description": out.get("phase", ""),
                        "confidence": out_conf
                    })
                    color = (0, 0, 255) if out_conf >= BEHAVIOR_ALERT_THRESHOLD else (0, 165, 255)
                    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 3)
                    cv2.putText(annotated, f"{out['type']}: {out_conf:.0%}",
                                (box[0], box[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (100,100,100), 2)

        result_path = f"alerts/result_{uuid.uuid4()}.jpg"
        cv2.imwrite(result_path, annotated)

        result_image_b64 = None
        if os.path.exists(result_path):
            with open(result_path, "rb") as f:
                result_image_b64 = base64.b64encode(f.read()).decode("utf-8")
            os.remove(result_path)

        rd.close()
        return {
            "behaviors": behaviors,
            "behaviors_count": len(behaviors),
            "result_image": result_image_b64,
            "type": "image"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    """分析上传的视频（走实时行为检测器 + YOLOv8-pose，逐帧多目标追踪）"""
    if not REALTIME_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Realtime behavior engine not available")

    temp_path = f"alerts/upload_{uuid.uuid4()}.mp4"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="无法读取视频")

        pose = YOLO('yolov8n-pose.pt')
        rd = RealtimeBehaviorDetector()

        behaviors = []          # 展平后的 [(frame, time, type, conf), ...]
        per_track_peak = {}     # track_id -> (frame_idx, time, type, max_conf)
        suspicious_frames = set()
        frame_idx = 0
        PAD = 60

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            h, w = frame.shape[:2]

            if frame_idx % BEHAVIOR_CHECK_INTERVAL != 0:
                continue

            res = pose.track(frame, persist=True, verbose=False, classes=[0])
            if res[0].boxes is None or res[0].boxes.id is None:
                continue
            boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
            tids = res[0].boxes.id.cpu().numpy().astype(int)

            for box, tid in zip(boxes, tids):
                sx1, sy1 = max(0, box[0]-PAD), max(0, box[1]-PAD)
                sx2, sy2 = min(w, box[2]+PAD), min(h, box[3]+PAD)
                sub = frame[sy1:sy2, sx1:sx2].copy()

                out = rd.analyze(sub, f"video-{int(tid)}")
                # 藏匿事件带 alert_score，伸手提示带 confidence，两者兼容读取
                out_conf = float(out.get('alert_score', out.get('confidence', 0.0))) if out else 0.0
                if out and out_conf >= 0.50:
                    peak = per_track_peak.get(int(tid))
                    if peak is None or out_conf > peak[3]:
                        per_track_peak[int(tid)] = (frame_idx, frame_idx / 30.0, out["type"], out_conf)
                    if out_conf >= BEHAVIOR_ALERT_THRESHOLD:
                        suspicious_frames.add(frame_idx)

        cap.release()

        # 组装返回：每 track 只出峰值行为（和实时告警去重策略一致）
        for tid, (fi, t, btype, bconf) in per_track_peak.items():
            behaviors.append({
                "frame": fi,
                "time": t,
                "type": btype,
                "description": f"track {tid} 峰值置信度",
                "confidence": bconf
            })

        rd.close()
        return {
            "behaviors": behaviors,
            "behaviors_count": len(behaviors),
            "suspicious_frames": sorted(suspicious_frames),
            "result_video": None,   # 视频标注编码 base64 太大，前端可后续补
            "type": "video"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    # 默认只绑定回环地址，避免未认证接口直接暴露到外网；
    # 需要局域网访问时通过环境变量 HOST=0.0.0.0 显式开启
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
