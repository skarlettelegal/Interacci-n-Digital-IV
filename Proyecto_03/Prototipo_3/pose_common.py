"""Funciones compartidas del proyecto de gestos corporales (MediaPipe Pose)."""
import json
import os
import time
import unicodedata
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

POSES_DIR = "poses"              # carpeta con las fotos
POSES_META = "poses.json"        # textos opcionales de cada foto
POSES_DB = "pose_descriptors.npz"
POSES_MANUAL = "poses_manual.json"   # poses marcadas a mano
IMG_EXT = (".jpg", ".jpeg", ".jpe", ".png", ".webp", ".bmp")

MODELS = {
    "full": ("pose_landmarker_full.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/1/pose_landmarker_full.task"),
    "lite": ("pose_landmarker_lite.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"),
}


def ensure_model(kind="full"):
    """Descarga el modelo la primera vez si no está en la carpeta."""
    filename, url = MODELS[kind]
    if os.path.exists(filename):
        return filename
    print(f"Descargando modelo de MediaPipe '{kind}' (solo la primera vez)...")
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        raise SystemExit(
            f"No se pudo descargar el modelo ({e}).\n"
            f"Descárgalo a mano desde:\n{url}\n"
            f"y guárdalo en esta carpeta con el nombre {filename}")
    return filename


# ---------------------------------------------------------------------------
# Puntos del cuerpo (33 landmarks de MediaPipe Pose) que usamos para comparar
# ---------------------------------------------------------------------------
NOSE, L_SH, R_SH = 0, 11, 12
JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# Peso de cada punto: codos y muñecas pesan más porque definen el "gesto"
WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
                    1.0, 1.0, 1.0, 1.0, 0.7, 0.7], np.float32)
JOINTS_ARR = np.array(JOINTS)
VIS_MIN = 0.5            # un punto con menos visibilidad se ignora
MIN_SHARED_WEIGHT = 9.0  # mínimo de puntos en común para poder comparar

CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23),
               (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


class PoseAnalyzer:
    def __init__(self, video=False, num_poses=1, kind="full", conf=0.5):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(kind)),
            running_mode=(vision.RunningMode.VIDEO if video
                          else vision.RunningMode.IMAGE),
            num_poses=num_poses,
            min_pose_detection_confidence=conf,
            min_pose_presence_confidence=conf,
            min_tracking_confidence=conf,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.video = video
        self._last_ts = 0

    def detect(self, bgr):
        """Devuelve una lista de (puntos_px (33,2), visibilidad (33,)) por persona."""
        h, w = bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            ts = int(time.monotonic() * 1000)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            res = self.landmarker.detect_for_video(image, ts)
        else:
            res = self.landmarker.detect(image)
        out = []
        for pose in res.pose_landmarks:
            xy = np.array([[p.x * w, p.y * h] for p in pose], np.float32)
            vis = np.array([p.visibility for p in pose], np.float32)
            out.append((xy, vis))
        return out


def usable(xy, vis, min_shoulder_px=20, min_vis=VIS_MIN):
    """¿Se ven ambos hombros lo bastante bien como para comparar?"""
    if vis[L_SH] < min_vis or vis[R_SH] < min_vis:
        return False
    return float(np.linalg.norm(xy[L_SH] - xy[R_SH])) >= min_shoulder_px


def normalize(xy):
    """Independiza la pose de la posición y el tamaño de la persona:
    origen en el centro de los hombros, unidad = ancho de hombros."""
    mid = (xy[L_SH] + xy[R_SH]) / 2
    width = max(float(np.linalg.norm(xy[L_SH] - xy[R_SH])), 1e-6)
    return ((xy - mid) / width).astype(np.float32)


def pose_distances(q, qv, refs, rvs):
    """Distancia entre la pose q y cada pose de referencia (menor = más parecida).
    Solo se comparan los puntos visibles en ambas."""
    idx = JOINTS_ARR
    mask = (qv[idx] >= VIS_MIN)[None, :] & (rvs[:, idx] >= VIS_MIN)
    w = mask * WEIGHTS[None, :]
    d2 = ((refs[:, idx, :] - q[idx][None, :, :]) ** 2).sum(-1)
    wsum = w.sum(1)
    dist = np.sqrt((w * d2).sum(1) / np.maximum(wsum, 1e-6))
    dist = dist + 0.2 * (1.0 - wsum / float(WEIGHTS.sum()))  # prefiere más puntos en común
    dist[wsum < MIN_SHARED_WEIGHT] = np.inf
    return dist


# ---------------------------------------------------------------------------
# Archivos
# ---------------------------------------------------------------------------
def imread(path):
    """Lee una imagen aunque la ruta tenga tildes o ñ (cv2.imread falla en Windows)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_meta():
    if not os.path.exists(POSES_META):
        return {}
    with open(POSES_META, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_db():
    if not os.path.exists(POSES_DB):
        raise SystemExit(f"No existe {POSES_DB}. Ejecuta primero: python pose_enroll.py")
    d = np.load(POSES_DB, allow_pickle=False)
    return {k: d[k] for k in d.files}


# Poses marcadas a mano con pose_editor.py: {foto: [persona, ...]}, donde cada
# persona es {"numero de punto": [x, y, visible]} con x, y como fracción de la foto.
def load_manual():
    if not os.path.exists(POSES_MANUAL):
        return {}
    with open(POSES_MANUAL, encoding="utf-8") as f:
        return json.load(f)


def save_manual(data):
    with open(POSES_MANUAL, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def manual_to_arrays(person):
    """Persona guardada a mano -> (puntos (33,2) en fracción, visibilidad (33,))."""
    pts = np.zeros((33, 2), np.float32)
    vis = np.zeros(33, np.float32)
    for k, (x, y, v) in person.items():
        j = int(k)
        pts[j] = (x, y)
        vis[j] = 1.0 if v else 0.0
    return pts, vis


def draw_skeleton(img, pts, vis, color=(80, 220, 120), thick=3, vmin=VIS_MIN):
    for a, b in CONNECTIONS:
        if vis[a] >= vmin and vis[b] >= vmin:
            cv2.line(img, (int(pts[a][0]), int(pts[a][1])),
                     (int(pts[b][0]), int(pts[b][1])), color, thick, cv2.LINE_AA)
    for j in JOINTS:
        if vis[j] >= vmin:
            cv2.circle(img, (int(pts[j][0]), int(pts[j][1])), thick + 1,
                       (255, 255, 255), -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Texto con tildes sobre imágenes OpenCV (cv2.putText no soporta tildes/ñ)
# ---------------------------------------------------------------------------
_FONT_PATHS = [
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_font_cache = {}


def _font(size):
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for p in _FONT_PATHS:
        if os.path.exists(p):
            font = ImageFont.truetype(p, size)
            break
    if font is None:
        try:
            font = ImageFont.load_default(size)
        except TypeError:
            font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def put_text(img, text, org, size=28, color=(255, 255, 255)):
    if not HAS_PIL:
        text = "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")
        cv2.putText(img, text, (org[0], org[1] + size), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 40, color, 2, cv2.LINE_AA)
        return img
    font = _font(size)
    x, y = org
    l, t, r, b = font.getbbox(text)
    pad = 4
    x0, y0 = max(0, x + l - pad), max(0, y + t - pad)
    x1, y1 = min(img.shape[1], x + r + pad), min(img.shape[0], y + b + pad)
    if x1 <= x0 or y1 <= y0:
        return img
    roi = img[y0:y1, x0:x1]
    pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((x - x0, y - y0), text, font=font,
                             fill=(color[2], color[1], color[0]))
    img[y0:y1, x0:x1] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img
