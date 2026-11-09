"""Gestos corporales en tiempo real: a la izquierda tu cámara, a la derecha la
foto histórica con la pose más parecida a la tuya. Sin textos ni barras.

Uso:  python pose_app.py [--camera 0] [--fullscreen] [--lite]
Teclas:  s = mostrar/ocultar el esqueleto | q o ESC = salir
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

from pose_common import (POSES_DIR, PoseAnalyzer, draw_skeleton, imread,
                         load_db, normalize, pose_distances, usable)

PANEL_H = 640          # alto de la ventana
ALPHA = 0.5            # suavizado de la pose (más bajo = más estable)
LOST_SECONDS = 1.0     # sin persona este tiempo -> se limpia la foto
SWITCH_MARGIN = 0.03   # cuánto mejor debe ser otra foto para cambiar
SWITCH_FRAMES = 4      # fotogramas seguidos antes de cambiar de foto


def fit(img, w, h):
    """Redimensiona manteniendo proporción y centra sobre fondo negro."""
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    resized = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))))
    canvas = np.zeros((h, w, 3), np.uint8)
    y, x = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def open_camera(source):
    """Abre una cámara por número (0, 1, 2…) o por dirección web (DroidCam IP)."""
    if str(source).isdigit():
        idx = int(source)
        if sys.platform.startswith("win"):
            return cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        return cv2.VideoCapture(idx)
    return cv2.VideoCapture(source)


class Photos:
    """Carga cada foto una sola vez."""

    def __init__(self):
        self.imgs = {}

    def get(self, name):
        if name not in self.imgs:
            img = imread(os.path.join(POSES_DIR, name))
            self.imgs[name] = img if img is not None else np.zeros((400, 400, 3), np.uint8)
        return self.imgs[name]


class State:
    def __init__(self):
        self.smooth = None
        self.svis = None
        self.last_seen = 0.0
        self.cur = None
        self.cand = None
        self.cand_n = 0
        self.dist = None


def update_match(state, xy, vis, db, now):
    """Actualiza la pose suavizada y elige la foto más parecida (con histéresis
    para que no parpadee entre fotos)."""
    q = normalize(xy)
    if state.smooth is None:
        state.smooth, state.svis = q, vis.copy()
    else:
        state.smooth = ALPHA * q + (1 - ALPHA) * state.smooth
        state.svis = ALPHA * vis + (1 - ALPHA) * state.svis
    state.last_seen = now

    d = pose_distances(state.smooth, state.svis, db["norm"], db["vis"])
    best = int(np.argmin(d))
    if not np.isfinite(d[best]):
        state.dist = None
        return
    if state.cur is None or not np.isfinite(d[state.cur]):
        state.cur, state.cand, state.cand_n = best, None, 0
    elif best != state.cur and d[best] < d[state.cur] - SWITCH_MARGIN:
        state.cand_n = state.cand_n + 1 if state.cand == best else 1
        state.cand = best
        if state.cand_n >= SWITCH_FRAMES:
            state.cur, state.cand, state.cand_n = best, None, 0
    else:
        state.cand, state.cand_n = None, 0
    state.dist = float(d[state.cur])


def build_photo_panel(db, idx, photos, w, h, show_skel):
    """Foto + recuadro de la persona (si hay varias). Se arma una vez y se guarda."""
    name = str(db["files"][idx])
    img = photos.get(name).copy()
    ih, iw = img.shape[:2]
    pts = db["raw"][idx] * np.array([iw, ih], np.float32)
    vis = db["vis"][idx]
    n_people = int((db["files"] == db["files"][idx]).sum())
    if n_people > 1:   # hay varias personas: marcar a quién corresponde la pose
        ok = vis >= 0.5
        x0, y0 = pts[ok].min(0) - 0.03 * iw
        x1, y1 = pts[ok].max(0) + 0.03 * iw
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)),
                      (80, 220, 120), max(2, iw // 250))
    if show_skel:
        draw_skeleton(img, pts, vis, thick=max(2, iw // 300))
    return fit(img, w, h)


def process(frame_bgr, analyzer, db, photos, state, panels, show_skel, now=None):
    """Procesa un fotograma de la cámara y devuelve la imagen a mostrar."""
    now = time.time() if now is None else now
    frame = cv2.flip(frame_bgr, 1)            # espejo: se ve como un espejo
    fh, fw = frame.shape[:2]
    cw = int(fw * PANEL_H / fh)

    poses = analyzer.detect(frame)
    if poses and usable(*poses[0]):
        xy, vis = poses[0]
        update_match(state, xy, vis, db, now)
        if show_skel:
            draw_skeleton(frame, xy, vis, thick=max(2, fw // 250))
    elif now - state.last_seen > LOST_SECONDS:
        state.smooth = state.svis = state.cur = state.dist = None

    cam = cv2.resize(frame, (cw, PANEL_H))

    if state.cur is None:
        key = ("vacio", cw)
        if key not in panels:
            panels[key] = np.zeros((PANEL_H, cw, 3), np.uint8)
        right = panels[key]
    else:
        key = (state.cur, show_skel, cw)
        if key not in panels:
            panels[key] = build_photo_panel(db, state.cur, photos,
                                            cw, PANEL_H, show_skel)
        right = panels[key]
    return np.hstack([cam, right])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0",
                    help="número de cámara (0, 1, 2…) o URL, ej: http://IP:4747/video")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--lite", action="store_true",
                    help="modelo más liviano, para computadores lentos")
    args = ap.parse_args()

    db = load_db()
    analyzer = PoseAnalyzer(video=True, num_poses=1, kind="lite" if args.lite else "full")
    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara '{args.camera}'.")

    win = "Gestos corporales"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    photos, state, panels = Photos(), State(), {}
    show_skel = True
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow(win, process(frame, analyzer, db, photos, state,
                                panels, show_skel))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            show_skel = not show_skel
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
