"""Lee las fotos de la carpeta 'poses' y guarda la pose corporal de cada una.

Uso:  python pose_enroll.py
Ejecutar cada vez que agregues, quites o cambies fotos, o después de usar
pose_editor.py.

- Las fotos que marcaste a mano con pose_editor.py usan esa pose.
- Las demás se detectan automáticamente. Si hay varias personas, cada una se
  registra como una pose distinta. Si no encuentra a nadie en la foto completa,
  la busca también por trozos (sirve para grupos o personas pequeñas).
- Crea la carpeta 'revision' con cada foto y su esqueleto dibujado, para
  comprobar a simple vista si quedó bien.
"""
import os

import cv2
import numpy as np

from pose_common import (IMG_EXT, L_SH, POSES_DB, POSES_DIR, R_SH, PoseAnalyzer,
                         draw_skeleton, imread, load_manual, load_meta,
                         manual_to_arrays, normalize, usable)

MAX_SIDE = 1920       # las fotos más grandes se reducen solo para detectar
CONF = 0.3            # confianza mínima de detección (más bajo = más tolerante)
MIN_VIS = 0.3         # visibilidad mínima de los hombros para aceptar una pose
REVIEW_DIR = "revision"


def _tiles(w, h):
    """4 trozos que se superponen, de 60% del ancho y alto de la foto."""
    tw, th = int(w * 0.6), int(h * 0.6)
    return [(x, y, x + tw, y + th) for x in (0, w - tw) for y in (0, h - th)]


def _score(xy, vis):
    return float(vis[[0, 11, 12, 13, 14, 15, 16]].mean())


def _same_person(a, b):
    ma, mb = (a[0][L_SH] + a[0][R_SH]) / 2, (b[0][L_SH] + b[0][R_SH]) / 2
    wa = np.linalg.norm(a[0][L_SH] - a[0][R_SH])
    wb = np.linalg.norm(b[0][L_SH] - b[0][R_SH])
    return float(np.linalg.norm(ma - mb)) < 0.6 * max(wa, wb)


def detect_all(analyzer, img):
    """Detecta personas en la foto entera y, además, por trozos ampliados.
    Devuelve una lista de (puntos en píxeles, visibilidad)."""
    h, w = img.shape[:2]
    found = [p for p in analyzer.detect(img) if usable(*p, min_vis=MIN_VIS)]

    extra = []
    for (x0, y0, x1, y1) in _tiles(w, h):
        crop = img[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        s = max(1.0, min(3.0, 1000.0 / max(ch, cw)))
        if s > 1.0:
            crop = cv2.resize(crop, (int(cw * s), int(ch * s)),
                              interpolation=cv2.INTER_CUBIC)
        for xy, v in analyzer.detect(crop):
            xy = xy / s + np.array([x0, y0], np.float32)
            if usable(xy, v, min_vis=MIN_VIS):
                extra.append((xy, v))

    # añadir las de los trozos que no estén ya detectadas (quedándose con la mejor)
    for cand in sorted(extra, key=lambda p: -_score(*p)):
        if not any(_same_person(cand, f) for f in found):
            found.append(cand)
    return found


def save_review(img, found, path):
    out = img.copy()
    t = max(2, img.shape[1] // 300)
    for k, (xy, v) in enumerate(found):
        draw_skeleton(out, xy, v, thick=t, vmin=MIN_VIS)
        mid = (xy[L_SH] + xy[R_SH]) / 2
        cv2.putText(out, str(k + 1), (int(mid[0]), int(mid[1]) - 2 * t - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, t / 3, (0, 255, 255), t, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", out)
    if ok:
        buf.tofile(path)


def main():
    if not os.path.isdir(POSES_DIR):
        raise SystemExit(f"Crea la carpeta '{POSES_DIR}' y pon ahí las fotos.")
    files = sorted(f for f in os.listdir(POSES_DIR) if f.lower().endswith(IMG_EXT))
    if not files:
        raise SystemExit(f"No hay fotos en '{POSES_DIR}' (jpg, jpeg, jpe, png, webp).")
    os.makedirs(REVIEW_DIR, exist_ok=True)

    meta = load_meta()
    manual = load_manual()
    for k in meta:
        if k not in files:
            print(f"[aviso]   poses.json menciona '{k}' pero no está en la carpeta")

    analyzer = None   # se carga solo si alguna foto lo necesita
    names, person, norm, raw, vis = [], [], [], [], []
    omitted = []

    for f in files:
        img = imread(os.path.join(POSES_DIR, f))
        if img is None:
            print(f"[omitida] {f}: no se pudo leer la imagen")
            omitted.append(f)
            continue
        h, w = img.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (int(w * s), int(h * s)))
            h, w = img.shape[:2]

        if f in manual:
            found = []
            for p in manual[f]:
                pts, v = manual_to_arrays(p)
                if v[L_SH] > 0 and v[R_SH] > 0:
                    found.append((pts * np.array([w, h], np.float32), v))
            origen = "a mano"
        else:
            if analyzer is None:
                analyzer = PoseAnalyzer(video=False, num_poses=6, conf=CONF)
            found = detect_all(analyzer, img)
            origen = "automática"

        if not found:
            print(f"[omitida] {f}")
            omitted.append(f)
            continue
        for k, (xy, v) in enumerate(found):
            names.append(f)
            person.append(k)
            norm.append(normalize(xy))
            raw.append(xy / np.array([w, h], np.float32))
            vis.append(v)
        save_review(img, found, os.path.join(REVIEW_DIR, os.path.splitext(f)[0] + ".jpg"))
        n = "" if len(found) == 1 else f", {len(found)} personas"
        sin_texto = "" if f in meta else "  (sin texto en poses.json)"
        print(f"[ok]      {f} ({origen}{n}){sin_texto}")

    if not names:
        print("\nNo se registró ninguna pose.")
        print("Marca las poses a mano con:  python pose_editor.py")
        return
    np.savez(POSES_DB, files=np.array(names), person=np.array(person),
             norm=np.array(norm, np.float32), raw=np.array(raw, np.float32),
             vis=np.array(vis, np.float32))
    print(f"\nListo: {len(names)} pose(s) de {len(set(names))} foto(s) "
          f"guardadas en {POSES_DB}")
    if omitted:
        print(f"\nQuedaron fuera ({len(omitted)}): " + ", ".join(omitted))
        print("Para marcar su pose a mano ejecuta:  python pose_editor.py")
    print(f"La carpeta '{REVIEW_DIR}' muestra los esqueletos sobre cada foto.")


if __name__ == "__main__":
    main()
