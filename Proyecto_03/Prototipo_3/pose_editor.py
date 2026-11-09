"""Editor para marcar o corregir A MANO la pose de cada foto.

Uso:
  python pose_editor.py            fotos que el programa no pudo detectar + las que ya editaste
  python pose_editor.py --todas    todas las fotos (para corregir también las automáticas)
  python pose_editor.py --foto 5a.png

Cómo se usa (ver también el texto en la parte de arriba de la ventana):
  - Arrastra los puntos del esqueleto hasta que calcen sobre la persona.
  - Cuadro al centro del pecho: arrastra para mover todo el esqueleto.
  - + / -  : agrandar o achicar el esqueleto.
  - Clic derecho sobre un punto: ocultarlo (parte del cuerpo que no se ve) o volver a mostrarlo.
  - L: ocultar/mostrar las piernas (para fotos donde solo se ve el torso)
  - A: agregar otra persona    X: borrar la persona activa    R: reiniciar la foto
  - N: foto siguiente    B: foto anterior    Q o Esc: guardar y salir
Al terminar, ejecuta  python pose_enroll.py  para que se usen tus correcciones.
"""
import argparse
import math
import os

import cv2
import numpy as np

from pose_common import (CONNECTIONS, IMG_EXT, JOINTS, L_SH, POSES_DB, POSES_DIR,
                         R_SH, PoseAnalyzer, imread, load_manual, manual_to_arrays,
                         put_text, save_manual)

MAX_W, MAX_H = 1200, 740      # tamaño máximo de la foto en pantalla
HEADER_H = 96
MIN_CANVAS_W = 960
GRAB_PX = 16                  # a qué distancia del punto se "agarra"
MAX_SIDE = 1920
COLORS = [(80, 220, 120), (60, 180, 255), (255, 140, 80),
          (200, 100, 255), (80, 230, 230), (255, 255, 100)]
NAMES = {0: "nariz", 11: "hombro", 12: "hombro", 13: "codo", 14: "codo",
         15: "muñeca", 16: "muñeca", 23: "cadera", 24: "cadera",
         25: "rodilla", 26: "rodilla", 27: "tobillo", 28: "tobillo"}


def joint_label(j):
    if j == 0:
        return "nariz"
    # En una persona de frente, los puntos impares quedan a la DERECHA de la foto
    side = "derecha" if j % 2 == 1 else "izquierda"
    return f"{NAMES[j]} ({side} de la foto)"


def template(w, h):
    """Persona de pie, de frente, con los brazos abajo, como fracción de la foto."""
    sw = 0.14 * h
    mid = np.array([w / 2, 0.30 * h], np.float32)
    px = np.zeros((33, 2), np.float32)
    px[11], px[12] = mid + [sw / 2, 0], mid - [sw / 2, 0]
    px[0] = mid + [0, -0.7 * sw]
    for sh, el, wr, hip, kn, an, sg in ((11, 13, 15, 23, 25, 27, 1),
                                        (12, 14, 16, 24, 26, 28, -1)):
        px[el] = px[sh] + [sg * 0.1 * sw, sw]
        px[wr] = px[sh] + [sg * 0.1 * sw, 2 * sw]
        px[hip] = mid + [sg * 0.4 * sw, 1.6 * sw]
        px[kn] = px[hip] + [0, 1.5 * sw]
        px[an] = px[hip] + [0, 3 * sw]
    vis = np.zeros(33, np.float32)
    vis[JOINTS] = 1.0
    return px / np.array([w, h], np.float32), vis


class Person:
    def __init__(self, pts, vis):
        self.pts = pts.astype(np.float32).copy()   # fracción de la foto
        self.vis = vis.astype(np.float32).copy()

    def to_dict(self):
        return {str(j): [round(float(self.pts[j][0]), 5),
                         round(float(self.pts[j][1]), 5),
                         1 if self.vis[j] > 0 else 0] for j in JOINTS}


class Editor:
    def __init__(self, files, manual, analyzer=None, detect=None):
        self.files, self.manual = files, manual
        self.analyzer, self.detect = analyzer, detect
        self.auto_cache = {}
        self.i = 0
        self.msg = ""
        self.drag = None
        self.hover = None
        self.load(0)

    # ---------------- carga y guardado ----------------
    def auto_persons(self, name, img):
        if self.analyzer is None or self.detect is None:
            return []
        if name not in self.auto_cache:
            h, w = img.shape[:2]
            s = min(1.0, MAX_SIDE / max(h, w))
            small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1.0 else img
            sh, sw = small.shape[:2]
            res = []
            for xy, v in self.detect(self.analyzer, small):
                pts = xy / np.array([sw, sh], np.float32)
                vis = np.zeros(33, np.float32)
                vis[JOINTS] = (v[JOINTS] >= 0.5).astype(np.float32)
                res.append((pts, vis))
            self.auto_cache[name] = res
        return [Person(p, v) for p, v in self.auto_cache[name]]

    def initial_persons(self, name, img, use_manual=True):
        if use_manual and name in self.manual:
            return [Person(*manual_to_arrays(p)) for p in self.manual[name]]
        persons = self.auto_persons(name, img)
        return persons or [Person(*template(img.shape[1], img.shape[0]))]

    def load(self, i):
        self.i = i
        name = self.files[i]
        img = imread(os.path.join(POSES_DIR, name))
        if img is None:
            img = np.zeros((480, 640, 3), np.uint8)
        self.img = img
        self.h, self.w = img.shape[:2]
        s = min(MAX_W / self.w, MAX_H / self.h)
        self.dw, self.dh = max(1, int(self.w * s)), max(1, int(self.h * s))
        self.base = cv2.resize(img, (self.dw, self.dh))
        self.cw = max(self.dw, MIN_CANVAS_W)
        self.ox = (self.cw - self.dw) // 2
        self.persons = self.initial_persons(name, img)
        self.active = 0
        self.dirty = False
        self.drop_manual = False
        self.drag = self.hover = None

    def save_current(self):
        name = self.files[self.i]
        if self.dirty:
            if self.persons:
                self.manual[name] = [p.to_dict() for p in self.persons]
            else:
                self.manual.pop(name, None)
        elif self.drop_manual:
            self.manual.pop(name, None)
        save_manual(self.manual)

    def go(self, delta):
        j = self.i + delta
        if 0 <= j < len(self.files):
            self.save_current()
            self.load(j)
            self.msg = "Guardado"
        else:
            self.msg = "No hay más fotos en esa dirección"

    # ---------------- edición ----------------
    def disp(self, p):
        return p.pts * np.array([self.dw, self.dh], np.float32) + \
            np.array([self.ox, HEADER_H], np.float32)

    def nearest(self, x, y, include_center=True):
        best, bd = None, GRAB_PX
        for pi, p in enumerate(self.persons):
            d = self.disp(p)
            for j in JOINTS:
                dist = math.hypot(d[j][0] - x, d[j][1] - y)
                if dist < bd:
                    best, bd = (pi, j), dist
            if include_center:
                c = (d[L_SH] + d[R_SH]) / 2
                dist = math.hypot(c[0] - x, c[1] - y)
                if dist < bd:
                    best, bd = (pi, "c"), dist
        return best

    def on_mouse(self, event, x, y, flags=0, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self.nearest(x, y)
            if hit:
                self.active = hit[0]
                self.drag = (hit, x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag:
                (pi, j), lx, ly = self.drag
                p = self.persons[pi]
                if j == "c":
                    delta = np.array([(x - lx) / self.dw, (y - ly) / self.dh], np.float32)
                    p.pts[JOINTS] += delta
                else:
                    p.pts[j] = ((x - self.ox) / self.dw, (y - HEADER_H) / self.dh)
                    p.vis[j] = 1.0
                self.drag = ((pi, j), x, y)
                self.dirty = True
                self.hover = (pi, j) if j != "c" else None
            else:
                hit = self.nearest(x, y, include_center=False)
                self.hover = hit
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            hit = self.nearest(x, y, include_center=False)
            if hit:
                pi, j = hit
                self.active = pi
                p = self.persons[pi]
                p.vis[j] = 0.0 if p.vis[j] > 0 else 1.0
                self.dirty = True

    def scale(self, factor):
        if not self.persons:
            return
        p = self.persons[self.active]
        size = np.array([self.dw, self.dh], np.float32)
        d = p.pts * size
        c = (d[L_SH] + d[R_SH]) / 2
        p.pts = ((c + (d - c) * factor) / size).astype(np.float32)
        self.dirty = True

    def toggle_legs(self):
        """Oculta (o vuelve a mostrar) caderas, rodillas y tobillos de la persona activa."""
        if not self.persons:
            return
        legs = [23, 24, 25, 26, 27, 28]
        p = self.persons[self.active]
        p.vis[legs] = 0.0 if p.vis[legs].max() > 0 else 1.0
        self.dirty = True

    def add_person(self):
        self.persons.append(Person(*template(self.w, self.h)))
        self.active = len(self.persons) - 1
        self.dirty = True

    def delete_person(self):
        if self.persons:
            self.persons.pop(self.active)
            self.active = max(0, self.active - 1)
            self.dirty = True

    def reset(self):
        self.persons = self.initial_persons(self.files[self.i], self.img, use_manual=False)
        self.active = 0
        self.dirty = False
        self.drop_manual = True
        self.msg = "Foto reiniciada"

    # ---------------- dibujo ----------------
    def render(self):
        canvas = np.zeros((HEADER_H + self.dh, self.cw, 3), np.uint8)
        canvas[:HEADER_H] = (28, 28, 28)
        canvas[HEADER_H:, self.ox:self.ox + self.dw] = self.base
        for pi, p in enumerate(self.persons):
            color = COLORS[pi % len(COLORS)]
            act = pi == self.active
            d = self.disp(p)
            pt = lambda j: (int(d[j][0]), int(d[j][1]))
            for a, b in CONNECTIONS:
                if p.vis[a] > 0 and p.vis[b] > 0:
                    cv2.line(canvas, pt(a), pt(b), color, 3 if act else 2, cv2.LINE_AA)
            for j in JOINTS:
                if p.vis[j] > 0:
                    cv2.circle(canvas, pt(j), 7 if act else 5, color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt(j), 8 if act else 6, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.drawMarker(canvas, pt(j), (150, 150, 150),
                                   cv2.MARKER_TILTED_CROSS, 12, 2)
            c = (d[L_SH] + d[R_SH]) / 2
            cv2.rectangle(canvas, (int(c[0]) - 9, int(c[1]) - 9),
                          (int(c[0]) + 9, int(c[1]) + 9), color, 2)
        if self.hover and self.hover[1] != "c" and self.hover[0] < len(self.persons):
            pi, j = self.hover
            x, y = self.disp(self.persons[pi])[j]
            put_text(canvas, joint_label(j), (int(x) + 14, int(y) - 26), 18, (0, 255, 255))

        name = self.files[self.i]
        tag = "  [editada a mano]" if name in self.manual else ""
        put_text(canvas, f"Foto {self.i + 1}/{len(self.files)}: {name}{tag}   |   "
                         f"persona activa {self.active + 1 if self.persons else 0} de "
                         f"{len(self.persons)}   {self.msg}", (12, 6), 18)
        put_text(canvas, "Arrastra los puntos · cuadro del pecho: mover todo · "
                         "+ / − : tamaño · clic derecho: ocultar o mostrar punto",
                 (12, 36), 16, (200, 200, 200))
        put_text(canvas, "N: siguiente  B: anterior  A: agregar persona  X: borrar persona  "
                         "L: ocultar piernas  R: reiniciar  Q: guardar y salir",
                 (12, 62), 16, (200, 200, 200))
        return canvas

    def key(self, k):
        """Procesa una tecla. Devuelve False si hay que salir."""
        if k in (ord("q"), 27):
            self.save_current()
            return False
        if k == ord("n"):
            self.go(+1)
        elif k == ord("b"):
            self.go(-1)
        elif k == ord("a"):
            self.add_person()
        elif k == ord("x"):
            self.delete_person()
        elif k == ord("r"):
            self.reset()
        elif k == ord("l"):
            self.toggle_legs()
        elif k in (ord("+"), ord("=")):
            self.scale(1.05)
        elif k in (ord("-"), ord("_")):
            self.scale(1 / 1.05)
        return True


def choose_files(args):
    all_files = sorted(f for f in os.listdir(POSES_DIR) if f.lower().endswith(IMG_EXT))
    if args.foto:
        if args.foto not in all_files:
            raise SystemExit(f"No encuentro '{args.foto}' en la carpeta '{POSES_DIR}'.")
        return [args.foto]
    manual = load_manual()
    if args.todas or not os.path.exists(POSES_DB):
        return all_files
    done = set(np.load(POSES_DB, allow_pickle=False)["files"].tolist())
    return [f for f in all_files if f not in done or f in manual]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todas", action="store_true", help="abrir todas las fotos")
    ap.add_argument("--foto", help="abrir solo esta foto, ej: 5a.png")
    args = ap.parse_args()

    if not os.path.isdir(POSES_DIR):
        raise SystemExit(f"No existe la carpeta '{POSES_DIR}'.")
    files = choose_files(args)
    if not files:
        raise SystemExit("No hay fotos pendientes: todas se detectaron solas.\n"
                         "Usa  python pose_editor.py --todas  para revisarlas igual.")

    analyzer = detect = None
    try:   # la detección automática solo sirve como punto de partida
        from pose_enroll import CONF, detect_all
        analyzer = PoseAnalyzer(video=False, num_poses=6, conf=CONF)
        detect = detect_all
    except (Exception, SystemExit):
        print("No se pudo cargar el detector: se empezará con una figura de plantilla.")

    print(f"{len(files)} foto(s) para editar. Se abre la ventana…")
    ed = Editor(files, load_manual(), analyzer, detect)
    win = "Editor de poses"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, ed.on_mouse)
    while True:
        cv2.imshow(win, ed.render())
        k = cv2.waitKey(20) & 0xFF
        if k != 255 and not ed.key(k):
            break
    cv2.destroyAllWindows()
    print("Listo. Ahora ejecuta:  python pose_enroll.py")


if __name__ == "__main__":
    main()
