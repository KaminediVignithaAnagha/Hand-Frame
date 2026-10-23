"""
effects.py
-----------
Everything about the BACKGROUND PICTURES and their FILTERS, plus the
function that draws the "window" effect inside the rectangle.

ELI5: you drop 2-5 photos into the `pictures/` folder. Each photo can be
shown with a filter (warm, cool, black & white, vintage, painted, pop).
Filtered versions are made once and remembered, so it stays fast.
"""

import os
import time

import cv2
import numpy as np

PICTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pictures")

SUPPORTED = (".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif")


# ---------- picture loading ----------

def cover_resize(img, w, h):
    """Resize to fill w x h without stretching (crops the extra edges)."""
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    resized = cv2.resize(img, (int(iw * scale) + 1, int(ih * scale) + 1))
    y1 = (resized.shape[0] - h) // 2
    x1 = (resized.shape[1] - w) // 2
    return resized[y1:y1 + h, x1:x1 + w]


def read_image(path):
    """Returns (image_or_None, reason_if_failed). Handles iPhone/Mac HEIC too."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".heic", ".heif"):
            try:
                from pillow_heif import register_heif_opener
                from PIL import Image
            except ImportError:
                return None, ("HEIC photo -- run 'python3 -m pip install pillow-heif' "
                              "or export it as JPG from Preview")
            register_heif_opener()
            pil = Image.open(path).convert("RGB")
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), ""
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None, "could not be read as an image"
        return img, ""
    except Exception as e:
        return None, f"error: {e}"


def load_pictures(w, h, max_pictures=5):
    pics, names = [], []
    print(f"\nLooking for pictures in: {PICTURES_DIR}")
    if not os.path.isdir(PICTURES_DIR):
        print("  -> that folder does not exist!")
    else:
        files = [f for f in sorted(os.listdir(PICTURES_DIR)) if not f.startswith(".")]
        files = [f for f in files if not f.lower().endswith(".txt")]
        if not files:
            print("  -> the folder is empty.")
        for fname in files:
            if len(pics) >= max_pictures:
                print(f"  - {fname}: skipped (only the first {max_pictures} pictures are used)")
                continue
            if not fname.lower().endswith(SUPPORTED):
                print(f"  - {fname}: skipped (not a supported picture type)")
                continue
            img, reason = read_image(os.path.join(PICTURES_DIR, fname))
            if img is None:
                print(f"  - {fname}: skipped ({reason})")
                continue
            pics.append(cover_resize(img, w, h))
            names.append(fname)
            print(f"  - {fname}: loaded")

    if not pics:
        print("No usable pictures found -- using colour gradients instead.\n")
        for i, (c1, c2) in enumerate([((255, 80, 80), (80, 80, 255)),
                                      ((80, 255, 160), (255, 160, 80)),
                                      ((200, 60, 200), (60, 200, 255))]):
            g = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(h):
                t = y / h
                g[y, :] = [int(c1[k] * (1 - t) + c2[k] * t) for k in range(3)]
            pics.append(g)
            names.append(f"gradient {i + 1}")
    else:
        print(f"Using {len(pics)} picture(s).\n")
    return pics, names


# ---------- filters (each takes a BGR image, returns a BGR image) ----------

def f_original(img):
    return img


def f_warm(img):
    b, g, r = cv2.split(img.astype(np.float32))
    return cv2.merge([np.clip(b * 0.8, 0, 255), g, np.clip(r * 1.2, 0, 255)]).astype(np.uint8)


def f_cool(img):
    b, g, r = cv2.split(img.astype(np.float32))
    return cv2.merge([np.clip(b * 1.25, 0, 255), g, np.clip(r * 0.8, 0, 255)]).astype(np.uint8)


def f_bw(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=-15)  # a bit more contrast
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def f_vintage(img):
    sepia = np.array([[0.131, 0.534, 0.272],
                      [0.168, 0.686, 0.349],
                      [0.189, 0.769, 0.393]], dtype=np.float32)
    return cv2.transform(img, sepia)


def f_paint(img):
    return cv2.stylization(img, sigma_s=100, sigma_r=0.35)


def f_pop(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.6, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


FILTERS = [
    ("original", f_original),
    ("warm", f_warm),
    ("cool", f_cool),
    ("black & white", f_bw),
    ("vintage", f_vintage),
    ("painted", f_paint),
    ("pop", f_pop),
]


# ---------- background manager ----------

class BackgroundManager:
    def __init__(self, frame_w, frame_h, auto_seconds=3.0):
        self.pictures, self.names = load_pictures(frame_w, frame_h)
        self.pic_idx = 0
        self.filter_idx = 1  # start on "warm" so you see a filter straight away
        self.auto = True
        self.auto_seconds = auto_seconds
        self._last_switch = time.monotonic()
        self._cache = {}

    def current(self):
        key = (self.pic_idx, self.filter_idx)
        if key not in self._cache:
            fn = FILTERS[self.filter_idx][1]
            self._cache[key] = fn(self.pictures[self.pic_idx])
        return self._cache[key]

    def label(self):
        return f"{self.names[self.pic_idx]} | {FILTERS[self.filter_idx][0]}"

    def next_picture(self):
        self.pic_idx = (self.pic_idx + 1) % len(self.pictures)
        self._last_switch = time.monotonic()

    def next_filter(self):
        self.filter_idx = (self.filter_idx + 1) % len(FILTERS)

    def update(self):
        """Call once per frame: auto-changes the picture every few seconds."""
        if self.auto and len(self.pictures) > 1:
            if time.monotonic() - self._last_switch > self.auto_seconds:
                self.next_picture()


# ---------- the rectangle "window" effect ----------

def render_window(frame, background, person_mask, rect, alpha):
    """
    Inside rect (x1, y1, x2, y2): you stay clear (from the webcam) and the
    filtered picture shows behind you. Outside the rect: normal webcam.
    alpha (0..1) fades the whole window in/out.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = rect
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return frame

    output = frame.copy()
    roi_frame = frame[y1:y2, x1:x2].astype(np.float32)
    roi_bg = background[y1:y2, x1:x2].astype(np.float32)
    roi_mask = person_mask[y1:y2, x1:x2][..., None]

    composite = roi_frame * roi_mask + roi_bg * (1.0 - roi_mask)
    blended = roi_frame * (1.0 - alpha) + composite * alpha
    output[y1:y2, x1:x2] = blended.astype(np.uint8)
    return output