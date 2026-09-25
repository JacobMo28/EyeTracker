import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import time
import json
import os
import math
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont


# ---------------- Screen size ----------------
root = tk.Tk()
root.withdraw()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

EDGE = 0.05

# ---------------- Design system ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GAZE_SESSION_DIR = os.path.join(SCRIPT_DIR, "gaze_sessions")

os.makedirs(
    GAZE_SESSION_DIR,
    exist_ok=True
)

FONT_PATH = os.path.join(SCRIPT_DIR, "fonts", "Inter-Regular.ttf")
FONT_FALLBACK = "C:/Windows/Fonts/segoeui.ttf"

COLOR_BG = (23, 17, 13)           # #0D1117 BGR
COLOR_BAR = (34, 27, 22)          # #161B22 BGR
COLOR_ACCENT = (246, 130, 59)     # #3B82F6 BGR

COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_MUTED = (200, 200, 200)
COLOR_TEXT_MUTED_BGR = (180, 180, 180)
COLOR_TEXT_DIM = (140, 140, 140)
COLOR_ACCENT_RGB = (59, 130, 246)
COLOR_SUCCESS_BGR = (100, 200, 80)
COLOR_SUCCESS_RGB = (80, 200, 100)
COLOR_WARN_BGR = (80, 80, 220)
COLOR_WARN_RGB = (220, 80, 80)

TOP_BAR_H = 36
FONT_SIZE_TITLE = 38
FONT_SIZE_BODY = 24
FONT_SIZE_SMALL = 18
BODY_LINE_SPACING = 36
CAMERA_CORNER_RADIUS = 12
GAZE_TRAIL_LEN = 8
CROSSFADE_DURATION = 0.5
SESSION_COMPLETE_HOLD = 3.0
COUNTDOWN_STEP = 1.0

debug_mode = False
splash_active = True
instructions_active = False
last_key_time = time.time()

# Gaze cursor trail
gaze_trail = []

# Calibration UX
calib_flash_until = 0.0
calib_flash_pos = None

# Image session UX
session_countdown_active = False
session_countdown_start = None
session_complete_active = False
session_complete_start = None
session_complete_filename = None
image_crossfade_active = False
image_crossfade_start = None
crossfade_prev_img = None
crossfade_prev_rect = None

# ---------------- Time ----------------
start_time = time.time()

# ---------------- Recording ----------------
recording = False
record_start_time = None
record_duration = 10  # seconds

gaze_recording = []


# ---------------- Image presentation mode ----------------
loaded_images = []
current_image_index = 0
image_showing = False
image_start_time = None
image_duration = 10
current_image_id = None


def import_images():
    global loaded_images
    files = filedialog.askopenfilenames(
        title="Select images",
        filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.bmp *.webp")
        ]
    )

    loaded_images = list(files)
    print(f"Loaded {len(loaded_images)} images")


def start_image_session():
    global session_countdown_active, session_countdown_start

    if not loaded_images:
        print("No images loaded")
        return

    session_countdown_active = True
    session_countdown_start = time.time()
    print("Image session countdown started")


def begin_image_session_after_countdown():
    global image_showing, current_image_index, image_start_time
    global session_countdown_active, session_countdown_start

    session_countdown_active = False
    session_countdown_start = None
    current_image_index = 0
    image_showing = True
    image_start_time = time.time()
    start_gaze_recording()
    print("Image recording started")


def get_display_image():
    global current_image_id

    if current_image_index >= len(loaded_images):
        return None

    path = loaded_images[current_image_index]
    current_image_id = os.path.basename(path)

    img = cv2.imread(path)

    if img is None:
        return None

    h, w = img.shape[:2]
    scale = min(screen_w / w, screen_h / h) * 0.8

    img = cv2.resize(
        img,
        (int(w * scale), int(h * scale))
    )

    return img


def image_screen_rect(img):
    ih, iw = img.shape[:2]
    x = (screen_w - iw) // 2
    y = (screen_h - ih) // 2
    return x, y, iw, ih


def advance_image():
    global current_image_index, image_start_time, image_showing
    global image_crossfade_active, image_crossfade_start
    global crossfade_prev_img, crossfade_prev_rect
    global session_complete_active, session_complete_start, session_complete_filename

    prev_img = get_display_image()
    if prev_img is not None:
        crossfade_prev_img = prev_img.copy()
        crossfade_prev_rect = image_screen_rect(prev_img)

    current_image_index += 1

    if current_image_index >= len(loaded_images):
        image_showing = False
        recording_stop()
        session_complete_filename = save_gaze_data()
        session_complete_active = True
        session_complete_start = time.time()
        image_crossfade_active = False
        crossfade_prev_img = None
        print("Image session complete")
    else:
        image_crossfade_active = True
        image_crossfade_start = time.time()
        image_start_time = time.time()


def recording_stop():
    global recording
    recording = False

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ---------------- MediaPipe ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ---------------- Fullscreen window ----------------
window_name = "Eye Tracker"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ---------------- Calibration / learning ----------------
order = ["tl", "tc", "tr", "ml", "c", "mr", "bl", "bc", "br"]
samples = []
idx = 0
calibration_done = False

# Regression model: raw gaze -> screen x/y
# model = (beta_x, beta_y)
model = None

# Fix 1: yaw/pitch clamp bounds (set after calibration to prevent extrapolation)
calib_yaw_min   = -0.3
calib_yaw_max   =  0.3
calib_pitch_min = -0.3
calib_pitch_max =  0.3

# Smoothing
smoothed_x = None
smoothed_y = None

# Continuous correction mode
correction_mode = False
correction_raw_gaze = None
correction_target = None
correction_gaze_samples = []    # Fix 3: accumulate gaze throughout correction session

# Calibration sample collection (Fix 1 + Fix 5)
COLLECT_FRAMES = 25             # frames to average per calibration point
BLINK_THRESHOLD = 0.15          # min eye-openness ratio to accept a frame
collecting = False              # True while gathering frames for current point
pending_gaze_samples = []       # frames accumulated so far for current point

# Head-position lock
# The regression model only ever sees gaze/pose data from wherever the user's
# head was during calibration, so accuracy falls apart if they drift from
# that spot afterward. We capture a reference head box during the first
# calibration point and show it continuously so the user can see exactly
# where to keep their head, and get an immediate green/red signal if they move.
POS_TOLERANCE = 0.035           # allowed center drift, fraction of frame dimension
SIZE_TOLERANCE = 0.12           # allowed box-size (i.e. distance from camera) drift, fraction

YAW_TOLERANCE = 0.16     # radians (~9.2 degrees)
PITCH_TOLERANCE = 0.16   # radians (~9.2 degrees)

ref_yaw = None
ref_pitch = None

head_yaw_error = 0.0
head_pitch_error = 0.0

head_ref_bbox = None            # (x1,y1,x2,y2) normalized frame coords, set after point 1
head_ref_size = None            # reference box size (max of w,h) used for distance comparison
pending_bbox_samples = []       # bbox frames accumulated during the first calib point
in_head_position = True         # this frame's status vs. the reference box
head_pos_dx = 0.0               # this frame's horizontal drift (normalized, + = moved right)
head_pos_dy = 0.0               # this frame's vertical drift (normalized, + = moved down)
head_size_ratio = 1.0           # this frame's size ratio vs reference (>1 = closer to camera)

# Backspace toggles this on/off, letting the presenter bypass the position
# lock (red box) during a live demo without it actually blocking calibration.
head_lock_override = False


# ---------------- Helpers ----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clamp01(v):
    return clamp(v, 0.0, 1.0)


def draw_tiny_dot(img, pos, color, radius=3):
    cv2.circle(img, pos, radius, color, -1, lineType=cv2.LINE_AA)


def target_pos(key):
    return {
        "tl": (int(screen_w * EDGE), int(screen_h * EDGE)),
        "tc": (int(screen_w * 0.50), int(screen_h * EDGE)),
        "tr": (int(screen_w * (1 - EDGE)), int(screen_h * EDGE)),

        "ml": (int(screen_w * EDGE), int(screen_h * 0.50)),
        "c":  (int(screen_w * 0.50), int(screen_h * 0.50)),
        "mr": (int(screen_w * (1 - EDGE)), int(screen_h * 0.50)),

        "bl": (int(screen_w * EDGE), int(screen_h * (1 - EDGE))),
        "bc": (int(screen_w * 0.50), int(screen_h * (1 - EDGE))),
        "br": (int(screen_w * (1 - EDGE)), int(screen_h * (1 - EDGE))),
    }[key]



# ---------------- Render infrastructure ----------------

_font_cache = {}


def get_font(size):
    if size not in _font_cache:
        for path in (FONT_PATH, FONT_FALLBACK):
            try:
                _font_cache[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def build_vignette_mask(w, h, strength=0.50):
    y_coords = np.linspace(-1, 1, h, dtype=np.float32)
    x_coords = np.linspace(-1, 1, w, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)
    dist = np.sqrt(xx * xx + yy * yy)
    alpha = np.clip((dist - 0.3) / 0.9, 0, 1) ** 1.4 * strength
    return alpha[..., np.newaxis]


VIGNETTE_ALPHA = build_vignette_mask(screen_w, screen_h)


def apply_vignette(canvas):
    canvas[:] = np.clip(
        canvas.astype(np.float32) * (1.0 - VIGNETTE_ALPHA),
        0, 255
    ).astype(np.uint8)


class TextRenderer:
    """Batch PIL text pass — one BGR↔RGB round-trip per frame."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        self.draw = ImageDraw.Draw(self.pil_img)

    def text(self, position, text, size=FONT_SIZE_BODY, fill=COLOR_TEXT):
        if text:
            self.draw.text(position, text, font=get_font(size), fill=fill)

    def text_box(self, lines, x=40, y=None, size=FONT_SIZE_BODY, line_spacing=BODY_LINE_SPACING):
        if y is None:
            y = TOP_BAR_H + 16
        for i, line in enumerate(lines):
            if line:
                self.text((x, y + i * line_spacing), line, size=size)

    def text_centered(self, text, y, size=FONT_SIZE_TITLE, fill=COLOR_TEXT):
        font = get_font(size)
        bbox = self.draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        self.text(((screen_w - w) // 2, y), text, size=size, fill=fill)

    def text_right(self, text, y, size=FONT_SIZE_BODY, fill=COLOR_TEXT_MUTED, margin=20):
        font = get_font(size)
        bbox = self.draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        self.text((screen_w - w - margin, y), text, size=size, fill=fill)

    def text_centered_fade(self, text, y, size=FONT_SIZE_TITLE, fill=COLOR_TEXT, opacity=1.0):
        if opacity <= 0.01:
            return
        faded = tuple(int(c * opacity) for c in fill)
        self.text_centered(text, y, size=size, fill=faded)

    def key_chip(self, x, y, key, label="", opacity=1.0):
        if opacity <= 0.01:
            return x
        chip_h = 34
        key_text = f" {key} "
        font = get_font(FONT_SIZE_SMALL)
        kb = self.draw.textbbox((0, 0), key_text, font=font)
        kw = kb[2] - kb[0]
        chip_w = kw + 20
        if label:
            lb = self.draw.textbbox((0, 0), label, font=font)
            chip_w = kw + (lb[2] - lb[0]) + 28

        bg = tuple(int(c * opacity) for c in (40, 48, 58))
        border = tuple(int(c * opacity) for c in COLOR_ACCENT_RGB)
        text_fill = tuple(int(255 * opacity) for _ in range(3))
        muted_fill = tuple(int(c * opacity) for c in COLOR_TEXT_MUTED)

        self.draw.rounded_rectangle(
            (x, y, x + chip_w, y + chip_h),
            radius=8,
            fill=bg,
            outline=border,
            width=1,
        )
        self.draw.text((x + 10, y + 7), key_text, font=font, fill=text_fill)
        if label:
            self.draw.text((x + kw + 18, y + 7), label, font=font, fill=muted_fill)
        return x + chip_w + 12

    def key_chip_row(self, chips, y, opacity=1.0):
        if opacity <= 0.01:
            return
        total_w = 0
        for key, label in chips:
            chip_h = 34
            key_text = f" {key} "
            font = get_font(FONT_SIZE_SMALL)
            kb = self.draw.textbbox((0, 0), key_text, font=font)
            kw = kb[2] - kb[0]
            chip_w = kw + 20
            if label:
                lb = self.draw.textbbox((0, 0), label, font=font)
                chip_w = kw + (lb[2] - lb[0]) + 28
            total_w += chip_w + 12
        x = (screen_w - total_w) // 2
        for key, label in chips:
            x = self.key_chip(x, y, key, label, opacity=opacity)

    def flush(self):
        self.canvas[:] = cv2.cvtColor(np.array(self.pil_img), cv2.COLOR_RGB2BGR)


def get_mode_name():
    if splash_active:
        return "Intro"
    if instructions_active:
        return "Before You Start"
    if session_countdown_active:
        return "Get Ready"
    if session_complete_active:
        return "Complete"
    if image_showing:
        return "Image Session"
    if correction_mode:
        return "Correction"
    if idx < len(order):
        return "Calibration"
    return "Tracking"


def format_elapsed(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def draw_top_bar(ui):
    text_y = max(4, (TOP_BAR_H - FONT_SIZE_BODY) // 2)
    ui.text((16, text_y), get_mode_name(), size=FONT_SIZE_BODY, fill=COLOR_TEXT_MUTED)
    ui.text_right(format_elapsed(time.time() - start_time), y=text_y)


def hint_opacity():
    age = time.time() - last_key_time
    if age <= 8.0:
        return 1.0
    return clamp01(1.0 - (age - 8.0) / 2.0)


def apply_dim_overlay(canvas, alpha=0.62):
    overlay = np.full_like(canvas, COLOR_BG)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)


def make_rounded_mask(w, h, radius):
    mask = np.zeros((h, w), dtype=np.uint8)
    r = min(radius, w // 2, h // 2)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    cv2.circle(mask, (r, r), r, 255, -1)
    cv2.circle(mask, (w - r, r), r, 255, -1)
    cv2.circle(mask, (r, h - r), r, 255, -1)
    cv2.circle(mask, (w - r, h - r), r, 255, -1)
    return mask


def paste_camera_rounded(canvas, frame, x, y, face_detected):
    h, w = frame.shape[:2]
    end_x = min(x + w, screen_w)
    end_y = min(y + h, screen_h)
    vis_w = end_x - x
    vis_h = end_y - y
    if vis_w <= 0 or vis_h <= 0:
        return 0, 0, 0, 0

    crop = frame[:vis_h, :vis_w].copy()
    if not face_detected:
        red_tint = np.zeros_like(crop)
        red_tint[:, :] = COLOR_WARN_BGR
        crop = cv2.addWeighted(crop, 0.72, red_tint, 0.28, 0)

    mask = make_rounded_mask(vis_w, vis_h, CAMERA_CORNER_RADIUS)
    mask_f = (mask.astype(np.float32) / 255.0)[..., np.newaxis]
    roi = canvas[y:end_y, x:end_x].astype(np.float32)
    canvas[y:end_y, x:end_x] = (
        crop.astype(np.float32) * mask_f + roi * (1.0 - mask_f)
    ).astype(np.uint8)
    return vis_w, vis_h, x, y


def draw_progress_bar(canvas, x, y, width, height, progress, color=COLOR_ACCENT):
    progress = clamp01(progress)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), COLOR_BAR, -1)
    fill_w = int(width * progress)
    if fill_w > 0:
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + height), color, -1)


def draw_calib_progress_grid(canvas, current_idx, pulse_t):
    grid_cx = screen_w // 2
    grid_cy = TOP_BAR_H + 72
    span_x = 140
    span_y = 100
    dot_r = 7

    for i, key in enumerate(order):
        col = i % 3
        row = i // 3
        px = grid_cx - span_x + col * span_x
        py = grid_cy - span_y + row * span_y

        if i < current_idx:
            cv2.circle(canvas, (px, py), dot_r, COLOR_ACCENT, -1, lineType=cv2.LINE_AA)
        elif i == current_idx:
            pulse_r = dot_r + int(3 * (0.5 + 0.5 * math.sin(pulse_t * 4.0)))
            cv2.circle(canvas, (px, py), pulse_r, COLOR_ACCENT, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (px, py), dot_r, COLOR_ACCENT, -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(canvas, (px, py), dot_r, COLOR_TEXT_MUTED_BGR, 1, lineType=cv2.LINE_AA)


def draw_calib_target(canvas, key, collecting, collect_progress, pulse_t, flash_active):
    x, y = target_pos(key)
    if flash_active:
        cv2.circle(canvas, (x, y), 14, COLOR_SUCCESS_BGR, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, (x - 7, y), (x - 2, y + 6), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (x - 2, y + 6), (x + 9, y - 7), (255, 255, 255), 2, cv2.LINE_AA)
        return x, y

    pulse_r = 16 + int(8 * (0.5 + 0.5 * math.sin(pulse_t * 3.0)))
    ring_overlay = canvas.copy()
    cv2.circle(ring_overlay, (x, y), pulse_r, COLOR_ACCENT, 2, lineType=cv2.LINE_AA)
    cv2.addWeighted(ring_overlay, 0.55, canvas, 0.45, 0, canvas)
    cv2.circle(canvas, (x, y), 7, COLOR_ACCENT, -1, lineType=cv2.LINE_AA)

    if collecting and collect_progress > 0:
        arc_r = 28
        angle = int(360 * collect_progress)
        cv2.ellipse(
            canvas, (x, y), (arc_r, arc_r), -90, 0, angle,
            COLOR_ACCENT, 3, cv2.LINE_AA,
        )
    return x, y


def update_gaze_trail(x, y):
    global gaze_trail
    gaze_trail.append((int(x), int(y)))
    if len(gaze_trail) > GAZE_TRAIL_LEN:
        gaze_trail.pop(0)


def draw_gaze_cursor(canvas, x, y):
    n = len(gaze_trail)
    for i, (tx, ty) in enumerate(gaze_trail[:-1]):
        t = (i + 1) / max(n, 1)
        radius = max(2, int(5 * t))
        alpha = 0.12 + 0.28 * t
        dot_overlay = canvas.copy()
        cv2.circle(dot_overlay, (tx, ty), radius, COLOR_ACCENT, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(dot_overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    ring_overlay = canvas.copy()
    cv2.circle(ring_overlay, (x, y), 18, COLOR_ACCENT, 1, lineType=cv2.LINE_AA)
    cv2.addWeighted(ring_overlay, 0.60, canvas, 0.40, 0, canvas)
    cv2.circle(canvas, (x, y), 7, COLOR_ACCENT, -1, lineType=cv2.LINE_AA)


def draw_crosshair(canvas, cx, cy, size=18, color=(255, 255, 255), gap=5, thickness=2):
    cv2.line(canvas, (cx - size, cy), (cx - gap, cy), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (cx + gap, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - size), (cx, cy - gap), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy + gap), (cx, cy + size), color, thickness, cv2.LINE_AA)


def draw_correction_guides(canvas, gaze_xy, target_xy):
    gx, gy = int(gaze_xy[0]), int(gaze_xy[1])
    tx, ty = int(target_xy[0]), int(target_xy[1])
    draw_gaze_cursor(canvas, gx, gy)
    draw_crosshair(canvas, tx, ty)
    cv2.arrowedLine(
        canvas, (gx, gy), (tx, ty),
        COLOR_ACCENT, 2, tipLength=0.12, line_type=cv2.LINE_AA,
    )


def draw_stimulus_image(canvas, img, crossfade_t=None, prev_img=None, prev_rect=None):
    if img is None:
        return
    x, y, iw, ih = image_screen_rect(img)
    black = np.zeros_like(img)

    if crossfade_t is not None and prev_img is not None and prev_rect is not None:
        px, py, pw, ph = prev_rect
        if crossfade_t < 0.5:
            alpha = crossfade_t * 2.0
            if pw == iw and ph == ih:
                blended = cv2.addWeighted(prev_img, 1.0 - alpha, black, alpha, 0)
            else:
                blended = cv2.addWeighted(prev_img, 1.0 - alpha, black, alpha, 0)
            canvas[py:py + ph, px:px + pw] = blended
        else:
            alpha = (crossfade_t - 0.5) * 2.0
            blended = cv2.addWeighted(black, 1.0 - alpha, img, alpha, 0)
            canvas[y:y + ih, x:x + iw] = blended
    else:
        canvas[y:y + ih, x:x + iw] = img


def draw_splash(ui):
    ui.text_centered("Eye Tracking Demo", screen_h // 2 - 80, size=52)
    ui.text_centered("Gaze-based stimulus analysis", screen_h // 2 - 10, size=FONT_SIZE_BODY, fill=COLOR_TEXT_MUTED)
    ui.text_centered("Press any key to begin calibration", screen_h // 2 + 50, size=FONT_SIZE_BODY, fill=COLOR_TEXT_DIM)


def draw_instructions(ui):
    ui.text_centered("Before You Start", screen_h // 2 - 190, size=FONT_SIZE_TITLE)

    lines = [
        "In a moment you'll look at 9 dots and press C to lock in each one.",
        "",
        "The key thing: wherever your head is for that FIRST dot becomes",
        "your \"home\" position for the rest of the session.",
        "",
        "Once it's set, a box appears around your face in the camera",
        "preview below \u2014 green means you're still in place, red means",
        "you've drifted and should move back before continuing.",
        "",
        "So find a comfortable position you can hold, then get started.",
    ]
    y = screen_h // 2 - 130
    for line in lines:
        if line:
            ui.text_centered(line, y, size=FONT_SIZE_BODY, fill=COLOR_TEXT_MUTED)
        y += BODY_LINE_SPACING

    ui.text_centered("Press any key to begin calibration", y + 20, size=FONT_SIZE_BODY, fill=COLOR_TEXT_DIM)


def draw_countdown(ui):
    elapsed = time.time() - session_countdown_start
    if elapsed >= COUNTDOWN_STEP * 4:
        return True

    step = int(elapsed // COUNTDOWN_STEP)
    local_t = elapsed - step * COUNTDOWN_STEP
    opacity = math.sin(local_t / COUNTDOWN_STEP * math.pi)

    if step <= 2:
        label = str(3 - step)
        ui.text_centered_fade(label, screen_h // 2 - 40, size=120, opacity=opacity)
    else:
        ui.text_centered_fade("GO", screen_h // 2 - 40, size=100, fill=COLOR_ACCENT_RGB, opacity=opacity)
    return False


def draw_session_complete(ui):
    ui.text_centered("Session Complete", screen_h // 2 - 60, size=FONT_SIZE_TITLE)
    if session_complete_filename:
        ui.text_centered(session_complete_filename, screen_h // 2 - 10, size=FONT_SIZE_BODY, fill=COLOR_TEXT_MUTED)
    ui.key_chip_row(
        [("S", "New Session"), ("R", "Recalibrate")],
        screen_h // 2 + 50,
    )


def avg_point(face, ids):
    xs = [face.landmark[i].x for i in ids]
    ys = [face.landmark[i].y for i in ids]
    return float(np.mean(xs)), float(np.mean(ys))


def eye_openness_ratio(face, upper_id, lower_id, corner_a_id, corner_b_id):
    """
    Vertical eyelid gap divided by horizontal eye width.
    Drops sharply during a blink; use BLINK_THRESHOLD to gate sample collection.
    Left eye:  upper=159, lower=145, corners=33,133
    Right eye: upper=386, lower=374, corners=362,263
    """
    vert  = abs(face.landmark[upper_id].y - face.landmark[lower_id].y)
    horiz = abs(face.landmark[corner_a_id].x - face.landmark[corner_b_id].x)
    if horiz < 1e-6:
        return 0.0
    return vert / horiz


def normalize_eye(iris_xy, corner_a_xy, corner_b_xy, v_top_anchor_xy, v_bottom_anchor_xy):
    """
    Normalize iris position within the eye region.

    Horizontal: iris position relative to the eye corners.
    Vertical: eyebrow above + cheekbone below, which are more stable than eyelids.
    """
    ix, iy = iris_xy
    ax, ay = corner_a_xy
    bx, by = corner_b_xy
    top_y = v_top_anchor_xy[1]
    bottom_y = v_bottom_anchor_xy[1]

    left_x = min(ax, bx)
    right_x = max(ax, bx)

    if abs(right_x - left_x) < 1e-6 or abs(bottom_y - top_y) < 1e-6:
        return None

    norm_x = (ix - left_x) / (right_x - left_x)
    norm_y = (iy - top_y) / (bottom_y - top_y)
    return norm_x, norm_y


def poly_features(gx, gy, yaw=0.0, pitch=0.0):
    # Quadratic gaze terms + linear head-pose terms + cross terms.
    return np.array([
        1.0,
        gx, gy,
        gx * gx, gx * gy, gy * gy,
        yaw, pitch,
        gx * yaw, gy * pitch,
    ], dtype=np.float64)


def fit_model(samples):
    """
    Fit a small regularized quadratic model:
        screen_x = f(raw_gaze_x, raw_gaze_y, yaw, pitch)
        screen_y = g(raw_gaze_x, raw_gaze_y, yaw, pitch)
    """
    if len(samples) < 6:
        return None

    X = np.vstack([poly_features(gx, gy, yaw, pitch) for (gx, gy, yaw, pitch), _ in samples])
    yx = np.array([sx for _, (sx, sy) in samples], dtype=np.float64)
    yy = np.array([sy for _, (sx, sy) in samples], dtype=np.float64)

    lam = 1e-3
    eye = np.eye(X.shape[1], dtype=np.float64)
    eye[0, 0] = 0.0  # do not regularize bias term

    xtx = X.T @ X + lam * eye
    beta_x = np.linalg.solve(xtx, X.T @ yx)
    beta_y = np.linalg.solve(xtx, X.T @ yy)
    return beta_x, beta_y

def predict_screen(gx, gy, yaw, pitch, model):
    if model is None:
        return None
    feats = poly_features(gx, gy, yaw, pitch)
    beta_x, beta_y = model
    x = float(feats @ beta_x)
    y = float(feats @ beta_y)
    return x, y

def refit_model():
    global model, calib_yaw_min, calib_yaw_max, calib_pitch_min, calib_pitch_max
    model = fit_model(samples)
    if samples:
        calib_yaw_min   = min(s[0][2] for s in samples)
        calib_yaw_max   = max(s[0][2] for s in samples)
        calib_pitch_min = min(s[0][3] for s in samples)
        calib_pitch_max = max(s[0][3] for s in samples)


def is_arrow_left(key):
    return key in (2424832, 81, 65361)
def is_arrow_up(key):
    return key in (2490368, 82, 65362)
def is_arrow_right(key):
    return key in (2555904, 83, 65363)
def is_arrow_down(key):
    return key in (2621440, 84, 65364)


# ---------------- Head pose ----------------
# 6 stable 3-D face landmarks (mm, generic model).
# Order: nose tip, chin, L eye corner, R eye corner, L mouth corner, R mouth corner.
HEAD_POSE_LM_IDS = [1, 152, 33, 263, 57, 287]
HEAD_POSE_3D = np.array([
    [   0.0,    0.0,    0.0],   # nose tip
    [   0.0, -330.0,  -65.0],   # chin
    [-225.0,  170.0, -135.0],   # left eye outer corner
    [ 225.0,  170.0, -135.0],   # right eye outer corner
    [-150.0, -150.0, -125.0],   # left mouth corner
    [ 150.0, -150.0, -125.0],   # right mouth corner
], dtype=np.float64)


def get_head_pose(face, cam_w, cam_h):
    """
    Return (yaw, pitch, rvec, tvec, cam_mat).
    Yaw   > 0 → face turned right.
    Pitch > 0 → face tilted upward.
    Returns (0.0, 0.0, None, None, None) if solvePnP fails.
    """
    pts_2d = np.array([
        [face.landmark[i].x * cam_w, face.landmark[i].y * cam_h]
        for i in HEAD_POSE_LM_IDS
    ], dtype=np.float64)

    focal = float(cam_w)
    cx, cy = cam_w / 2.0, cam_h / 2.0
    cam_mat = np.array([
        [focal, 0.0,   cx],
        [0.0,   focal, cy],
        [0.0,   0.0,  1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        HEAD_POSE_3D, pts_2d, cam_mat, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, None, None, None

    rot_mat, _ = cv2.Rodrigues(rvec)

    # ZYX Euler decomposition — correct matrix element indices:
    #   R[2,0] = -sin(pitch)  →  pitch = arcsin(-R[2,0])
    #   R[1,0]/R[0,0] = tan(yaw)  →  yaw = arctan2(R[1,0], R[0,0])
    # np.clip guards against numerical drift outside [-1, 1] that would NaN arcsin.
    pitch = float(np.arcsin(float(np.clip(-rot_mat[2, 0], -1.0, 1.0))))
    yaw   = float(np.arctan2(float(rot_mat[1, 0]), float(rot_mat[0, 0])))
    return yaw, pitch, rvec, tvec, cam_mat

def get_face_bbox(face):
    """Normalized (x1, y1, x2, y2) bounding box over all face landmarks."""
    xs = [lm.x for lm in face.landmark]
    ys = [lm.y for lm in face.landmark]
    return (min(xs), min(ys), max(xs), max(ys))


def check_head_position(
    bbox,
    ref_bbox,
    ref_size,
    yaw,
    pitch,
    reference_yaw,
    reference_pitch
):
    """
    Checks:
    - head position in frame
    - head distance
    - head orientation

    Returns:
    (valid, dx, dy, size_ratio, yaw_error, pitch_error)
    """

    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0

    rcx = (ref_bbox[0] + ref_bbox[2]) / 2.0
    rcy = (ref_bbox[1] + ref_bbox[3]) / 2.0

    dx = cx - rcx
    dy = cy - rcy


    cur_size = max(
        bbox[2] - bbox[0],
        bbox[3] - bbox[1]
    )

    size_ratio = cur_size / ref_size if ref_size else 1.0


    yaw_error = yaw - reference_yaw
    pitch_error = pitch - reference_pitch


    pos_ok = (
        abs(dx) < POS_TOLERANCE and
        abs(dy) < POS_TOLERANCE
    )

    size_ok = (
        1.0 - SIZE_TOLERANCE
        <
        size_ratio
        <
        1.0 + SIZE_TOLERANCE
    )

    pose_ok = (
        abs(yaw_error) < YAW_TOLERANCE and
        abs(pitch_error) < PITCH_TOLERANCE
    )


    return (
        pos_ok and size_ok and pose_ok,
        dx,
        dy,
        size_ratio,
        yaw_error,
        pitch_error
    )

def draw_head_position_guide(frame, cam_w, cam_h, ref_bbox, bbox, in_position):
    """Draws the reference box (dashed) and the live box (green/red) on the camera feed."""
    if ref_bbox is not None:
        rx1, ry1 = int(ref_bbox[0] * cam_w), int(ref_bbox[1] * cam_h)
        rx2, ry2 = int(ref_bbox[2] * cam_w), int(ref_bbox[3] * cam_h)
        dash = 8
        for x in range(rx1, rx2, dash * 2):
            cv2.line(frame, (x, ry1), (min(x + dash, rx2), ry1), (210, 210, 210), 1, cv2.LINE_AA)
            cv2.line(frame, (x, ry2), (min(x + dash, rx2), ry2), (210, 210, 210), 1, cv2.LINE_AA)
        for y in range(ry1, ry2, dash * 2):
            cv2.line(frame, (rx1, y), (rx1, min(y + dash, ry2)), (210, 210, 210), 1, cv2.LINE_AA)
            cv2.line(frame, (rx2, y), (rx2, min(y + dash, ry2)), (210, 210, 210), 1, cv2.LINE_AA)

    if bbox is not None:
        x1, y1 = int(bbox[0] * cam_w), int(bbox[1] * cam_h)
        x2, y2 = int(bbox[2] * cam_w), int(bbox[3] * cam_h)
        color = COLOR_SUCCESS_BGR if in_position else COLOR_WARN_BGR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def draw_pose_debug(frame, face, cam_w, cam_h, rvec, tvec, cam_mat, yaw, pitch):
    """
    Draws the 6 solvePnP landmark dots (cyan), a nose direction arrow (green),
    and a yaw/pitch readout at the bottom of the camera feed.
    """
    if rvec is None:
        return

    # 6 pose landmark dots in cyan so they're visually distinct
    for lm_id in HEAD_POSE_LM_IDS:
        lm = face.landmark[lm_id]
        cv2.circle(
            frame,
            (int(lm.x * cam_w), int(lm.y * cam_h)),
            4, (0, 255, 255), -1, lineType=cv2.LINE_AA
        )

    # Nose direction arrow — project nose tip and a point 500mm in front of it
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    nose_tip_3d = np.array([[0.0, 0.0,   0.0]], dtype=np.float64)
    nose_fwd_3d = np.array([[0.0, 0.0, 500.0]], dtype=np.float64)

    nose_2d, _ = cv2.projectPoints(nose_tip_3d, rvec, tvec, cam_mat, dist_coeffs)
    fwd_2d,  _ = cv2.projectPoints(nose_fwd_3d, rvec, tvec, cam_mat, dist_coeffs)

    p1 = (int(nose_2d[0][0][0]), int(nose_2d[0][0][1]))
    p2 = (int(fwd_2d[0][0][0]),  int(fwd_2d[0][0][1]))

    cv2.arrowedLine(frame, p1, p2, (0, 255, 0), 2, tipLength=0.3, line_type=cv2.LINE_AA)

    # Yaw / pitch readout at the bottom of the camera feed
    cv2.putText(
        frame,
        f"Yaw:{np.degrees(yaw):+.1f}  Pitch:{np.degrees(pitch):+.1f}",
        (20, cam_h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA
    )

def start_gaze_recording():
    global recording
    global record_start_time
    global gaze_recording

    recording = True
    record_start_time = time.time()
    gaze_recording = []

    print("Started gaze recording")

def save_gaze_data():

    filename = f"gaze_{int(time.time())}.json"

    filepath = os.path.join(
        GAZE_SESSION_DIR,
        filename
    )
    with open(filepath, "w") as f:
        json.dump({
            "screen_w": screen_w,
            "screen_h": screen_h,
            "points": gaze_recording
        }, f, indent=2)

    print(f"Saved {len(gaze_recording)} gaze points to {filepath}")

    return filepath


# ---------------- Main loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cam_h, cam_w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    gaze = None
    eye_openness = None   # Fix 5: computed per-frame for blink gating
    head_pose = (0.0, 0.0)   # (yaw, pitch); updated each frame when face is detected
    current_bbox = None      # normalized (x1,y1,x2,y2) face box, computed every frame

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        current_bbox = get_face_bbox(face)

        # Left eye geometry
        left_corner_a = face.landmark[33]
        left_corner_b = face.landmark[133]

        # Right eye geometry
        right_corner_a = face.landmark[362]
        right_corner_b = face.landmark[263]

        # Stable vertical anchors
        left_v_top = face.landmark[70]
        left_v_bottom = face.landmark[116]
        right_v_top = face.landmark[300]
        right_v_bottom = face.landmark[345]

        # Iris landmark clusters
        left_iris_ids = [468, 469, 470, 471, 472]
        right_iris_ids = [473, 474, 475, 476, 477]

        left_iris = avg_point(face, left_iris_ids)
        right_iris = avg_point(face, right_iris_ids)

        if debug_mode:
            for i in left_iris_ids:
                lm = face.landmark[i]
                draw_tiny_dot(frame, (int(lm.x * cam_w), int(lm.y * cam_h)), (0, 255, 0), 1)

            for i in right_iris_ids:
                lm = face.landmark[i]
                draw_tiny_dot(frame, (int(lm.x * cam_w), int(lm.y * cam_h)), (0, 255, 255), 1)

            lcx, lcy = int(left_iris[0] * cam_w), int(left_iris[1] * cam_h)
            rcx, rcy = int(right_iris[0] * cam_w), int(right_iris[1] * cam_h)
            draw_tiny_dot(frame, (lcx, lcy), (0, 200, 0), 2)
            draw_tiny_dot(frame, (rcx, rcy), (0, 200, 200), 2)

            for lm in [left_v_top, left_v_bottom, right_v_top, right_v_bottom]:
                draw_tiny_dot(frame, (int(lm.x * cam_w), int(lm.y * cam_h)), (255, 165, 0), 2)

        left_norm = normalize_eye(
            left_iris,
            (left_corner_a.x, left_corner_a.y),
            (left_corner_b.x, left_corner_b.y),
            (left_v_top.x, left_v_top.y),
            (left_v_bottom.x, left_v_bottom.y),
        )

        right_norm = normalize_eye(
            right_iris,
            (right_corner_a.x, right_corner_a.y),
            (right_corner_b.x, right_corner_b.y),
            (right_v_top.x, right_v_top.y),
            (right_v_bottom.x, right_v_bottom.y),
        )

        vals = []
        if left_norm is not None:
            vals.append(left_norm)
        if right_norm is not None:
            vals.append(right_norm)

        if vals:
            norm_x = sum(v[0] for v in vals) / len(vals)
            norm_y = sum(v[1] for v in vals) / len(vals)
            yaw, pitch, hp_rvec, hp_tvec, hp_cam_mat = get_head_pose(face, cam_w, cam_h)
            head_pose = (yaw, pitch)
            gaze = (norm_x, norm_y, yaw, pitch)
            if debug_mode:
                draw_pose_debug(frame, face, cam_w, cam_h, hp_rvec, hp_tvec, hp_cam_mat, yaw, pitch)

            # Fix 5: eye openness — average of both eyes
            left_open  = eye_openness_ratio(face, 159, 145, 33,  133)
            right_open = eye_openness_ratio(face, 386, 374, 362, 263)
            eye_openness = (left_open + right_open) / 2

            if debug_mode:
                debug_scale = 300
                debug_x = int(norm_x * debug_scale)
                debug_y = int(norm_y * debug_scale)
                debug_x = max(0, min(cam_w - 1, debug_x))
                debug_y = max(0, min(cam_h - 1, debug_y))
                cv2.circle(frame, (debug_x, debug_y), 6, (255, 0, 255), -1)

    # Head-position lock: update this frame's status against the reference box
    if head_ref_bbox is not None and current_bbox is not None:
        in_head_position, head_pos_dx, head_pos_dy, head_size_ratio, head_yaw_error, head_pitch_error = check_head_position(
            current_bbox,
            head_ref_bbox,
            head_ref_size,
            head_pose[0],
            head_pose[1],
            ref_yaw,
            ref_pitch
        )

    else:

        in_head_position = True
        head_pos_dx = 0.0
        head_pos_dy = 0.0
        head_size_ratio = 1.0
        head_yaw_error = 0.0
        head_pitch_error = 0.0
    if current_bbox is not None:
        draw_head_position_guide(frame, cam_w, cam_h, head_ref_bbox, current_bbox, in_head_position)

    # Fix 1 + Fix 5: accumulate frames for current calibration point
    eyes_ok = eye_openness is not None and eye_openness > BLINK_THRESHOLD
    if collecting and gaze is not None and eyes_ok:
        pending_gaze_samples.append(gaze)
        if current_bbox is not None:
            pending_bbox_samples.append(current_bbox)

    if collecting and len(pending_gaze_samples) >= COLLECT_FRAMES:
        avg_gx    = sum(g[0] for g in pending_gaze_samples) / len(pending_gaze_samples)
        avg_gy    = sum(g[1] for g in pending_gaze_samples) / len(pending_gaze_samples)
        avg_yaw   = sum(g[2] for g in pending_gaze_samples) / len(pending_gaze_samples)
        avg_pitch = sum(g[3] for g in pending_gaze_samples) / len(pending_gaze_samples)
        avg_gaze = (avg_gx, avg_gy, avg_yaw, avg_pitch)
        current_key = order[idx]
        samples.append((avg_gaze, target_pos(current_key)))
        print(
            f"Saved {current_key}: gaze=({avg_gx:.3f},{avg_gy:.3f}) "
            f"pose=({avg_yaw:.3f},{avg_pitch:.3f}) -> screen={target_pos(current_key)} "
            f"({COLLECT_FRAMES} frames averaged)"
        )

        # Lock in the head-position reference box from the first calibration
        # point, so the rest of calibration (and tracking) can be checked
        # against "where the user was sitting when this all worked."
        if head_ref_bbox is None and pending_bbox_samples:

            n = len(pending_bbox_samples)

            ref_x1 = sum(b[0] for b in pending_bbox_samples) / n
            ref_y1 = sum(b[1] for b in pending_bbox_samples) / n
            ref_x2 = sum(b[2] for b in pending_bbox_samples) / n
            ref_y2 = sum(b[3] for b in pending_bbox_samples) / n


            head_ref_bbox = (
                ref_x1,
                ref_y1,
                ref_x2,
                ref_y2
            )

            head_ref_size = max(
                ref_x2 - ref_x1,
                ref_y2 - ref_y1
            )


            ref_yaw = avg_yaw
            ref_pitch = avg_pitch


            print(
                f"Head locked. "
                f"Yaw:{np.degrees(ref_yaw):.1f} "
                f"Pitch:{np.degrees(ref_pitch):.1f}"
            )

        pending_gaze_samples = []
        pending_bbox_samples = []
        collecting = False
        calib_flash_pos = target_pos(current_key)
        calib_flash_until = time.time() + 0.45
        idx += 1
        if idx == len(order):
            calibration_done = True
            refit_model()
            print(f"Calibration complete with {len(samples)} samples.")

    # Fix 3: accumulate gaze throughout the correction session
    if correction_mode and gaze is not None and eyes_ok:
        correction_gaze_samples.append(gaze)

    # Fullscreen canvas
    canvas = np.full((screen_h, screen_w, 3), COLOR_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (screen_w, TOP_BAR_H), COLOR_BAR, -1)

    pulse_t = time.time()
    face_detected = results.multi_face_landmarks is not None
    show_camera = not splash_active and not session_countdown_active

    camera_x = 20
    camera_y = max(TOP_BAR_H + 10, screen_h - cam_h - 20)
    visible_w = 0
    visible_h = 0

    if show_camera:
        visible_w, visible_h, camera_x, camera_y = paste_camera_rounded(
            canvas, frame, camera_x, camera_y, face_detected,
        )

    ui_chips = None
    ui_chip_opacity = 1.0
    ui_title = None
    ui_body = None
    ui_title_centered = None
    ui_counter = None
    ui_distance = None
    image_progress = None

    if session_complete_active:
        if time.time() - session_complete_start >= SESSION_COMPLETE_HOLD:
            session_complete_active = False

    if session_complete_active:
        apply_dim_overlay(canvas)

    elif instructions_active:
        apply_dim_overlay(canvas)

    elif image_showing:
        img = get_display_image()
        crossfade_t = None
        if image_crossfade_active and image_crossfade_start is not None:
            crossfade_t = clamp(
                (time.time() - image_crossfade_start) / CROSSFADE_DURATION, 0.0, 1.0,
            )
            if crossfade_t >= 1.0:
                image_crossfade_active = False
                crossfade_prev_img = None
                crossfade_prev_rect = None
                crossfade_t = None
                image_start_time = time.time()

        draw_stimulus_image(
            canvas, img, crossfade_t, crossfade_prev_img, crossfade_prev_rect,
        )

        if image_start_time is not None and not image_crossfade_active:
            image_progress = 1.0 - clamp(
                (time.time() - image_start_time) / image_duration, 0.0, 1.0,
            )

        ui_counter = f"{current_image_index + 1} / {len(loaded_images)}"

        if (
            not image_crossfade_active
            and image_start_time is not None
            and time.time() - image_start_time >= image_duration
        ):
            advance_image()

    elif correction_mode:
        apply_dim_overlay(canvas)
        ui_title_centered = "Correction Mode"
        ui_body = ["Use arrow keys to move the target"]
        ui_chips = [("Enter", "Save"), ("Esc", "Cancel")]
        ui_chip_opacity = hint_opacity()

    elif idx < len(order):
        apply_dim_overlay(canvas)
        flash_active = (
            calib_flash_pos is not None
            and time.time() < calib_flash_until
        )
        if flash_active:
            x, y = calib_flash_pos
            cv2.circle(canvas, (x, y), 14, COLOR_SUCCESS_BGR, -1, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x - 7, y), (x - 2, y + 6), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (x - 2, y + 6), (x + 9, y - 7), (255, 255, 255), 2, cv2.LINE_AA)
        else:
            collect_progress = len(pending_gaze_samples) / COLLECT_FRAMES if collecting else 0
            draw_calib_target(
                canvas, order[idx], collecting, collect_progress, pulse_t, False,
            )
            draw_calib_progress_grid(canvas, idx, pulse_t)

    else:
        if not calibration_done:
            calibration_done = True
            refit_model()
            print("Calibration complete. Tracking active.")
        ui_title_centered = "Tracking Active"
        ui_chips = [
            ("I", "Import Images"),
            ("S", "Start Session"),
            ("M", "Correct"),
            ("R", "Reset"),
        ]
        ui_chip_opacity = hint_opacity()

    # Map gaze to screen
    if calibration_done and gaze is not None and model is not None:
        gx, gy, raw_yaw, raw_pitch = gaze
        clamped_yaw   = clamp(raw_yaw,   calib_yaw_min,   calib_yaw_max)
        clamped_pitch = clamp(raw_pitch, calib_pitch_min, calib_pitch_max)
        pred = predict_screen(gx, gy, clamped_yaw, clamped_pitch, model)
        if pred is not None:
            pred_x, pred_y = pred

            alpha = 0.45
            if smoothed_x is None:
                smoothed_x = pred_x
                smoothed_y = pred_y
            else:
                smoothed_x = smoothed_x * (1 - alpha) + pred_x * alpha
                smoothed_y = smoothed_y * (1 - alpha) + pred_y * alpha

            smoothed_x = max(0, min(screen_w - 1, smoothed_x))
            smoothed_y = max(0, min(screen_h - 1, smoothed_y))
            update_gaze_trail(smoothed_x, smoothed_y)

            if correction_mode and correction_target is not None:
                draw_correction_guides(
                    canvas,
                    (smoothed_x, smoothed_y),
                    correction_target,
                )
                dist = math.hypot(
                    correction_target[0] - smoothed_x,
                    correction_target[1] - smoothed_y,
                )
                ui_distance = f"{dist:.0f} px"
            elif (
                not image_showing
                and not splash_active
                and not session_countdown_active
                and idx >= len(order)
                and not correction_mode
            ):
                draw_gaze_cursor(canvas, int(smoothed_x), int(smoothed_y))

    # apply_vignette(canvas)

    if (
        head_ref_bbox is not None
        and not in_head_position
        and not head_lock_override
        and not splash_active
        and not instructions_active
        and not session_countdown_active
    ):
        cv2.rectangle(canvas, (0, 0), (screen_w - 1, screen_h - 1), COLOR_WARN_BGR, 6)

    if head_lock_override:
        ui_override_note = "Position lock overridden (Backspace)"
    else:
        ui_override_note = None

    ui = TextRenderer(canvas)
    draw_top_bar(ui)

    if ui_override_note:
        ui.text(
            (16, TOP_BAR_H + 8),
            ui_override_note,
            size=FONT_SIZE_SMALL,
            fill=COLOR_ACCENT_RGB,
        )

    if splash_active:
        draw_splash(ui)
    elif instructions_active:
        draw_instructions(ui)
    elif session_countdown_active:
        if draw_countdown(ui):
            begin_image_session_after_countdown()
    elif session_complete_active:
        draw_session_complete(ui)
    else:
        if ui_title_centered:
            ui.text_centered(ui_title_centered, TOP_BAR_H + 24, size=FONT_SIZE_TITLE)

        if ui_title:
            ui.text_box([ui_title], y=TOP_BAR_H + 20, size=FONT_SIZE_TITLE)

        if ui_body:
            ui.text_box(ui_body, y=TOP_BAR_H + 72, size=FONT_SIZE_BODY)

        if ui_counter:
            ui.text_right(ui_counter, y=TOP_BAR_H + 8, size=FONT_SIZE_SMALL, fill=COLOR_TEXT_DIM, margin=24)

        if ui_distance:
            ui.text_centered(ui_distance, TOP_BAR_H + 60, size=FONT_SIZE_BODY, fill=COLOR_TEXT_MUTED)

        if ui_chips:
            ui.key_chip_row(ui_chips, screen_h - 72, opacity=ui_chip_opacity)

        if idx < len(order) and not (
            calib_flash_pos is not None and time.time() < calib_flash_until
        ):
            tx, ty = target_pos(order[idx])
            ui.key_chip(tx - 65, ty + 52, "C", "Save point")

        if show_camera:
            ui.text(
                (camera_x, camera_y - 22),
                "CAMERA",
                size=FONT_SIZE_SMALL,
                fill=COLOR_TEXT_DIM,
            )
            if not face_detected:
                badge = "⚠ No face detected"
                font = get_font(FONT_SIZE_SMALL)
                bb = ui.draw.textbbox((0, 0), badge, font=font)
                bw = bb[2] - bb[0]
                ui.text(
                    (camera_x + (visible_w - bw) // 2, camera_y + visible_h // 2 - 10),
                    badge,
                    size=FONT_SIZE_SMALL,
                    fill=COLOR_WARN_RGB,
                )
            elif head_ref_bbox is not None and in_head_position:
                ui.text(
                    (camera_x, camera_y + visible_h + 8),
                    "In position \u2014 hold still",
                    size=FONT_SIZE_SMALL,
                    fill=COLOR_TEXT_DIM,
                )
            elif head_ref_bbox is not None and not in_head_position:
                pos_parts = []
                if head_pos_dx > POS_TOLERANCE:
                    pos_parts.append("move left")
                elif head_pos_dx < -POS_TOLERANCE:
                    pos_parts.append("move right")
                if head_pos_dy > POS_TOLERANCE:
                    pos_parts.append("move up")
                elif head_pos_dy < -POS_TOLERANCE:
                    pos_parts.append("move down")
                if head_size_ratio > 1.0 + SIZE_TOLERANCE:
                    pos_parts.append("move back")
                elif head_size_ratio < 1.0 - SIZE_TOLERANCE:
                    pos_parts.append("move closer")

                rot_parts = []
                if head_yaw_error > YAW_TOLERANCE:
                    rot_parts.append("turn head left")
                elif head_yaw_error < -YAW_TOLERANCE:
                    rot_parts.append("turn head right")
                if head_pitch_error > PITCH_TOLERANCE:
                    rot_parts.append("tilt head down")
                elif head_pitch_error < -PITCH_TOLERANCE:
                    rot_parts.append("tilt head up")

                msg_y = camera_y + visible_h + 8
                if pos_parts:
                    ui.text(
                        (camera_x, msg_y),
                        "Position: " + ", ".join(pos_parts),
                        size=FONT_SIZE_SMALL,
                        fill=COLOR_WARN_RGB,
                    )
                    msg_y += 24
                if rot_parts:
                    ui.text(
                        (camera_x, msg_y),
                        "Angle: " + ", ".join(rot_parts),
                        size=FONT_SIZE_SMALL,
                        fill=COLOR_ACCENT_RGB,
                    )
                if not pos_parts and not rot_parts:
                    ui.text(
                        (camera_x, msg_y),
                        "Return to the box",
                        size=FONT_SIZE_SMALL,
                        fill=COLOR_WARN_RGB,
                    )

    if debug_mode:
        if results.multi_face_landmarks and gaze is not None:
            norm_x, norm_y = gaze[0], gaze[1]
            ui.text(
                (camera_x, camera_y - 28),
                f"RAW: ({norm_x:.3f}, {norm_y:.3f})",
                size=FONT_SIZE_BODY,
                fill=COLOR_ACCENT_RGB,
            )
            yaw, pitch = head_pose
            ui.text(
                (camera_x, camera_y + visible_h + 8),
                f"Yaw: {np.degrees(yaw):+.1f}  Pitch: {np.degrees(pitch):+.1f}",
                size=FONT_SIZE_BODY,
                fill=COLOR_ACCENT_RGB,
            )

        if smoothed_x is not None and smoothed_y is not None and gaze is not None:
            raw_gx, raw_gy = gaze[0], gaze[1]
            ui.text(
                (40, screen_h - 120),
                f"Looking at: ({int(smoothed_x)}, {int(smoothed_y)})",
                size=FONT_SIZE_BODY,
            )
            ui.text(
                (40, screen_h - 88),
                f"Raw gaze: ({raw_gx:.3f}, {raw_gy:.3f})",
                size=FONT_SIZE_BODY,
                fill=COLOR_TEXT_DIM,
            )

    ui.flush()

    if image_showing and image_progress is not None:
        draw_progress_bar(canvas, 0, screen_h - 4, screen_w, 4, image_progress)

    if recording:
        session_time = time.time() - record_start_time

        if smoothed_x is not None and smoothed_y is not None:
            gaze_recording.append({
                "time": round(session_time, 3),
                "x": int(smoothed_x),
                "y": int(smoothed_y),
                "image_id": current_image_id
            })

        if not image_showing and session_time >= record_duration:
            recording = False
            save_gaze_data()

    cv2.imshow(window_name, canvas)


# ---------------- Key handling ----------------
    key = cv2.waitKeyEx(1)

    if key != -1:
        last_key_time = time.time()

    if splash_active and key != -1 and (key & 0xFF) != ord('q'):
        splash_active = False
        instructions_active = True

    elif instructions_active and key != -1 and (key & 0xFF) != ord('q'):
        instructions_active = False

    # Initial 9-point calibration — C starts a collection window
    if (key & 0xFF) == ord('c') and gaze is not None and idx < len(order) and not correction_mode and not collecting:
        if head_ref_bbox is not None and not in_head_position and not head_lock_override:
            print("Move back into the highlighted box before collecting this point.")
        else:
            collecting = True
            pending_gaze_samples = []
            pending_bbox_samples = []
            print(f"Collecting frames for point {order[idx]}...")

    # Start continuous correction mode
    if (
        calibration_done
        and not correction_mode
        and (key & 0xFF) == ord('m')
        and gaze is not None
        and model is not None
    ):

        correction_mode = True
        correction_raw_gaze = gaze
        correction_gaze_samples = []   # Fix 3: start fresh accumulation

        predicted = predict_screen(*gaze, model)

        if predicted is None:
            predicted = (screen_w / 2, screen_h / 2)

        correction_target = [
            int(predicted[0]),
            int(predicted[1])
        ]

        print("Correction mode started.")


    # Import images
    if (key & 0xFF) == ord('i'):
        import_images()

    # Toggle debug overlays
    if (key & 0xFF) == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'on' if debug_mode else 'off'}")

    # Start image recording session
    if (key & 0xFF) == ord('s'):
        if session_complete_active:
            session_complete_active = False
            start_image_session()
        elif not image_showing and not session_countdown_active:
            start_image_session()

    # ---------------- Correction mode ----------------
    if correction_mode and correction_target is not None:

        MOVE_SPEED = 10

        # Arrow movement
        if is_arrow_left(key):
            correction_target[0] -= MOVE_SPEED

        elif is_arrow_right(key):
            correction_target[0] += MOVE_SPEED

        elif is_arrow_up(key):
            correction_target[1] -= MOVE_SPEED

        elif is_arrow_down(key):
            correction_target[1] += MOVE_SPEED

        # Keep target on screen
        correction_target[0] = clamp(
            correction_target[0],
            0,
            screen_w - 1
        )

        correction_target[1] = clamp(
            correction_target[1],
            0,
            screen_h - 1
        )

        # Save corrected calibration sample
        if key == 13 or key == 10:

            # Fix 3: use the average of all gaze frames collected during the
            # correction session rather than the single snapshot taken at M press
            if correction_gaze_samples:
                avg_gx    = sum(g[0] for g in correction_gaze_samples) / len(correction_gaze_samples)
                avg_gy    = sum(g[1] for g in correction_gaze_samples) / len(correction_gaze_samples)
                avg_yaw   = sum(g[2] for g in correction_gaze_samples) / len(correction_gaze_samples)
                avg_pitch = sum(g[3] for g in correction_gaze_samples) / len(correction_gaze_samples)
                saved_gaze = (avg_gx, avg_gy, avg_yaw, avg_pitch)
            else:
                saved_gaze = correction_raw_gaze  # fallback if no frames accumulated

            samples.append(
                (
                    saved_gaze,
                    (
                        correction_target[0],
                        correction_target[1]
                    )
                )
            )

            frame_count = len(correction_gaze_samples) if correction_gaze_samples else 1

            refit_model()

            correction_mode = False
            correction_raw_gaze = None
            correction_target = None
            correction_gaze_samples = []

            smoothed_x = None
            smoothed_y = None

            print(
                f"Saved correction sample ({frame_count} frames averaged). "
                f"Total samples: {len(samples)}"
            )

        # Cancel correction mode
        elif (key & 0xFF) == 27:

            correction_mode = False
            correction_raw_gaze = None
            correction_target = None
            correction_gaze_samples = []

            print("Correction cancelled.")

    # Backspace toggles the head-position lock override, so a presenter can
    # keep going in a live demo even if the red box isn't cooperating.
    if (key & 0xFF) == 8:
        head_lock_override = not head_lock_override
        print(f"Head position lock override: {'ON' if head_lock_override else 'OFF'}")

    # Reset calibration
    if (key & 0xFF) == ord('r') and not correction_mode:
        if session_complete_active:
            session_complete_active = False
        session_countdown_active = False
        session_countdown_start = None
        image_crossfade_active = False
        crossfade_prev_img = None
        gaze_trail.clear()
        calib_flash_until = 0.0
        calib_flash_pos = None
        samples.clear()
        idx = 0
        calibration_done  = False
        model             = None
        smoothed_x        = None
        smoothed_y        = None
        correction_mode   = False
        correction_raw_gaze   = None
        correction_target     = None
        correction_gaze_samples = []
        collecting            = False
        pending_gaze_samples  = []
        head_ref_bbox         = None
        head_ref_size         = None
        pending_bbox_samples  = []
        in_head_position      = True
        print("Calibration reset. Starting over.")

    # Quit
    if (key & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()