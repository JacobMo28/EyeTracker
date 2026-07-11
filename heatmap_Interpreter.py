import json
import cv2
import numpy as np
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog
import os


# ============================================================
# FILE SELECTION
# ============================================================

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select gaze session JSON",
    filetypes=[("JSON files", "*.json")]
)
if not file_path:
    print("No file selected, exiting.")
    exit()

image_folder = filedialog.askdirectory(title="Select folder containing session images")
if not image_folder:
    print("No image folder selected, exiting.")
    exit()


# ============================================================
# LOAD JSON
# ============================================================

with open(file_path, 'r') as f:
    raw = json.load(f)

# Support both old format (plain list) and new format (object with metadata)
if isinstance(raw, list):
    data = raw
    SCREEN_W = int(input("Old format — enter screen width used during recording: "))
    SCREEN_H = int(input("Old format — enter screen height used during recording: "))
else:
    data = raw["points"]
    SCREEN_W = raw["screen_w"]
    SCREEN_H = raw["screen_h"]
    print(f"Screen resolution from file: {SCREEN_W}x{SCREEN_H}")


# ============================================================
# SETTINGS
# ============================================================

BLUR_KERNEL   = 71   # larger = wider/softer hotspots (must be odd)
HEATMAP_ALPHA = 0.6  # heatmap opacity over background image

LINGER_MARGIN      = 30  # px radius for gaze linger detection
LINGER_COUNTER_MAX = 6   # frames before a point is considered a fixation
LINGER_MAX_POINTS  = 25  # max fixation points to draw on gaze map

FIXATION_RADIUS    = 18  # circle size for fixation dots
LINE_COLOR         = (255, 255, 255)
LINE_THICKNESS     = 2


# ============================================================
# HELPERS
# ============================================================

def is_edge_clamped(x, y):
    return x <= 0 or y <= 0 or x >= SCREEN_W - 1 or y >= SCREEN_H - 1


def load_background(img_ID):
    """Load and center the background image the same way main.py displays it."""
    img_path = os.path.join(image_folder, img_ID)
    bg = cv2.imread(img_path)

    if bg is None:
        print(f"Warning: could not load {img_path}, using dark background")
        return np.ones((SCREEN_H, SCREEN_W, 3), dtype=np.uint8) * 30

    h, w = bg.shape[:2]
    scale = min(SCREEN_W / w, SCREEN_H / h) * 0.8
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(bg, (new_w, new_h))

    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    x_off = (SCREEN_W - new_w) // 2
    y_off = (SCREEN_H - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = img_resized

    return canvas


def save_output(img, name):
    out_name = f"{name}_{img_ID.replace(' ', '_')}.png"
    cv2.imwrite(out_name, img)
    print(f"Saved: {out_name}")


# ============================================================
# HEATMAP
# ============================================================

def generate_heatmap(points, img_ID):
    accumulator = np.zeros((SCREEN_H, SCREEN_W), dtype=np.float32)
    for (x, y) in points:
        accumulator[y, x] += 1

    blurred      = cv2.GaussianBlur(accumulator, (BLUR_KERNEL, BLUR_KERNEL), 0)
    normalized   = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
    colored      = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)

    bg      = load_background(img_ID)
    overlay = cv2.addWeighted(colored, HEATMAP_ALPHA, bg, 1 - HEATMAP_ALPHA, 0)

    save_output(overlay, "heatmap")


# ============================================================
# GAZE MAP (fixation points + path lines)
# ============================================================

def find_fixations(points):
    """Return a list of (x, y) fixation points from raw gaze data."""
    fixations = []
    counter   = 0
    prev_x, prev_y = None, None

    for (x, y) in points:
        if prev_x is None:
            prev_x, prev_y = x, y
            continue

        dif_x = abs(prev_x - x)
        dif_y = abs(prev_y - y)

        if dif_x < LINGER_MARGIN and dif_y < LINGER_MARGIN:
            counter += 1
        else:
            counter = 0

        if counter >= LINGER_COUNTER_MAX:
            fixations.append((x, y))
            counter = 0

        prev_x, prev_y = x, y

    return fixations[:LINGER_MAX_POINTS]


def generate_gaze_map(points, img_ID):
    fixations = find_fixations(points)

    if not fixations:
        print(f"No fixations found for {img_ID}")
        return

    bg = load_background(img_ID)

    # Draw lines between fixations
    for i in range(1, len(fixations)):
        cv2.line(bg, fixations[i - 1], fixations[i], LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)

    # Draw fixation circles with numbered labels
    for i, (x, y) in enumerate(fixations):
        # Color fades from green (first) to red (last)
        t      = i / max(len(fixations) - 1, 1)
        color  = (0, int(255 * (1 - t)), int(255 * t))  # BGR: green -> red

        cv2.circle(bg, (x, y), FIXATION_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(bg, (x, y), FIXATION_RADIUS, (255, 255, 255), 1, cv2.LINE_AA)

        label    = str(i + 1)
        font     = cv2.FONT_HERSHEY_SIMPLEX
        scale    = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        cv2.putText(bg, label, (x - tw // 2, y + th // 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    save_output(bg, "gazemap")


# ============================================================
# MAIN
# ============================================================

points_by_image = defaultdict(list)
for point in data:
    img_ID = point['image_id']
    x, y   = point['x'], point['y']
    if is_edge_clamped(x, y):
        continue
    if img_ID:
        points_by_image[img_ID].append((x, y))

for img_ID, points in points_by_image.items():
    print(f"\nProcessing: {img_ID} ({len(points)} points)")
    generate_heatmap(points, img_ID)
    generate_gaze_map(points, img_ID)