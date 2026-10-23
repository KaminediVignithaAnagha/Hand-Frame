"""
hand_tracker.py  (MediaPipe Tasks version -- works on new Python/MediaPipe)
---------------------------------------------------------------------------
Same job as before: find your hand and give back 21 (x, y) pixel points.
The only change is HOW: newer MediaPipe removed `mp.solutions`, so we use
the "Tasks" API, which needs a small model file (hand_landmarker.task).
The file downloads automatically the first time you run the program.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

WRIST = 0
FINGERTIPS = [4, 8, 12, 16, 20]


def ensure_model():
    """Download the model file once, if it isn't already next to this script."""
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand model (one time only, ~8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "Could not download the model automatically.\n"
            "Run this in Terminal from the project folder instead:\n"
            f"  curl -L -o hand_landmarker.task {MODEL_URL}\n"
            f"(Original error: {e})"
        )


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._start = time.monotonic()
        self._last_ts = -1

    def process(self, frame_bgr):
        """
        Returns a list of hands; each hand is a list of 21 (x_px, y_px) tuples.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))

        # VIDEO mode needs timestamps that always increase
        ts = int((time.monotonic() - self._start) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self.landmarker.detect_for_video(mp_image, ts)

        h, w = frame_bgr.shape[:2]
        all_hands = []
        for hand in result.hand_landmarks:
            all_hands.append([(int(lm.x * w), int(lm.y * h)) for lm in hand])
        return all_hands

    def draw(self, frame_bgr, points):
        """Draws small circles on each landmark point, for debugging."""
        for (x, y) in points:
            cv2.circle(frame_bgr, (x, y), 5, (0, 255, 0), cv2.FILLED)
        return frame_bgr

    @staticmethod
    def openness_score(points):
        wrist = np.array(points[WRIST])
        return float(np.mean([np.linalg.norm(np.array(points[t]) - wrist) for t in FINGERTIPS]))
