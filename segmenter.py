"""
segmenter.py
-------------
Finds "you" (the person) in each webcam frame.

ELI5: a small AI model looks at the frame and, for every pixel, says how
sure it is that the pixel is background / hair / skin / clothes. We take
"1 minus the background score" = "how sure it's YOU". That's our person
mask: 1.0 = you, 0.0 = not you.

The model file downloads automatically the first time (about 16 MB).
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
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "selfie_multiclass_256x256.tflite")


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading person-segmentation model (one time only, ~16 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "Could not download the model automatically.\n"
            "Run this in Terminal from the project folder instead:\n"
            f"  curl -L -o selfie_multiclass_256x256.tflite {MODEL_URL}\n"
            f"(Original error: {e})"
        )


class Segmenter:
    def __init__(self):
        ensure_model()
        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)
        self._start = time.monotonic()
        self._last_ts = -1
        self._prev_mask = None

    def get_mask(self, frame_bgr):
        """Returns a float mask (H x W), 1.0 = you, 0.0 = background."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))

        ts = int((time.monotonic() - self._start) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self.segmenter.segment_for_video(mp_image, ts)

        # confidence_masks[0] = "background". Person = 1 - background.
        background = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32).copy()
        person = 1.0 - background

        if person.shape != (h, w):
            person = cv2.resize(person, (w, h))

        # Firm up the edges a little, then soften them so the cutout looks clean
        person = np.clip((person - 0.3) / 0.4, 0.0, 1.0)
        person = cv2.GaussianBlur(person, (7, 7), 0)

        # Blend with the previous frame to reduce flicker
        if self._prev_mask is not None and self._prev_mask.shape == person.shape:
            person = 0.6 * person + 0.4 * self._prev_mask
        self._prev_mask = person
        return person
