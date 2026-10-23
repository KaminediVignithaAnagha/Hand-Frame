"""
main.py
--------
Hand-Frame Filter.

Make a rectangle with BOTH hands (thumb + index finger on each hand, like
a photographer's frame). The 4 fingertips are the 4 corners.

  - Fingers together / small rectangle  -> nothing happens (normal camera)
  - Open wide / big rectangle           -> the window appears: inside it you
                                           stay clear and your uploaded
                                           pictures (with filters) show
                                           behind you. No border line.

Controls:
  q  quit
  n  next picture
  f  next filter
  a  turn auto picture-changing on/off
  d  show/hide the helper dots + picture label (hide them for a clean look)
  m  debug: show the person cut-out mask (white = you, black = background)
"""

import cv2
import numpy as np

from hand_tracker import HandTracker
from segmenter import Segmenter
from effects import BackgroundManager, render_window

THUMB_TIP = 4
INDEX_TIP = 8

MIN_SIDE = 50      # below this size the window is fully hidden
FULL_SIDE = 130    # above this size the window is fully visible
HOLD_FRAMES = 6    # keep the rectangle this many frames if a hand blips out


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check camera permission / index.")
        return

    frame_w, frame_h = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    tracker = HandTracker(max_hands=2)
    segmenter = Segmenter()
    backgrounds = BackgroundManager(frame_w, frame_h)

    smooth_rect = None   # [x1, y1, x2, y2] as floats
    missing = 0
    show_helpers = True
    show_mask = False

    print("Controls: q quit | n next picture | f next filter | a auto-change | d dots/label | m mask debug")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        # --- 1. find the 4 fingertips (thumb + index on each of 2 hands) ---
        hands = tracker.process(frame)
        corner_points = []
        if len(hands) >= 2:
            for hand in hands[:2]:
                corner_points.append(hand[THUMB_TIP])
                corner_points.append(hand[INDEX_TIP])

        if corner_points:
            xs = [p[0] for p in corner_points]
            ys = [p[1] for p in corner_points]
            target = np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)
            smooth_rect = target if smooth_rect is None else 0.5 * smooth_rect + 0.5 * target
            missing = 0
        else:
            missing += 1
            if missing > HOLD_FRAMES:
                smooth_rect = None

        # --- 2. how "open" is the rectangle? (0 = closed, 1 = fully open) ---
        output = frame.copy()
        alpha = 0.0
        if smooth_rect is not None:
            side = min(smooth_rect[2] - smooth_rect[0], smooth_rect[3] - smooth_rect[1])
            alpha = float(np.clip((side - MIN_SIDE) / (FULL_SIDE - MIN_SIDE), 0.0, 1.0))

        # --- 3. draw the window effect only when it's open (no border line) ---
        if alpha > 0:
            backgrounds.update()
            person_mask = segmenter.get_mask(frame)
            rect = tuple(int(v) for v in smooth_rect)
            if show_mask:
                gray = (person_mask * 255).astype(np.uint8)
                output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                output = render_window(frame, backgrounds.current(), person_mask, rect, alpha)
            if show_helpers:
                cv2.putText(output, backgrounds.label(), (10, frame_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # helper dots on the 4 fingertips
        if show_helpers:
            for (x, y) in corner_points:
                cv2.circle(output, (x, y), 7, (0, 255, 0), cv2.FILLED)
            if len(hands) < 2:
                cv2.putText(output, "Show BOTH hands: thumb + index finger on each",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Hand-Frame Filter", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            backgrounds.next_picture()
        elif key == ord('f'):
            backgrounds.next_filter()
        elif key == ord('a'):
            backgrounds.auto = not backgrounds.auto
        elif key == ord('d'):
            show_helpers = not show_helpers
        elif key == ord('m'):
            show_mask = not show_mask

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()