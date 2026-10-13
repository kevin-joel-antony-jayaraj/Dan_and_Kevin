"""Detect a red object with a webcam and print "fire" when one is found.

Setup (on the Raspberry Pi):
    sudo apt install python3-opencv     # or: pip install opencv-python numpy

Run:
    python3 detect_red.py              # headless (no window), fine over SSH
    python3 detect_red.py --show       # show camera view + mask for tuning
Press Ctrl+C (or 'q' in the preview window) to quit.
"""

import argparse

import cv2
import numpy as np

# Red wraps around the ends of the HSV hue range (0-179 in OpenCV),
# so we need two ranges and combine them.
RED_LOW_1 = np.array([0, 120, 70])
RED_HIGH_1 = np.array([10, 255, 255])
RED_LOW_2 = np.array([170, 120, 70])
RED_HIGH_2 = np.array([179, 255, 255])

# Ignore tiny red specks (noise). Tune for your camera/distance.
MIN_AREA = 1500


def find_red(frame):
    """Return (mask, contour) for the largest red blob, contour is None if no red."""
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1) | cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_AREA:
        return mask, None
    return mask, biggest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--show", action="store_true", help="show preview windows")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break

            mask, contour = find_red(frame)

            if contour is not None:
                print("fire")

            if args.show:
                if contour is not None:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("camera", frame)
                cv2.imshow("red mask", mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
