"""Creates two DEMO videos (drawn rectangles, not real footage):
    sample_videos/sample_safe_traffic.mp4
    sample_videos/sample_accident_traffic.mp4
Run:  python generate_sample_video.py
"""
import os
import cv2
import numpy as np

W, H, FPS, N_FRAMES = 640, 360, 30, 150
CAR_W, CAR_H = 80, 40
RED, BLUE, YELLOW, GREEN = (0, 0, 220), (220, 60, 0), (0, 220, 220), (0, 170, 0)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_videos")


def _road():
    img = np.full((H, W, 3), 90, np.uint8)
    for y in (120, 240):
        for x in range(0, W, 50):
            cv2.line(img, (x, y), (x + 25, y), (255, 255, 255), 2)
    return img


def _draw_car(img, x, y, color):
    x, y = int(x), int(y)
    cv2.rectangle(img, (x, y), (x + CAR_W, y + CAR_H), color, -1)
    cv2.rectangle(img, (x + 22, y + 8), (x + 58, y + 32), (60, 60, 60), -1)   # windows
    for wx in (x + 8, x + CAR_W - 18):
        cv2.rectangle(img, (wx, y - 3), (wx + 10, y), (0, 0, 0), -1)
        cv2.rectangle(img, (wx, y + CAR_H), (wx + 10, y + CAR_H + 3), (0, 0, 0), -1)


def scene_frames(kind="safe", n_frames=N_FRAMES):
    """Yield BGR frames for 'safe' or 'accident' traffic."""
    if kind == "safe":
        cars = [[10, 40, 5.0, RED], [540, 160, -4.0, BLUE], [10, 280, 3.0, YELLOW]]
    else:   # head-on collision in the middle lane + normal traffic in the bottom lane
        cars = [[10, 160, 5.0, RED], [550, 160, -5.0, BLUE], [10, 280, 3.0, GREEN]]
    for f in range(n_frames):
        img = _road()
        for c in cars:
            _draw_car(img, c[0], c[1], c[3])
        yield img
        if kind == "accident":
            red, blue = cars[0], cars[1]
            if red[0] + CAR_W > blue[0]:          # contact -> both vehicles stop abruptly
                red[2] *= 0.15; blue[2] *= 0.15
        for c in cars:
            c[0] += c[2]
            if kind == "safe" and (c[0] > W or c[0] < -CAR_W):   # wrap around
                c[0] = -CAR_W if c[2] > 0 else W


def write_video(kind, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for frame in scene_frames(kind):
        vw.write(frame)
    vw.release()
    return path


def sample_path(kind):
    return os.path.join(OUT_DIR, f"sample_{kind}_traffic.mp4")


if __name__ == "__main__":
    for k in ("safe", "accident"):
        print("created", write_video(k, sample_path(k)))
