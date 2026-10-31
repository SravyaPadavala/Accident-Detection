"""Flask dashboard for the Real-Time Road Accident Detection System.
Run:  python app.py    then open the address printed in the terminal.
"""
import itertools
import os
import time
import cv2
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename
import database
from detector.pipeline import AccidentPipeline
from generate_sample_video import scene_frames, sample_path, write_video, FPS

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
ALLOWED = {".mp4", ".avi", ".mov", ".mkv"}
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
STATE = {"status": "idle", "vehicles": 0, "probability": 0.0, "classifier": None}


def _jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


def _stream(frames, pipe, fps, wall_clock=False):
    """Run `frames` through the pipeline and yield an MJPEG stream."""
    STATE.update(status="running", classifier=pipe.classifier_available)
    t0 = time.time()
    try:
        for i, frame in enumerate(frames):
            start = time.time()
            now = (time.time() - t0) if wall_clock else i / fps
            out = pipe.process_frame(frame, now)
            STATE.update(vehicles=pipe.vehicles, probability=round(pipe.prob, 3))
            yield _jpeg(out)
            time.sleep(max(0.0, 1.0 / fps - (time.time() - start)))   # keep ~real-time speed
    finally:
        STATE.update(status="idle", vehicles=0, probability=0.0)


def _video_frames(path):
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="No video file selected"), 400
    name = secure_filename(f.filename)
    if os.path.splitext(name)[1].lower() not in ALLOWED:
        return jsonify(error="Unsupported file type (use MP4, AVI, MOV or MKV)"), 400
    f.save(os.path.join(UPLOAD_DIR, name))
    return jsonify(filename=name)


@app.route("/sample/<kind>", methods=["POST"])
def sample(kind):
    if kind not in ("safe", "accident"):
        return jsonify(error="kind must be 'safe' or 'accident'"), 400
    p = sample_path(kind)
    if not os.path.exists(p):
        write_video(kind, p)
    return jsonify(filename=f"sample_{kind}_traffic.mp4", sample=True)


@app.route("/stream/<filename>")
def stream(filename):
    detector = request.args.get("detector", "yolo")
    sample_dir = os.path.dirname(sample_path("safe"))
    path = os.path.join(sample_dir if filename.startswith("sample_") else UPLOAD_DIR, secure_filename(filename))
    if not os.path.exists(path):
        return jsonify(error="Video not found"), 404
    try:
        pipe = AccidentPipeline(detector=detector, source=f"video:{filename}")
    except Exception as e:                                   # e.g. ultralytics not installed
        return jsonify(error=f"Could not start detector: {e}"), 500
    fps = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS) or FPS
    return Response(_stream(_video_frames(path), pipe, fps),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live")
def live():
    detector = request.args.get("detector", "yolo")
    try:
        pipe = AccidentPipeline(detector=detector, source="live-camera")
    except Exception as e:
        return jsonify(error=f"Could not start detector: {e}"), 500
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        def cam():
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield frame
            finally:
                cap.release()
        frames = cam()
    else:                       # simulated fallback when no webcam exists
        pipe.source, pipe.detector = "simulated-camera", AccidentPipeline(detector="blob").detector
        frames = (f for gen in itertools.cycle(["safe", "accident"]) for f in scene_frames(gen))
    return Response(_stream(frames, pipe, FPS, wall_clock=True),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
def api_stats():
    return jsonify({**database.get_stats(), **STATE})


@app.route("/api/incidents")
def api_incidents():
    return jsonify(database.list_incidents())


@app.route("/api/incidents/<int:incident_id>", methods=["DELETE"])
def api_delete(incident_id):
    return jsonify(deleted=database.delete_incident(incident_id))


if __name__ == "__main__":
    database.init_db()
    app.run(debug=False, threaded=True)
