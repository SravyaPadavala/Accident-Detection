# Real-Time Road Accident Detection System (Deep Learning + Computer Vision)

Pipeline: **video -> YOLOv8 vehicle detection -> centroid tracking -> motion features
(speed, deceleration, IoU, distance) -> 16-frame window -> BiGRU + temporal attention
(PyTorch) -> P(accident) -> alert + evidence frame + SQLite record -> Flask dashboard.**

## 1. Setup
```
python -m venv venv
venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Linux/Mac)
pip install -r requirements.txt
```

## 2. Create demo videos and train the classifier
```
python generate_sample_video.py        # sample_videos/sample_safe_traffic.mp4 + sample_accident_traffic.mp4
python train_accident_detector.py      # trains on 1,600 synthetic sequences, saves models/accident_bigru.pt
```
The training script prints validation accuracy / precision / recall / F1 and the confusion
matrix for every epoch. **Copy the numbers you actually get into your report** (section 16).

## 3. Run the app
```
python app.py
```
Open the address printed in the terminal (usually http://127.0.0.1:5000).

* **Demo: safe / accident** buttons use the drawn demo videos with the *Colour blobs* detector.
* **Upload & analyse** a real traffic video with the *YOLOv8* detector (first run downloads `yolov8n.pt`).
* **Live camera** uses the webcam; if none exists it plays a simulated feed.
* Incidents appear in the history table with evidence image, and a beep + red banner is shown.

## 4. Run tests
```
python test_app.py
```

## Files
| File | Purpose |
|---|---|
| app.py | Flask routes, video/live streaming, JSON API |
| database.py | SQLite incident storage |
| detector/yolo_vehicle.py | YOLOv8 wrapper (+ colour-blob detector for demo videos) |
| detector/tracker.py | Centroid tracker, IoU, velocity/deceleration features |
| detector/accident_model.py | BiGRU + attention model, checkpoint loading |
| detector/synthetic.py | Synthetic training-sequence generator |
| detector/pipeline.py | Ties everything together, alerts, evidence capture |
| train_accident_detector.py | Training script |
| generate_sample_video.py | Demo video generator |
| test_app.py | Automated tests |

## Important notes
* The classifier is trained on **synthetic** sequences. A high validation score on synthetic data
  is NOT real-world accuracy. For real use, fine-tune/test on labelled accident videos (CCD, CADP, Kaggle).
* Threshold (0.65) and cooldown (4 s) are in `detector/pipeline.py`.
* The demo videos are drawn rectangles, not real footage - YOLO cannot detect them, hence the blob detector.
* Alerts are model estimates and must be reviewed by a human. The app does not contact emergency services.
