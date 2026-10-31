"""Automated tests. Run:  python test_app.py
The classifier test trains a small model first if no checkpoint exists."""
import os
import tempfile
import unittest
import numpy as np

_tmp = tempfile.mkdtemp()
os.environ["ACCIDENT_DB"] = os.path.join(_tmp, "test.db")
import database
from detector.tracker import CentroidTracker, frame_features, iou, SEQ_LEN, NUM_FEATURES
from detector.yolo_vehicle import ColorBlobDetector
from detector.synthetic import make_dataset
from detector import accident_model
from generate_sample_video import scene_frames

try:
    import torch  # noqa
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _tiny_model():
    """Train a quick model into a temp file (keeps tests independent of models/)."""
    from train_accident_detector import train
    path = os.path.join(_tmp, "m.pt")
    train(epochs=6, out_path=path, verbose=False)
    return accident_model.load_classifier(path)


class Tests(unittest.TestCase):
    def test_1_database(self):
        database.init_db()
        i = database.add_incident(0.9, "evidence/x.jpg", "unit-test", 2)
        self.assertEqual(database.list_incidents()[0]["id"], i)
        self.assertEqual(database.get_stats()["total_incidents"], 1)
        self.assertTrue(database.delete_incident(i))
        self.assertEqual(database.get_stats()["total_incidents"], 0)

    def test_2_tracker_kinematics(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        tr = CentroidTracker()
        for x in (0, 10, 20, 30, 31):                       # moves 10 px/frame, then stops
            tracks = tr.update([(x, 0, x + 20, 20)])
        self.assertEqual(len(tracks), 1)
        self.assertGreater(tracks[0].decel, 3.0)            # sudden deceleration detected
        f = frame_features(tracks, (360, 640, 3))
        self.assertEqual(f.shape, (NUM_FEATURES,))
        X, y = make_dataset(100)
        self.assertEqual(X.shape, (100, SEQ_LEN, NUM_FEATURES))
        self.assertAlmostEqual(float(y.mean()), 0.5)        # balanced

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_3_classifier(self):
        clf = _tiny_model()
        self.assertIsNotNone(clf)
        X, y = make_dataset(200, seed=7)
        p = np.array([clf.predict(s) for s in X])
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertGreater(((p >= 0.5) == (y == 1)).mean(), 0.8)

    @unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
    def test_4_pipeline_on_samples(self):
        from detector.pipeline import AccidentPipeline
        clf = _tiny_model()
        ev = os.path.join(_tmp, "ev")
        results = {}
        for kind in ("safe", "accident"):
            pipe = AccidentPipeline(detector=ColorBlobDetector(), classifier=clf,
                                    source="test", evidence_dir=ev)
            for i, fr in enumerate(scene_frames(kind)):
                pipe.process_frame(fr, i / 30)
            results[kind] = len(pipe.incidents)
        self.assertEqual(results["safe"], 0)
        self.assertGreaterEqual(results["accident"], 1)

    def test_5_flask_endpoints(self):
        import app as app_module
        c = app_module.app.test_client()
        self.assertEqual(c.get("/").status_code, 200)
        self.assertEqual(c.get("/api/stats").status_code, 200)
        self.assertEqual(c.get("/api/incidents").status_code, 200)
        self.assertEqual(c.post("/upload").status_code, 400)           # no file
        self.assertEqual(c.get("/stream/missing.mp4").status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
