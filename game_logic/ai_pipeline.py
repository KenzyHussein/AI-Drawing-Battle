import os
import time
import cv2
import numpy as np

from Camera.hand_tracking import HandTracker
from Camera.gesture_detection import GestureDetector
from Camera.drawing_capture import DrawingCapture
from computer_vision.drawing_processing import preprocess_drawing
from Machine_Learning.predict import DrawingPredictor, CLASSES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_drawing_model", "cnn_drawing_model.keras")


class AIDrawingPipeline:
    def __init__(self, inference_interval=0.4):
        self.inference_interval = inference_interval
        self.last_inference_time = 0.0
        self.cached_top3 = []

        # Initialize Member 1 & Member 3 components
        self.hand_tracker = HandTracker()
        self.gesture_detector = GestureDetector(start_threshold=0.10, stop_threshold=0.13)
        self.drawing_capture = DrawingCapture()
        self.predictor = DrawingPredictor(model_path=CNN_MODEL_PATH)

    def reset_round(self):
        """Resets strokes and cached predictions for a new round."""
        self.drawing_capture.clear()
        self.drawing_capture.start_new_drawing()
        self.gesture_detector.is_drawing = False
        self.last_inference_time = 0.0
        self.cached_top3 = []

    def process_camera_frame(self, frame):
        """Runs Member 1's hand tracking, pinch detection, and stroke capture."""
        annotated_frame, hand_landmarks = self.hand_tracker.process_frame(frame)
        is_drawing, _ = self.gesture_detector.detect(hand_landmarks)
        self.drawing_capture.update(hand_landmarks, is_drawing)
        return annotated_frame, is_drawing

    def get_all_strokes(self):
        """Returns all completed strokes plus the active stroke being drawn."""
        strokes = list(self.drawing_capture.completed_strokes)
        if self.drawing_capture.current_stroke:
            strokes.append(self.drawing_capture.current_stroke)
        return strokes

    def _strokes_to_canvas(self, strokes, canvas_size=256):
        """Reconstructs Member 1's normalized strokes in RAM matching Member 2's canvas format."""
        canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
        for stroke in strokes:
            points = []
            for pt in stroke:
                x_norm = np.clip(float(pt["x"]), 0.0, 1.0)
                y_norm = np.clip(float(pt["y"]), 0.0, 1.0)
                x = int(round(x_norm * (canvas_size - 1)))
                y = int(round(y_norm * (canvas_size - 1)))
                points.append((x, y))

            for i in range(1, len(points)):
                cv2.line(canvas, points[i - 1], points[i], color=0, thickness=4, lineType=cv2.LINE_AA)
            if len(points) == 1:
                cv2.circle(canvas, points[0], 2, 0, -1)
        return canvas

    def predict_current_drawing(self, force=False):
        """Runs throttled inference using Member 2's preprocessing and Member 3's CNN."""
        strokes = self.get_all_strokes()
        if not strokes:
            return []

        now = time.time()
        if not force and (now - self.last_inference_time) < self.inference_interval:
            return self.cached_top3

        self.last_inference_time = now
        try:
            canvas = self._strokes_to_canvas(strokes, canvas_size=256)
            # Member 2 preprocessing returns float32 [0.0, 1.0] (white bg, black ink)
            processed_float = preprocess_drawing(canvas, output_size=28)
            # Convert back to uint8 [0, 255] so Member 3's predict_top3 handles inversion & scaling cleanly
            processed_uint8 = (processed_float * 255.0).astype(np.uint8)
            self.cached_top3 = self.predictor.predict_top3(processed_uint8)
        except ValueError:
            pass

        return self.cached_top3

    def close(self):
        self.hand_tracker.close()
