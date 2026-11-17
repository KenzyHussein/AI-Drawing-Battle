
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

CNN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn_drawing_model",
    "cnn_drawing_model.keras"
)


class AIDrawingPipeline:
    def __init__(self, inference_interval=0.4):
        self.inference_interval = inference_interval
        self.last_inference_time = 0.0
        self.cached_top3 = []

        # Identifies each new prediction.
        # This prevents the game from counting the same cached prediction repeatedly.
        self.prediction_id = 0

        # Initialize camera and AI components.
        self.hand_tracker = HandTracker()
        self.gesture_detector = GestureDetector(
            start_threshold=0.10,
            stop_threshold=0.13
        )
        self.drawing_capture = DrawingCapture()
        self.predictor = DrawingPredictor(model_path=CNN_MODEL_PATH)

    def reset_round(self):
        """Reset drawing data and predictions for a new round."""
        self.drawing_capture.clear()
        self.drawing_capture.start_new_drawing()

        self.gesture_detector.is_drawing = False

        self.last_inference_time = 0.0
        self.cached_top3 = []
        self.prediction_id = 0

    def process_camera_frame(self, frame):
        """Track the hand and capture drawing strokes."""
        annotated_frame, hand_landmarks = self.hand_tracker.process_frame(frame)

        is_drawing, _ = self.gesture_detector.detect(hand_landmarks)

        self.drawing_capture.update(hand_landmarks, is_drawing)

        return annotated_frame, is_drawing

    def get_all_strokes(self):
        """Return completed strokes and the currently active stroke."""
        strokes = list(self.drawing_capture.completed_strokes)

        if self.drawing_capture.current_stroke:
            strokes.append(self.drawing_capture.current_stroke)

        return strokes

    def _strokes_to_canvas(self, strokes, canvas_size=256):
        """Convert normalized hand-tracking strokes into a white canvas."""
        canvas = np.ones(
            (canvas_size, canvas_size),
            dtype=np.uint8
        ) * 255

        for stroke in strokes:
            points = []

            for pt in stroke:
                x_norm = np.clip(float(pt["x"]), 0.0, 1.0)
                y_norm = np.clip(float(pt["y"]), 0.0, 1.0)

                x = int(round(x_norm * (canvas_size - 1)))
                y = int(round(y_norm * (canvas_size - 1)))

                points.append((x, y))

            for i in range(1, len(points)):
                cv2.line(
                    canvas,
                    points[i - 1],
                    points[i],
                    color=0,
                    thickness=4,
                    lineType=cv2.LINE_AA
                )

            if len(points) == 1:
                cv2.circle(
                    canvas,
                    points[0],
                    2,
                    0,
                    -1
                )

        return canvas

    def predict_current_drawing(self, force=False):
        """
        Predict the current drawing.

        A new prediction is performed only after the inference interval.
        prediction_id increases only when a fresh prediction succeeds.
        """
        strokes = self.get_all_strokes()

        if not strokes:
            return []

        now = time.time()

        # Return cached predictions if the inference interval has not passed.
        if (
            not force
            and (now - self.last_inference_time) < self.inference_interval
        ):
            return self.cached_top3

        self.last_inference_time = now

        try:
            canvas = self._strokes_to_canvas(
                strokes,
                canvas_size=256
            )

            # Preprocessing returns a white background with black strokes.
            processed_float = preprocess_drawing(
                canvas,
                output_size=28
            )

            # Convert to uint8 for the predictor.
            processed_uint8 = (
                processed_float * 255.0
            ).astype(np.uint8)

            # Run a fresh prediction.
            new_predictions = self.predictor.predict_top3(
                processed_uint8
            )

            self.cached_top3 = new_predictions

            # Increment only after a successful fresh prediction.
            self.prediction_id += 1

        except ValueError as error:
            print(f"Prediction preprocessing error: {error}")
            self.cached_top3 = []

        return self.cached_top3

    def close(self):
        """Release camera tracking resources."""
        self.hand_tracker.close()
