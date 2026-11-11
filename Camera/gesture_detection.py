
import numpy as np


class GestureDetector:

    def __init__(self, start_threshold=0.10, stop_threshold=0.13):

        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold

        self.is_drawing = False

    def detect(self, hand_landmarks):

        # If no hand is detected, stop drawing
        if hand_landmarks is None:
            self.is_drawing = False
            return False, None

        # Get thumb tip (landmark 4)
        thumb = hand_landmarks[4]

        # Get index fingertip (landmark 8)
        index = hand_landmarks[8]

        # Calculate normalized distance
        distance = np.sqrt(
            (thumb.x - index.x) ** 2 +
            (thumb.y - index.y) ** 2
        )

        # Detect pinch with hysteresis
        if not self.is_drawing and distance < self.start_threshold:
            self.is_drawing = True

        elif self.is_drawing and distance > self.stop_threshold:
            self.is_drawing = False

        return self.is_drawing, distance
