
import pandas as pd
import time


class DrawingCapture:

    def __init__(self):

        self.current_stroke = []
        self.completed_strokes = []
        self.all_points = []

        self.stroke_id = 0

    def update(self, hand_landmarks, is_drawing):

        # Start a new stroke
        if is_drawing and not self.current_stroke:

            self.stroke_id += 1
            self.current_stroke = []

        # Capture index fingertip coordinates
        if is_drawing and hand_landmarks is not None:

            index = hand_landmarks[8]

            point = {
                "stroke_id": self.stroke_id,
                "x": float(index.x),
                "y": float(index.y),
                "timestamp": time.time()
            }

            self.current_stroke.append(point)
            self.all_points.append(point.copy())

        # Finish the current stroke
        elif not is_drawing and self.current_stroke:

            self.completed_strokes.append(self.current_stroke)
            self.current_stroke = []

    def clear(self):

        self.current_stroke.clear()
        self.completed_strokes.clear()
        self.all_points.clear()

        self.stroke_id = 0

    def save(self, filename="drawing_coordinates.csv"):

        if not self.all_points:
            print("No coordinates to save.")
            return

        df = pd.DataFrame(self.all_points)

        df.to_csv(filename, index=False)

        print("Coordinates saved!")
        print(df.head())

    def get_points(self):
        return self.all_points
