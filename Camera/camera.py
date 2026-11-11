
import cv2
from pathlib import Path

from hand_tracking import HandTracker
from gesture_detection import GestureDetector
from drawing_capture import DrawingCapture


class DrawingCamera:

    def __init__(self):

        self.tracker = HandTracker()
        self.detector = GestureDetector()
        self.capture = DrawingCapture()

        self.camera = cv2.VideoCapture(0)

        self.output_path = (
            Path(__file__).resolve().parent
            / "drawing_coordinates.csv"
        )

    def run(self):

        try:
            while self.camera.isOpened():

                ret, frame = self.camera.read()

                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                height, width, _ = frame.shape

                # 1. Hand tracking
                frame, landmarks = self.tracker.process_frame(frame)

                # 2. Gesture detection
                is_drawing, distance = self.detector.detect(landmarks)

                # 3. Coordinate capture
                self.capture.update(landmarks, is_drawing)

                # 4. Display status
                if is_drawing:
                    status = "DRAWING"
                    color = (0, 255, 0)

                elif landmarks is not None:
                    status = "NOT DRAWING"
                    color = (0, 0, 255)

                else:
                    status = "NO HAND DETECTED"
                    color = (0, 0, 255)

                cv2.putText(
                    frame, status, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2
                )

                # 5. Display captured strokes
                for stroke in (
                    self.capture.completed_strokes
                    + [self.capture.current_stroke]
                ):

                    for i in range(1, len(stroke)):

                        p1 = (
                            int(stroke[i - 1]["x"] * width),
                            int(stroke[i - 1]["y"] * height)
                        )

                        p2 = (
                            int(stroke[i]["x"] * width),
                            int(stroke[i]["y"] * height)
                        )

                        cv2.line(
                            frame, p1, p2,
                            (0, 255, 255), 3
                        )

                # 6. Display information
                cv2.putText(
                    frame,
                    f"Points: {len(self.capture.all_points)}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

                cv2.putText(
                    frame,
                    f"Strokes: {self.capture.stroke_id}",
                    (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

                cv2.imshow("AI Drawing Battle - Camera", frame)

                # 7. Keyboard controls
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("c"):
                    self.capture.clear()
                    self.detector.is_drawing = False
                    print("Drawing cleared!")

                elif key == ord("s"):
                    self.capture.save(str(self.output_path))

        finally:
            self.camera.release()
            self.tracker.close()
            cv2.destroyAllWindows()

    def get_drawing_coordinates(self):
        return self.capture.get_points()


if __name__ == "__main__":

    drawing_camera = DrawingCamera()
    drawing_camera.run()
