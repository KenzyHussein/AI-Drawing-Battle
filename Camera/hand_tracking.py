
import cv2
import mediapipe as mp
from pathlib import Path


class HandTracker:

    def __init__(self):

        # Locate the MediaPipe model
        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "hand_landmarker.task"
        )

        # MediaPipe setup
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1
        )

        self.landmarker = HandLandmarker.create_from_options(options)

        # Hand landmark connections
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def process_frame(self, frame):

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hand
        result = self.landmarker.detect(mp_image)

        hand_landmarks = None

        if result.hand_landmarks:

            hand_landmarks = result.hand_landmarks[0]

            height, width, _ = frame.shape
            points = []

            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)
                cv2.circle(frame, (x, y), 9, (255, 255, 255), 1)

            # Draw hand connections
            for start, end in self.connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    4
                )

            # Index fingertip = landmark 8
            index_x, index_y = points[8]

            cv2.circle(frame, (index_x, index_y), 15, (0, 0, 255), -1)
            cv2.circle(frame, (index_x, index_y), 18, (255, 255, 255), 2)

            cv2.putText(
                frame,
                f"Index: ({index_x}, {index_y})",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return frame, hand_landmarks

    def close(self):
        self.landmarker.close()
