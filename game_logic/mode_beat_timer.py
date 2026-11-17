
import random
import time
import cv2

from game_logic.ai_pipeline import CLASSES


def render_strokes(frame, strokes, color=(0, 255, 255), thickness=5):
    """Render captured hand-drawn strokes on the camera frame."""
    h, w, _ = frame.shape

    for stroke in strokes:
        for i in range(1, len(stroke)):
            p1 = (
                int(stroke[i - 1]["x"] * w),
                int(stroke[i - 1]["y"] * h)
            )

            p2 = (
                int(stroke[i]["x"] * w),
                int(stroke[i]["y"] * h)
            )

            cv2.line(
                frame,
                p1,
                p2,
                color,
                thickness
            )


def play_beat_the_timer(cap, pipeline, round_seconds=30):

    pipeline.reset_round()

    secret_word = random.choice(CLASSES)

    start_time = time.time()

    won = False
    score = 0

    # Recognition settings.
    CONFIDENCE_THRESHOLD = 75.0
    REQUIRED_CONFIRMATIONS = 3
    DRAWING_RELEASE_DELAY = 1.0

    # Track drawing activity.
    last_drawing_time = time.time()

    # Track fresh predictions only.
    last_checked_prediction_id = pipeline.prediction_id
    correct_count = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        elapsed = time.time() - start_time
        remaining = max(0.0, round_seconds - elapsed)

        # -----------------------------------------
        # 1. Hand tracking and drawing capture
        # -----------------------------------------

        frame, is_drawing = pipeline.process_camera_frame(frame)

        all_strokes = pipeline.get_all_strokes()

        render_strokes(frame, all_strokes)

        # Update the last time the player was drawing.
        if is_drawing:
            last_drawing_time = time.time()

            # Do not count predictions while the player is drawing.
            correct_count = 0

        time_since_last_draw = time.time() - last_drawing_time

        # -----------------------------------------
        # 2. AI prediction
        # -----------------------------------------

        top3 = pipeline.predict_current_drawing()

        target_confidence = 0.0

        for label, confidence in top3:
            if label == secret_word:
                target_confidence = confidence
                break

        # -----------------------------------------
        # 3. Confirm recognition
        # -----------------------------------------

        drawing_finished = (
            not is_drawing
            and time_since_last_draw >= DRAWING_RELEASE_DELAY
        )

        # Count only a new prediction, not a cached one.
        if pipeline.prediction_id != last_checked_prediction_id:

            last_checked_prediction_id = pipeline.prediction_id

            if drawing_finished:

                if target_confidence >= CONFIDENCE_THRESHOLD:
                    correct_count += 1
                else:
                    correct_count = 0

            else:
                correct_count = 0

        # -----------------------------------------
        # 4. Check win condition
        # -----------------------------------------

        if (
            drawing_finished
            and correct_count >= REQUIRED_CONFIRMATIONS
        ):

            won = True

            score = int(100 + (remaining * 10))

        # -----------------------------------------
        # 5. Display game information
        # -----------------------------------------

        if is_drawing:
            status_text = "DRAWING..."
            status_color = (0, 255, 0)

        elif not drawing_finished:
            status_text = "WAITING FOR YOU TO FINISH..."
            status_color = (0, 165, 255)

        elif correct_count > 0:
            status_text = (
                f"CONFIRMING AI PREDICTION "
                f"({correct_count}/{REQUIRED_CONFIRMATIONS})"
            )
            status_color = (0, 255, 255)

        else:
            status_text = "AI IS CHECKING YOUR DRAWING"
            status_color = (255, 255, 0)

        cv2.putText(
            frame,
            status_text,
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_color,
            2
        )

        cv2.putText(
            frame,
            f"TARGET: {secret_word.upper()}",
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"TIME: {remaining:.1f}s",
            (10, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 200, 255),
            2
        )

        cv2.putText(
            frame,
            "AI Top-3 Guesses:",
            (10, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            2
        )

        for idx, (label, confidence) in enumerate(top3[:3]):

            cv2.putText(
                frame,
                f"{idx + 1}. {label} ({confidence:.1f}%)",
                (10, 230 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # -----------------------------------------
        # 6. Display win message
        # -----------------------------------------

        if won:

            cv2.putText(
                frame,
                "YOU WON!",
                (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

        cv2.imshow(
            "AI Drawing Battle - Beat The Timer",
            frame
        )

        # -----------------------------------------
        # 7. Keyboard controls
        # -----------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):

            pipeline.reset_round()

            last_drawing_time = time.time()
            last_checked_prediction_id = pipeline.prediction_id
            correct_count = 0

        elif key == ord("q"):
            break

        # Keep the win message visible briefly before closing.
        elif won:
            cv2.waitKey(1000)
            break

        elif remaining <= 0:
            break

    cv2.destroyAllWindows()

    return {
        "mode": "Beat the Timer",
        "secret_word": secret_word,
        "won": won,
        "score": score
    }
