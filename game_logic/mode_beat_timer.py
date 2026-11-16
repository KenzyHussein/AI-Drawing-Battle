import random
import time
import cv2
from game_logic.ai_pipeline import CLASSES


def render_strokes(frame, strokes, color=(0, 255, 255), thickness=5):
    h, w, _ = frame.shape
    for stroke in strokes:
        for i in range(1, len(stroke)):
            p1 = (int(stroke[i - 1]["x"] * w), int(stroke[i - 1]["y"] * h))
            p2 = (int(stroke[i]["x"] * w), int(stroke[i]["y"] * h))
            cv2.line(frame, p1, p2, color, thickness)


def play_beat_the_timer(cap, pipeline, round_seconds=10):
    pipeline.reset_round()
    secret_word = random.choice(CLASSES)
    start_time = time.time()
    won = False
    score = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        elapsed = time.time() - start_time
        remaining = max(0.0, round_seconds - elapsed)

        # Member 1: Track hand, detect pinch, record strokes
        frame, is_drawing = pipeline.process_camera_frame(frame)
        all_strokes = pipeline.get_all_strokes()
        render_strokes(frame, all_strokes)

        # Member 2 + Member 3: Preprocess & predict Top-3 periodically
        top3 = pipeline.predict_current_drawing()
        top3_labels = [label for label, _ in top3]

        if secret_word in top3_labels:
            won = True
            score = int(100 + (remaining * 10))
            break

        # HUD Overlay
        status_text = "DRAWING (PINCH)" if is_drawing else "PINCH THUMB + INDEX TO DRAW"
        status_color = (0, 255, 0) if is_drawing else (0, 165, 255)
        cv2.putText(frame, status_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.putText(frame, f"TARGET: {secret_word.upper()}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(frame, f"TIME: {remaining:.1f}s", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)

        cv2.putText(frame, "AI Top-3 Guesses:", (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        for idx, (label, conf_pct) in enumerate(top3[:3]):
            cv2.putText(frame, f"{idx + 1}. {label} ({conf_pct:.1f}%)", (10, 230 + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AI Drawing Battle - Beat The Timer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            pipeline.reset_round()
        elif remaining <= 0 or key == ord("q"):
            break

    cv2.destroyAllWindows()
    return {"mode": "Beat the Timer", "secret_word": secret_word, "won": won, "score": score}
