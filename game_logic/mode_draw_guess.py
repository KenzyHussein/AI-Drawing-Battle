import random
import time
import cv2
import numpy as np
from game_logic.ai_pipeline import CLASSES
from game_logic.mode_beat_timer import render_strokes


def play_draw_and_guess(cap, pipeline, draw_seconds=15):
    pipeline.reset_round()
    secret_word = random.choice(CLASSES)
    print(f"\n[PLAYER 1 ONLY] Your secret word is: {secret_word.upper()}")
    input("Make sure Player 2 is looking away, then press ENTER to start drawing...")

    start_time = time.time()
    clean_canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        if clean_canvas.shape[:2] != frame.shape[:2]:
            clean_canvas = np.zeros_like(frame)

        remaining = max(0.0, draw_seconds - (time.time() - start_time))

        # Track hand & capture strokes (NO AI predictions run or shown here)
        frame, is_drawing = pipeline.process_camera_frame(frame)
        all_strokes = pipeline.get_all_strokes()
        render_strokes(frame, all_strokes, color=(0, 255, 255), thickness=5)

        status_text = "DRAWING (PINCH)" if is_drawing else "PINCH THUMB + INDEX TO DRAW"
        cv2.putText(frame, status_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"PLAYER 1 | Time: {remaining:.1f}s", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)
        cv2.putText(frame, "Press 'c' to clear | 'q' to finish early", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Mode 3: Player 1 Canvas (AI Hints Disabled)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            pipeline.reset_round()
        elif remaining <= 0 or key == ord("q"):
            break

    cv2.destroyAllWindows()

    # Show final drawing only to Player 2
    render_strokes(clean_canvas, pipeline.get_all_strokes(), color=(255, 255, 255), thickness=6)
    cv2.imshow("Player 2 - Guess This Drawing!", clean_canvas)
    cv2.waitKey(500)

    guess = input("\nPlayer 2, type your guess: ").strip().lower()
    cv2.destroyAllWindows()

    is_correct = (guess == secret_word)
    score = 150 if is_correct else 0
    return {"mode": "Draw and Guess", "secret_word": secret_word, "guess": guess, "won": is_correct, "score": score}
