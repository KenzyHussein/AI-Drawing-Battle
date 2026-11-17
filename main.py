import cv2
from game_logic.ai_pipeline import AIDrawingPipeline
from game_logic.mode_beat_timer import play_beat_the_timer
from game_logic.mode_draw_guess import play_draw_and_guess


def main():
    print("Loading MediaPipe HandLandmarker and CNN Model...")
    pipeline = AIDrawingPipeline(inference_interval=0.4)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        pipeline.close()
        return

    total_score = 0

    try:
        while True:
            print("\n====================================")
            print("         AI DRAWING BATTLE          ")
            print("====================================")
            print(f"Total Session Score: {total_score}")
            print("1. Mode 2: Beat the Timer (30s vs AI)")
            print("2. Mode 3: Draw & Guess (2 Players)")
            print("3. Exit")

            choice = input("Select an option (1/2/3): ").strip()

            if choice == "1":
                result = play_beat_the_timer(cap, pipeline, round_seconds=30)
                total_score += result["score"]
                status = "WIN!" if result["won"] else "TIME UP!"
                print(f"\n[{status}] Secret Word: {result['secret_word'].upper()} | +{result['score']} pts")

            elif choice == "2":
                result = play_draw_and_guess(cap, pipeline, draw_seconds=30)
                total_score += result["score"]
                status = "CORRECT!" if result["won"] else "WRONG!"
                print(f"\n[{status}] Word: {result['secret_word'].upper()} | Guess: {result['guess']} | +{result['score']} pts")

            elif choice == "3":
                print(f"Final Score: {total_score}. Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    finally:
        cap.release()
        pipeline.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
