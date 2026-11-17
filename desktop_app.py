
import random
import threading
import time

import cv2
import customtkinter as ctk
from PIL import Image

from PIL import ImageTk


# ============================================================
# CONFIGURATION
# ============================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CLASSES = [
    "apple", "car", "cat", "clock",
    "fish", "house", "star", "tree"
]

ROUND_DURATION = 30
CONFIDENCE_THRESHOLD = 75.0
REQUIRED_CONFIRMATIONS = 3
WORD_REVEAL_SECONDS = 5
COUNTDOWN_SECONDS = 3


# ============================================================
# MAIN DESKTOP APPLICATION
# ============================================================

def remaining_if_started(start_time, duration):
    return max(0, duration - (time.time() - start_time))


class DrawingBattleApp(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title("AI Drawing Battle")
        self.geometry("1250x820")
        self.minsize(1050, 700)

        self.configure(fg_color="#101322")

        # -----------------------------
        # Thread-safe shared state
        # -----------------------------

        self.lock = threading.Lock()

        self.running = True

        self.latest_frame = None
        self.latest_display_frame = None
        self.latest_hand_detected = False
        self.latest_strokes = []
        self.latest_top3 = []

        self.camera_status = "Opening camera..."
        self.ai_status = "Loading AI models..."

        self.pipeline_ready = False
        self.pipeline_error = None

        # -----------------------------
        # Game state
        # -----------------------------

        self.mode = "Beat the Timer"
        self.selected_player = "Player 1"

        self.active = False
        self.round_id = 0
        self.clear_id = 0

        self.secret_word = None

        self.start_time = None
        self.duration = ROUND_DURATION

        self.finished = False
        self.won = False
        self.score = 0

        # Cumulative scores are kept for the whole app session.
        self.player1_score = 0
        self.player2_score = 0
        self.result_message = ""

        self.status = "Ready to play"

        self.draw_guess_phase = False
        self.word_reveal_until = 0.0

        # -----------------------------
        # Build UI
        # -----------------------------

        self.build_ui()
        self.bind_all("<KeyPress-c>", self.handle_clear_key)
        self.bind_all("<KeyPress-C>", self.handle_clear_key)

        # -----------------------------
        # Start camera thread
        # -----------------------------

        self.camera_thread = threading.Thread(
            target=self.camera_worker,
            daemon=True
        )

        self.camera_thread.start()

        # -----------------------------
        # Start AI thread
        # -----------------------------

        self.ai_thread = threading.Thread(
            target=self.ai_worker,
            daemon=True
        )

        self.ai_thread.start()

        # -----------------------------
        # Start UI updates
        # -----------------------------

        self.update_camera()
        self.update_game_ui()

        self.protocol(
            "WM_DELETE_WINDOW",
            self.close_app
        )

    # ========================================================
    # UI DESIGN
    # ========================================================

    def build_ui(self):

        # Main container
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        header = ctk.CTkFrame(
            self,
            fg_color="#191E35",
            corner_radius=18
        )

        header.grid(
            row=0,
            column=0,
            padx=20,
            pady=(20, 10),
            sticky="ew"
        )

        ctk.CTkLabel(
            header,
            text="🎨 AI DRAWING BATTLE",
            font=("Arial", 28, "bold"),
            text_color="#FFFFFF"
        ).pack(
            side="left",
            padx=25,
            pady=18
        )

        self.connection_label = ctk.CTkLabel(
            header,
            text="● CONNECTING",
            font=("Arial", 13, "bold"),
            text_color="#F5B942"
        )

        self.connection_label.pack(
            side="right",
            padx=25
        )

        # ----------------------------------------------------
        # MAIN CONTENT
        # ----------------------------------------------------

        content = ctk.CTkFrame(
            self,
            fg_color="transparent"
        )

        content.grid(
            row=1,
            column=0,
            padx=20,
            pady=10,
            sticky="nsew"
        )

        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)

        # ====================================================
        # LEFT PANEL - CAMERA
        # ====================================================

        camera_panel = ctk.CTkFrame(
            content,
            fg_color="#191E35",
            corner_radius=18
        )

        camera_panel.grid(
            row=0,
            column=0,
            padx=(0, 10),
            sticky="nsew"
        )

        camera_panel.grid_columnconfigure(0, weight=1)
        camera_panel.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            camera_panel,
            text="LIVE CAMERA",
            font=("Arial", 20, "bold")
        ).grid(
            row=0,
            column=0,
            pady=15
        )

        self.camera_label = ctk.CTkLabel(
            camera_panel,
            text="Opening your local camera...",
            fg_color="#080B16",
            corner_radius=12
        )

        self.camera_label.grid(
            row=1,
            column=0,
            padx=15,
            pady=10,
            sticky="nsew"
        )

        self.camera_info = ctk.CTkLabel(
            camera_panel,
            text="Waiting for webcam...",
            font=("Arial", 12),
            text_color="#AAB2D0"
        )

        self.camera_info.grid(
            row=2,
            column=0,
            pady=10
        )

        # ====================================================
        # RIGHT PANEL - GAME CONTROLS
        # ====================================================

        right_panel = ctk.CTkScrollableFrame(
            content,
            fg_color="#191E35",
            corner_radius=18,
            scrollbar_button_color="#3D4664",
            scrollbar_button_hover_color="#596486"
        )

        right_panel.grid(
            row=0,
            column=1,
            padx=(10, 0),
            sticky="nsew"
        )

        right_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right_panel,
            text="GAME CONTROL",
            font=("Arial", 21, "bold")
        ).pack(
            pady=(20, 15)
        )

        # ----------------------------------------------------
        # GAME MODE
        # ----------------------------------------------------

        ctk.CTkLabel(
            right_panel,
            text="SELECT GAME MODE",
            font=("Arial", 12, "bold"),
            text_color="#AAB2D0"
        ).pack(pady=(5, 8))

        self.mode_selector = ctk.CTkSegmentedButton(
            right_panel,
            values=[
                "Beat the Timer",
                "Draw & Guess"
            ],
            command=self.change_mode
        )

        self.mode_selector.set("Beat the Timer")

        self.mode_selector.pack(
            padx=15,
            fill="x",
            pady=5
        )

        self.player_choice_label = ctk.CTkLabel(
            right_panel,
            text="CHOOSE YOUR PLAYER",
            font=("Arial", 12, "bold"),
            text_color="#AAB2D0"
        )
        self.player_choice_label.pack(pady=(12, 5))

        self.player_selector = ctk.CTkSegmentedButton(
            right_panel,
            values=["Player 1", "Player 2"],
            command=self.change_player
        )
        self.player_selector.set("Player 1")
        self.player_selector.pack(
            padx=20,
            fill="x",
            pady=(0, 5)
        )

        # START BUTTON - placed near the top so it remains visible
        self.start_button = ctk.CTkButton(
            right_panel,
            text="▶ START NEW ROUND",
            height=48,
            font=("Arial", 15, "bold"),
            fg_color="#2675E8",
            hover_color="#185CBF",
            command=self.start_round
        )

        self.start_button.pack(
            padx=20,
            pady=(12, 8),
            fill="x"
        )

        # ----------------------------------------------------
        # GUESS INPUT
        # ----------------------------------------------------

        self.guess_entry = ctk.CTkEntry(
            right_panel,
            placeholder_text="Player 2: Enter your guess",
            height=40
        )

        # Hidden until Player 1 submits or the drawing timer expires.
        self.guess_entry.configure(state="disabled")

        self.guess_entry.bind(
            "<Return>",
            lambda event: self.submit_guess()
        )

        self.guess_button = ctk.CTkButton(
            right_panel,
            text="SUBMIT GUESS",
            command=self.submit_guess,
            height=38,
            fg_color="#7756D8",
            hover_color="#6242BC",
            state="disabled"
        )

        # The guess button is packed only during Player 2's turn.


        self.submit_button = ctk.CTkButton(
            right_panel,
            text="SUBMIT DRAWING / END TURN",
            height=42,
            fg_color="#168B70",
            hover_color="#116D58",
            command=self.submit_drawing
        )

        self.submit_button.pack(
            padx=20,
            pady=(10, 5),
            fill="x"
        )


        # ----------------------------------------------------
        # TARGET WORD
        # ----------------------------------------------------

        self.target_label = ctk.CTkLabel(
            right_panel,
            text="TARGET: ---",
            font=("Arial", 23, "bold"),
            text_color="#42D9FF"
        )

        self.target_label.pack(pady=(25, 10))

        # ----------------------------------------------------
        # TIMER
        # ----------------------------------------------------

        self.timer_label = ctk.CTkLabel(
            right_panel,
            text="30",
            font=("Arial", 48, "bold"),
            text_color="#FFFFFF"
        )

        self.timer_label.pack(pady=5)

        ctk.CTkLabel(
            right_panel,
            text="SECONDS REMAINING",
            font=("Arial", 11),
            text_color="#AAB2D0"
        ).pack()

        self.timer_progress = ctk.CTkProgressBar(
            right_panel,
            progress_color="#42D9FF"
        )

        self.timer_progress.pack(
            padx=25,
            fill="x",
            pady=15
        )

        self.timer_progress.set(1)

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        self.status_label = ctk.CTkLabel(
            right_panel,
            text="Ready to play",
            font=("Arial", 14, "bold"),
            text_color="#42D9FF",
            wraplength=300
        )

        self.status_label.pack(pady=10)

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        self.score_label = ctk.CTkLabel(
            right_panel,
            text="SCORE: 0",
            font=("Arial", 20, "bold"),
            text_color="#F5C451"
        )

        self.score_label.pack(pady=(10, 4))

        self.scoreboard_label = ctk.CTkLabel(
            right_panel,
            text="PLAYER 1 SCORE: 0",
            font=("Arial", 14, "bold"),
            text_color="#8FC7FF"
        )
        self.scoreboard_label.pack(pady=(2, 10))

        self.result_label = ctk.CTkLabel(
            right_panel,
            text="",
            font=("Arial", 15, "bold"),
            text_color="#F5C451",
            justify="center",
            wraplength=300
        )
        self.result_label.pack(padx=12, pady=(4, 10))

        # ----------------------------------------------------
        # AI PREDICTIONS
        # ----------------------------------------------------

        ctk.CTkLabel(
            right_panel,
            text="AI PREDICTIONS",
            font=("Arial", 15, "bold")
        ).pack(pady=(20, 8))

        self.prediction_label = ctk.CTkLabel(
            right_panel,
            text="Waiting for your drawing...",
            font=("Arial", 13),
            justify="left",
            text_color="#D5D9EA"
        )

        self.prediction_label.pack(
            padx=15,
            pady=5
        )

        # ----------------------------------------------------
        # GAME BUTTONS
        # ----------------------------------------------------

        self.clear_button = ctk.CTkButton(
            right_panel,
            text="CLEAR DRAWING  (C)",
            height=40,
            fg_color="#3D4664",
            hover_color="#303851",
            command=self.clear_drawing
        )

        self.clear_button.pack(
            padx=20,
            pady=5,
            fill="x"
        )

        self.exit_button = ctk.CTkButton(
            right_panel,
            text="EXIT GAME",
            height=38,
            fg_color="#A83F55",
            hover_color="#853044",
            command=self.close_app
        )

        self.exit_button.pack(
            padx=20,
            pady=(10, 20),
            fill="x"
        )

        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        footer = ctk.CTkLabel(
            self,
            text=(
                "AI Drawing Battle  |  "
                "Pinch thumb + index finger to draw  |  Press C to clear"
            ),
            font=("Arial", 12),
            text_color="#7F89A8"
        )

        footer.grid(
            row=2,
            column=0,
            pady=(5, 15)
        )

        self.hide_guess_controls()

    # ========================================================
    # CAMERA THREAD
    # ========================================================

    def camera_worker(self):

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():

            cap.release()

            cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            with self.lock:
                self.camera_status = "Camera unavailable"

            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        with self.lock:
            self.camera_status = "Camera connected"

        while self.running:

            ret, frame = cap.read()

            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)

            with self.lock:
                self.latest_frame = frame

        cap.release()

    # ========================================================
    # AI PROCESSING THREAD
    # ========================================================

    def ai_worker(self):

        pipeline = None

        try:

            print("Loading AI pipeline...")

            from game_logic.ai_pipeline import AIDrawingPipeline

            pipeline = AIDrawingPipeline(
                inference_interval=0.8
            )

            with self.lock:
                self.pipeline_ready = True
                self.ai_status = "AI Ready"

            print("AI pipeline loaded successfully!")

            last_round_id = -1
            last_clear_id = -1

            last_drawing_time = time.time()

            last_prediction_id = 0
            correct_count = 0

            frame_counter = 0

            while self.running:

                with self.lock:

                    frame = (
                        self.latest_frame.copy()
                        if self.latest_frame is not None
                        else None
                    )

                    active = self.active
                    round_id = self.round_id
                    clear_id = self.clear_id

                    mode = self.mode
                    selected_player = self.selected_player
                    secret_word = self.secret_word
                    start_time = self.start_time
                    duration = self.duration
                    guessing = self.draw_guess_phase
                    word_reveal_until = self.word_reveal_until

                if frame is None:

                    time.sleep(0.02)
                    continue

                # Reset pipeline on new round.
                if (
                    round_id != last_round_id
                    or clear_id != last_clear_id
                ):

                    if active:
                        pipeline.reset_round()

                    last_round_id = round_id
                    last_clear_id = clear_id

                    last_drawing_time = time.time()
                    last_prediction_id = pipeline.prediction_id
                    correct_count = 0

                    with self.lock:
                        self.latest_strokes = []
                        self.latest_top3 = []

                frame_counter += 1

                # Process every second frame to keep the preview responsive.
                if frame_counter % 2 != 0:
                    time.sleep(0.005)
                    continue

                # -----------------------------
                # HAND TRACKING + CAMERA OVERLAY
                # -----------------------------

                drawing_enabled = (
                    active
                    and not guessing
                    and (mode != "Draw & Guess" or (start_time is not None and time.time() >= start_time))
                )

                if drawing_enabled:
                    annotated_frame, is_drawing = pipeline.process_camera_frame(frame)
                    strokes = pipeline.get_all_strokes()
                    hand_detected = bool(
                        getattr(pipeline, "last_hand_detected", is_drawing)
                    )
                else:
                    # Keep detecting and drawing hand landmarks even before a round starts.
                    # HandTracker exposes process_frame(), not detect().
                    annotated_frame, landmarks = pipeline.hand_tracker.process_frame(frame)
                    is_drawing = False
                    strokes = []
                    hand_detected = landmarks is not None

                with self.lock:
                    self.latest_display_frame = annotated_frame.copy()
                    self.latest_hand_detected = hand_detected

                # Do not capture strokes or run predictions outside the drawing phase.
                if not drawing_enabled:
                    time.sleep(0.005)
                    continue

                if is_drawing:
                    last_drawing_time = time.time()
                    correct_count = 0

                # -----------------------------
                # AI PREDICTION
                # -----------------------------

                top3 = []

                if mode == "Beat the Timer":
                    top3 = pipeline.predict_current_drawing()

                # -----------------------------
                # UPDATE SHARED DATA
                # -----------------------------

                with self.lock:

                    self.latest_strokes = list(strokes)
                    self.latest_top3 = list(top3)

                # -----------------------------
                # BEAT THE TIMER LOGIC
                # -----------------------------

                if mode == "Beat the Timer":

                    time_since_draw = (
                        time.time() - last_drawing_time
                    )

                    drawing_finished = (
                        not is_drawing
                        and time_since_draw >= 1.0
                    )

                    target_confidence = 0.0

                    for label, confidence in top3:

                        if label == secret_word:

                            target_confidence = confidence
                            break

                    if (
                        pipeline.prediction_id
                        != last_prediction_id
                    ):

                        last_prediction_id = (
                            pipeline.prediction_id
                        )

                        if drawing_finished:

                            if target_confidence >= CONFIDENCE_THRESHOLD:
                                correct_count += 1
                            else:
                                correct_count = 0

                        else:
                            correct_count = 0

                    with self.lock:

                        self.status = (
                            f"Confirming AI: {correct_count}/"
                            f"{REQUIRED_CONFIRMATIONS}"
                        )

                    if correct_count >= REQUIRED_CONFIRMATIONS:

                        remaining = max(
                            0,
                            duration - (
                                time.time() - start_time
                            )
                        )

                        score = int(100 + remaining * 10)

                        with self.lock:

                            if self.active:
                                self.active = False
                                self.won = True
                                self.finished = True
                                self.score = score
                                if selected_player == "Player 1":
                                    self.player1_score += score
                                else:
                                    self.player2_score += score
                                self.result_message = (
                                    f"{selected_player.upper()} WINS!\n"
                                    f"Correct answer: {secret_word.upper()}\n"
                                    f"Points earned: +{score}"
                                )
                                self.status = "YOU WON!"

                time.sleep(0.005)

        except Exception as error:

            print("AI pipeline error:", repr(error))

            with self.lock:

                self.pipeline_error = str(error)
                self.ai_status = "AI initialization failed"

        finally:

            if pipeline is not None:

                pipeline.close()

    # ========================================================
    # DRAW STROKES ON DISPLAY FRAME
    # ========================================================

    def render_strokes(self, frame, strokes):

        height, width = frame.shape[:2]

        for stroke in strokes:

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
                    frame,
                    p1,
                    p2,
                    (0, 255, 255),
                    4,
                    cv2.LINE_AA
                )

        return frame

    # ========================================================
    # CAMERA DISPLAY - TKINTER MAIN THREAD
    # ========================================================

    def update_camera(self):

        if not self.running:
            return

        with self.lock:

            source_frame = (
                self.latest_display_frame
                if self.latest_display_frame is not None
                else self.latest_frame
            )

            frame = source_frame.copy() if source_frame is not None else None

            strokes = list(self.latest_strokes)
            hand_detected = self.latest_hand_detected
            camera_status = self.camera_status
            ai_status = self.ai_status
            mode = self.mode
            active = self.active
            guessing = self.draw_guess_phase
            start_time = self.start_time
            reveal_until = self.word_reveal_until
            duration = self.duration
            status = self.status
            top3 = list(self.latest_top3)

        if frame is not None:

            frame = self.render_strokes(
                frame,
                strokes
            )

            # Helpful on-camera instructions and live hand-tracking status.
            cv2.putText(
                frame,
                "AI DRAWING BATTLE",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            tracking_text = "HAND DETECTED - PINCH TO DRAW" if hand_detected else "SHOW YOUR HAND / PINCH TO DRAW"
            tracking_color = (70, 220, 150) if hand_detected else (0, 190, 255)
            cv2.putText(
                frame,
                tracking_text,
                (18, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                tracking_color,
                2,
                cv2.LINE_AA
            )

            # Mode-specific information is drawn directly on the camera preview.
            if mode == "Beat the Timer":
                panel_x, panel_y = 18, 88
                panel_w, panel_h = 265, 125
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (12, 18, 38), -1)
                cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
                cv2.putText(frame, "TOP 3 AI PREDICTIONS", (panel_x + 12, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 220, 255), 2, cv2.LINE_AA)
                if top3:
                    for index, (label, confidence) in enumerate(top3[:3]):
                        line = f"{index + 1}. {label.upper()}  {confidence:.1f}%"
                        cv2.putText(frame, line, (panel_x + 14, panel_y + 57 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Draw to see predictions", (panel_x + 12, panel_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 215, 230), 1, cv2.LINE_AA)
            else:
                panel_x, panel_y = 18, 88
                panel_w, panel_h = 300, 112
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (12, 18, 38), -1)
                cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
                if guessing:
                    phase_text = "PLAYER 2: YOUR TURN!"
                    detail_text = "Enter your answer in the game panel"
                elif active and time.time() < reveal_until:
                    phase_text = "PLAYER 1: MEMORIZE THE WORD"
                    detail_text = "Remember the word before it disappears"
                elif active and start_time is not None and time.time() < start_time:
                    countdown_value = max(1, int(start_time - time.time() + 0.999))
                    phase_text = f"GET READY: {countdown_value}"
                    detail_text = "Get ready to start drawing!"
                elif active and start_time is not None and time.time() - start_time < 0.8:
                    phase_text = "DRAW!"
                    detail_text = "Player 1: start drawing now"
                elif active:
                    phase_text = "PLAYER 1: DRAW NOW!"
                    detail_text = "Player 2: look away from the screen"
                else:
                    phase_text = "DRAW & GUESS MODE"
                    detail_text = "Start a round to reveal the secret word"
                cv2.putText(frame, phase_text, (panel_x + 12, panel_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, detail_text, (panel_x + 12, panel_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
                if active and start_time is not None:
                    if time.time() < reveal_until:
                        remaining = reveal_until - time.time()
                        timer_text = f"WORD HIDES IN: {remaining:03.1f}s"
                    elif time.time() < start_time:
                        countdown_value = max(1, int(start_time - time.time() + 0.999))
                        timer_text = f"DRAW IN: {countdown_value}"
                    else:
                        remaining = max(0, duration - (time.time() - start_time))
                        timer_text = f"TIME: {remaining:04.1f}s"
                    cv2.putText(frame, timer_text, (panel_x + 12, panel_y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 240, 160), 2, cv2.LINE_AA)

            cv2.putText(
                frame,
                "PINCH THUMB + INDEX FINGER TO DRAW",
                (18, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(frame)

            image.thumbnail((720, 540))

            photo = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=image.size
            )

            self.camera_label.configure(
                image=photo,
                text=""
            )

            self.camera_label.image = photo

        else:

            self.camera_label.configure(
                image=None,
                text="Waiting for local webcam..."
            )

        with self.lock:

            ready = self.pipeline_ready
            error = self.pipeline_error

        if error:

            self.connection_label.configure(
                text="● AI ERROR",
                text_color="#F15B69"
            )

            self.camera_info.configure(
                text=f"AI Error: {error[:100]}"
            )

        elif ready:

            self.connection_label.configure(
                text="● CAMERA + AI READY",
                text_color="#48D597"
            )

            self.camera_info.configure(
                text=f"{camera_status} | {ai_status}"
            )

        else:

            self.connection_label.configure(
                text="● LOADING AI",
                text_color="#F5B942"
            )

            self.camera_info.configure(
                text=f"{camera_status} | {ai_status}"
            )

        self.after(30, self.update_camera)

    def hide_guess_controls(self):
        """Keep Player 2's answer controls completely out of view."""
        self.guess_entry.pack_forget()
        self.guess_button.pack_forget()
        self.guess_entry.configure(state="disabled")
        self.guess_button.configure(state="disabled")

    def show_guess_controls(self):
        """Show the answer controls only when Player 2's turn begins."""
        if self.mode != "Draw & Guess":
            self.hide_guess_controls()
            return

        self.guess_entry.pack(
            before=self.target_label,
            padx=20,
            pady=(15, 5),
            fill="x"
        )
        self.guess_button.pack(
            before=self.target_label,
            padx=20,
            pady=5,
            fill="x"
        )
        self.guess_entry.configure(state="normal")
        self.guess_button.configure(state="normal")
        self.guess_entry.focus()

    # ========================================================
    # CHANGE GAME MODE
    # ========================================================

    def change_player(self, selected_player):

       if self.mode != "Beat the Timer":
          return

       with self.lock:
          self.selected_player = selected_player
          self.status = f"{selected_player} selected"

       self.status_label.configure(text=f"{selected_player} selected")
       self.update_game_ui()

    def change_mode(self, selected_mode):

        if self.active:
            return

        if selected_mode == "Beat the Timer":
            self.player_choice_label.pack(pady=(12, 5))
            self.player_selector.pack(
                padx=20,
                fill="x",
                pady=(0, 5)
            )
        else:
            self.player_choice_label.pack_forget()
            self.player_selector.pack_forget()

        with self.lock:

            self.mode = selected_mode
            self.secret_word = None

            self.finished = False
            self.won = False

            self.latest_top3 = []
            self.latest_strokes = []

            self.status = "Ready to play"
            self.result_message = ""

        self.result_label.configure(text="")
        self.hide_guess_controls()
        self.guess_entry.delete(0, "end")
        self.submit_button.configure(state="disabled")

        self.target_label.configure(
            text="TARGET: ---"
        )

        self.prediction_label.configure(
            text="Waiting for your drawing..."
        )

    # ========================================================
    # START ROUND
    # ========================================================

    def start_round(self):

        with self.lock:

            if not self.pipeline_ready:

                self.status = "Please wait for AI initialization"
                return

            self.round_id += 1

            self.active = True

            self.secret_word = random.choice(CLASSES)

            self.duration = ROUND_DURATION
            now = time.time()
            if self.mode == "Draw & Guess":
                self.word_reveal_until = now + WORD_REVEAL_SECONDS
                # Drawing timer starts only after the 3-2-1 countdown.
                self.start_time = self.word_reveal_until + COUNTDOWN_SECONDS
            else:
                self.word_reveal_until = now
                self.start_time = now

            self.finished = False
            self.won = False
            self.score = 0
            self.result_message = ""

            self.draw_guess_phase = False

            self.latest_top3 = []
            self.latest_strokes = []

            self.status = "DRAWING..."

            secret_word = self.secret_word
            mode = self.mode

        self.result_label.configure(text="")
        self.hide_guess_controls()
        self.guess_entry.delete(0, "end")
        self.submit_button.configure(state="normal")

        if mode == "Beat the Timer":

            self.target_label.configure(
                text=f"{self.selected_player.upper()} - TARGET: {secret_word.upper()}"
            )

        else:

            self.target_label.configure(
                text=f"PLAYER 1 - MEMORIZE: {secret_word.upper()}"
            )

        self.prediction_label.configure(
            text="Waiting for your drawing..."
        )

    # ========================================================
    # KEYBOARD CLEAR SHORTCUT
    # ========================================================

    def handle_clear_key(self, event=None):

        # Do not intercept the letter C while Player 2 is typing an answer.
        try:
            focused = self.focus_get()
            if focused == self.guess_entry:
                return
        except Exception:
            pass

        self.clear_drawing()

    # ========================================================
    # SUBMIT DRAWING / END TURN EARLY
    # ========================================================

    def submit_drawing(self):

        with self.lock:
            if not self.active:
                return

            mode = self.mode
            secret_word = self.secret_word
            start_time = self.start_time
            duration = self.duration
            top3 = list(self.latest_top3)

            if mode == "Draw & Guess":
                if start_time is not None and time.time() < start_time:
                    self.status = "Wait until the secret word disappears before submitting."
                    return

                self.active = False
                self.finished = False
                self.draw_guess_phase = True
                self.status = "PLAYER 2: ENTER YOUR GUESS"
                self.latest_top3 = []

            else:
                remaining = max(
                    0.0,
                    duration - (time.time() - start_time)
                ) if start_time is not None else 0.0

                confidence = next(
                    (float(score) for label, score in top3 if label == secret_word),
                    0.0
                )
                correct = confidence >= CONFIDENCE_THRESHOLD

                self.active = False
                self.finished = True
                self.won = correct
                self.score = int(100 + remaining * 10) if correct else 0
                if correct:
                    if self.selected_player == "Player 1":
                        self.player1_score += self.score
                    else:
                        self.player2_score += self.score
                    self.result_message = (
                        f"{self.selected_player.upper()} WINS!\nCorrect answer: {secret_word.upper()}\n"
                        f"Points earned: +{self.score}\nAI confidence: {confidence:.1f}%"
                    )
                    self.status = "YOU WON!"
                else:
                    self.result_message = (
                        f"{self.selected_player.upper()} LOSES\nCorrect answer: {secret_word.upper()}\n"
                        f"Points earned: 0\nAI confidence: {confidence:.1f}%"
                    )
                    self.status = "AI DID NOT MATCH THE DRAWING"

        if mode == "Draw & Guess":
            self.target_label.configure(text="PLAYER 2: GUESS THE DRAWING")
            self.timer_label.configure(text="SUBMITTED")
            self.show_guess_controls()
        else:
            self.submit_button.configure(state="disabled")

    # ========================================================
    # CLEAR DRAWING
    # ========================================================

    def clear_drawing(self):

        with self.lock:

            if not self.active:
                return

            self.clear_id += 1

            self.latest_strokes = []
            self.latest_top3 = []

            self.status = "Drawing cleared"

    # ========================================================
    # DRAW & GUESS SUBMISSION
    # ========================================================

    def submit_guess(self):

        with self.lock:

            if (
                self.mode != "Draw & Guess"
                or not self.draw_guess_phase
                or self.secret_word is None
            ):
                return

            guess = self.guess_entry.get().strip().lower()

            if not guess:
                self.status = "Please enter a guess"
                return

            correct = guess == self.secret_word

            self.active = False
            self.finished = True
            self.won = correct

            self.score = 150 if correct else 0
            if correct:
                self.player2_score += self.score
                self.result_message = (
                    f"PLAYER 2 WINS!\nCorrect answer: {self.secret_word.upper()}\n"
                    f"Points earned: +{self.score}"
                )
                self.status = "CORRECT! PLAYER 2 WON!"
            else:
                self.score = 100
                self.player1_score += self.score
                self.result_message = (
                    f"PLAYER 1 WINS!\nCorrect answer: {self.secret_word.upper()}\n"
                    f"Points earned: +{self.score}"
                )
                self.status = "WRONG GUESS! PLAYER 1 WINS!"

            self.draw_guess_phase = False

        self.hide_guess_controls()
        self.submit_button.configure(state="disabled")

    # ========================================================
    # UPDATE GAME UI
    # ========================================================

    def update_game_ui(self):

        if not self.running:
            return

        with self.lock:

            active = self.active
            mode = self.mode

            start_time = self.start_time
            reveal_until = self.word_reveal_until
            duration = self.duration

            finished = self.finished
            won = self.won
            score = self.score

            status = self.status
            secret_word = self.secret_word
            player1_score = self.player1_score
            player2_score = self.player2_score
            selected_player = self.selected_player
            result_message = self.result_message

            top3 = list(self.latest_top3)

        # -----------------------------
        # TIMER
        # -----------------------------

        if active and start_time is not None:

            now = time.time()
            if mode == "Draw & Guess" and now < reveal_until:
                reveal_remaining = max(0, reveal_until - now)
                self.timer_label.configure(text=f"{reveal_remaining:.1f}", text_color="#FFFFFF")
                self.timer_progress.set(1.0)
                self.target_label.configure(
                    text=f"PLAYER 1 - MEMORIZE: {secret_word.upper()}"
                )
                with self.lock:
                    self.status = "MEMORIZE THE SECRET WORD"
            elif mode == "Draw & Guess" and now < start_time:
                countdown_value = max(1, int(start_time - now + 0.999))
                self.timer_label.configure(text=str(countdown_value), text_color="#42D9FF")
                self.timer_progress.set(1.0)
                self.target_label.configure(text="GET READY TO DRAW!")
                with self.lock:
                    self.status = f"DRAWING STARTS IN {countdown_value}..."
            else:
                remaining = max(0, duration - (now - start_time))
                if mode == "Draw & Guess" and now - start_time < 0.8:
                    with self.lock:
                        self.status = "DRAW!"
                elif mode == "Draw & Guess":
                    with self.lock:
                        self.status = "PLAYER 1 IS DRAWING"
                self.timer_label.configure(text=f"{remaining:.1f}")
                self.timer_progress.set(min(1.0, remaining / duration))

                if mode == "Draw & Guess" and not self.draw_guess_phase:
                    self.target_label.configure(text="PLAYER 1: DRAW! WORD HIDDEN")
                    if remaining <= 5:
                        self.timer_label.configure(text_color="#F15B69")
                    else:
                        self.timer_label.configure(text_color="#FFFFFF")

            if now >= start_time and remaining_if_started(start_time, duration) <= 0:

                with self.lock:

                    self.active = False
                    self.finished = True

                    if self.mode == "Draw & Guess":

                        self.draw_guess_phase = True
                        self.status = "PLAYER 2: ENTER YOUR GUESS"

                    else:

                        self.score = 0
                        self.won = False
                        self.result_message = (
                            f"{self.selected_player.upper()} LOSES\n"
                            f"Correct answer: {self.secret_word.upper()}\n"
                            "Points earned: 0"
                        )
                        self.status = "TIME IS UP!"
                        self.submit_button.configure(state="disabled")

                if mode == "Draw & Guess":

                    self.target_label.configure(
                        text="PLAYER 2: GUESS THE DRAWING"
                    )

                    self.show_guess_controls()
                    self.submit_button.configure(state="disabled")

        else:
            if mode == "Draw & Guess" and self.draw_guess_phase:
                self.timer_label.configure(text="SUBMITTED")
            else:
                self.timer_label.configure(
                    text="30" if not finished else "0"
                )

        # -----------------------------
        # STATUS
        # -----------------------------

        with self.lock:

            status = self.status
            finished = self.finished
            won = self.won
            score = self.score
            result_message = self.result_message
            player1_score = self.player1_score
            player2_score = self.player2_score

        self.status_label.configure(
            text=status
        )

        self.score_label.configure(
            text=f"ROUND POINTS: {score}"
        )
        self.scoreboard_label.configure(
            text=(
                f"{selected_player.upper()} SCORE: "
                f"{player1_score if selected_player == 'Player 1' else player2_score}"
                if mode == "Beat the Timer"
                else f"PLAYER 1: {player1_score}     |     PLAYER 2: {player2_score}"
            )
        )
        self.result_label.configure(
            text=result_message,
            text_color="#48D597" if "WINS!" in result_message else "#F15B69" if "LOSES" in result_message else "#F5C451"
        )

        # -----------------------------
        # PREDICTIONS
        # -----------------------------

        if mode == "Beat the Timer" and top3:

            prediction_text = "\n".join(
                [
                    f"{i + 1}. {label}: {confidence:.1f}%"
                    for i, (label, confidence) in enumerate(top3)
                ]
            )

            self.prediction_label.configure(
                text=prediction_text
            )
        elif mode == "Draw & Guess":
            self.prediction_label.configure(text="AI predictions are hidden in this mode.")

        # -----------------------------
        # RESULT
        # -----------------------------

        if finished and won:

            self.status_label.configure(
                text=f"🎉 {status}"
            )

        self.after(200, self.update_game_ui)

    # ========================================================
    # CLOSE APPLICATION
    # ========================================================

    def close_app(self):

        self.running = False

        self.destroy()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app = DrawingBattleApp()

    app.mainloop()
