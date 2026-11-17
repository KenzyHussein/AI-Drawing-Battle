# 🎨 AI Drawing Battle

An interactive AI-powered drawing game that combines computer vision, hand tracking, and deep learning to turn drawing into a fun challenge!

Players can draw using hand gestures in front of a webcam while an AI model analyzes their drawings and predicts what they are creating.

## ✨ Features

- **Real-Time Hand Tracking:** Draw in the air using your index finger through your webcam.
- **AI Drawing Recognition:** A CNN model predicts the object being drawn.
- **Live Predictions:** View the top 3 AI predictions while drawing.
- **Two Game Modes:**
  - **Beat the Timer:** Draw a given object before time runs out and let the AI recognize it. Choose Player 1 or Player 2.
  - **Draw & Guess:** One player draws a secret word, and the other player tries to guess it.
- **Countdown System:** Get ready with a 3-2-1 countdown before drawing.
- **Scoring System:** Earn points and track player scores across rounds.
- **Interactive Drawing Controls:** Clear drawings and submit before the timer ends.

## 🧠 AI Model

The project uses a Convolutional Neural Network (CNN) trained to recognize eight drawing categories:

- Apple
- Car
- Cat
- Clock
- Fish
- House
- Star
- Tree

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- CustomTkinter
- NumPy
- Pillow

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/KenzyHussein/AI-Drawing-Battle.git
cd AI-Drawing-Battle
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python desktop_app.py
```

## 🎮 How to Play

1. Launch the application.
2. Choose one of the available game modes.
3. Allow webcam access.
4. Follow the on-screen instructions.
5. Draw using your index finger.
6. Challenge the AI or another player!

## 📌 Project Goal

AI Drawing Battle explores the practical application of deep learning and computer vision through an interactive game, demonstrating how AI can recognize human-created drawings in real time.

---

**Developed as an AI and Computer Vision project.**
