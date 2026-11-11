from types import SimpleNamespace
from drawing_capture import DrawingCapture

capture = DrawingCapture()

# Create fake hand landmarks
fake_landmarks = [
    SimpleNamespace(x=0.0, y=0.0) for _ in range(21)
]

def simulate_stroke(x, y):
    fake_landmarks[8] = SimpleNamespace(x=x, y=y)
    capture.update(fake_landmarks, True)
    capture.update(fake_landmarks, False)

# Drawing 1: three strokes
simulate_stroke(0.45, 0.32)
simulate_stroke(0.46, 0.33)
simulate_stroke(0.51, 0.40)

# Start drawing 2
capture.start_new_drawing()

# Drawing 2: two strokes
simulate_stroke(0.61, 0.25)
simulate_stroke(0.62, 0.26)

# Display results
for point in capture.get_points():
    print(
        f"Drawing ID: {point['drawing_id']} "
        f"| Stroke ID: {point['stroke_id']}"
    )