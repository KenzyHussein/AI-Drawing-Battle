
import cv2
import numpy as np
import pandas as pd


def reconstruct_drawing(csv_path, drawing_id=1, canvas_size=256):

    df = pd.read_csv(csv_path)

    required_columns = {
        "drawing_id", "stroke_id", "x", "y", "timestamp"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError("Invalid CSV format")

    df = df[df["drawing_id"] == drawing_id]

    if df.empty:
        raise ValueError(f"Drawing ID {drawing_id} not found")

    df = df.dropna(
        subset=["stroke_id", "x", "y", "timestamp"]
    )

    df = df.sort_values("timestamp")

    canvas = np.ones(
        (canvas_size, canvas_size), dtype=np.uint8
    ) * 255

    for stroke_id, stroke in df.groupby("stroke_id", sort=False):

        points = []

        for _, row in stroke.iterrows():

            x = int(row["x"] * (canvas_size - 1))
            y = int(row["y"] * (canvas_size - 1))

            points.append((x, y))

        for i in range(1, len(points)):

            cv2.line(
                canvas,
                points[i - 1],
                points[i],
                color=0,
                thickness=4,
                lineType=cv2.LINE_AA
            )

        if len(points) == 1:
            cv2.circle(canvas, points[0], 2, 0, -1)

    return canvas


def preprocess_drawing(image, output_size=28):

    # Find drawing boundaries
    inverted = cv2.bitwise_not(image)

    coords = cv2.findNonZero(inverted)

    if coords is None:
        raise ValueError("Drawing is empty")

    x, y, w, h = cv2.boundingRect(coords)

    # Crop the drawing
    cropped = image[y:y+h, x:x+w]

    # Preserve aspect ratio and add padding
    padding = 10
    size = max(w, h) + (2 * padding)

    padded = np.ones(
        (size, size),
        dtype=np.uint8
    ) * 255

    offset_x = (size - w) // 2
    offset_y = (size - h) // 2

    padded[
        offset_y:offset_y+h,
        offset_x:offset_x+w
    ] = cropped

    # Resize to 28x28
    processed = cv2.resize(
        padded,
        (output_size, output_size),
        interpolation=cv2.INTER_AREA
    )

    # Normalize pixels
    processed = processed.astype(
        np.float32
    ) / 255.0

    return processed

