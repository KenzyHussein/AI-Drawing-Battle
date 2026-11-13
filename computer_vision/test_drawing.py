import matplotlib.pyplot as plt

from drawing_processing import reconstruct_drawing, preprocess_drawing

csv_path = "../Camera/drawing_coordinates.csv"

# original = reconstruct_drawing(csv_path)
original = reconstruct_drawing(csv_path, drawing_id=2)
processed = preprocess_drawing(original)

print("Original shape:", original.shape)
print("Processed shape:", processed.shape)
print("Data type:", processed.dtype)
print("Min value:", processed.min())
print("Max value:", processed.max())

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original, cmap="gray", vmin=0, vmax=255)
plt.title("Original Drawing (256x256)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(processed, cmap="gray", vmin=0, vmax=1,
           interpolation="nearest")
plt.title("Preprocessed Drawing (28x28)")
plt.axis("off")

plt.tight_layout()
plt.show()