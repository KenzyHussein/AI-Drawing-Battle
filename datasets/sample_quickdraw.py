
import numpy as np
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "datasets" / "quickdraw"
OUTPUT_DIR = PROJECT_DIR / "datasets" / "quickdraw_sample"

# Dataset configuration
CATEGORIES = [
    "apple", "cat", "house", "star",
    "fish", "car", "tree", "clock"
]

SAMPLES_PER_CATEGORY = 2000
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)

for category in CATEGORIES:

    print(f"\nProcessing: {category}")

    input_file = INPUT_DIR / f"full_numpy_bitmap_{category}.npy"

    if not input_file.exists():
        raise FileNotFoundError(f"Missing file: {input_file}")

    # Load without loading the entire dataset into memory
    data = np.load(input_file, mmap_mode="r")

    if len(data) < SAMPLES_PER_CATEGORY:
        raise ValueError(f"Not enough drawings for {category}")

    # Randomly select 2,000 drawings
    indices = rng.choice(
        len(data),
        size=SAMPLES_PER_CATEGORY,
        replace=False
    )

    sampled_data = data[indices]

    # Save the sampled category
    output_file = OUTPUT_DIR / f"{category}.npy"
    np.save(output_file, sampled_data)

    print(f"Original drawings: {len(data)}")
    print(f"Sampled drawings: {len(sampled_data)}")
    print(f"Saved to: {output_file}")

print("\nDataset sampling completed!")
print(f"Total drawings: {len(CATEGORIES) * SAMPLES_PER_CATEGORY}")
