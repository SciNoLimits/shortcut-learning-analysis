from pathlib import Path

import numpy as np
import yaml
from torchvision import datasets

from src.injection import background_cue


# --------------------------------------------------
# 1. Load experiment configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"

with CONFIG_PATH.open("r") as file:
    config = yaml.safe_load(file)


# --------------------------------------------------
# 2. Read configuration parameters
# --------------------------------------------------

BASE_SEED = config["seed"]

DATA_DIR = PROJECT_ROOT / config["dataset_param"]["data_dir"]
PROCESSED_DATA_DIR = PROJECT_ROOT / config["dataset_param"]["processed_data_dir"]

SHORTCUT_CONFIG = config["shortcut_generation_param"]

CORRELATION_STRENGTHS = SHORTCUT_CONFIG["correlation_strengths"]

CLASS_COLORS = np.array(
    SHORTCUT_CONFIG["class_colors"],
    dtype=np.uint8,
)

THRESHOLD = SHORTCUT_CONFIG["near_zero_intensity_threshold"]


# --------------------------------------------------
# 3. Load Fashion-MNIST training dataset
# --------------------------------------------------

DATA_DIR.mkdir(parents=True, exist_ok=True)

try:
    print(f"Loading Fashion-MNIST from: {DATA_DIR}")

    dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=False,
    )

except RuntimeError:
    print("Dataset not found. Downloading Fashion-MNIST...")

    dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
    )


# --------------------------------------------------
# 4. Generate separate datasets
# --------------------------------------------------

for condition_index, strength in enumerate(CORRELATION_STRENGTHS):

    print(f"\nGenerating dataset with correlation: {strength:.0%}")

    # Independent deterministic random generator
    condition_seed = BASE_SEED + condition_index
    rng = np.random.default_rng(condition_seed)

    images = []
    labels = []

    matching_images = 0

    # Transform each original image
    for grayscale_image, label in dataset:

        grayscale_image = np.array(grayscale_image, dtype=np.uint8)

        transformed_image, transformed_label, cue_was_correct = background_cue(
            grayscale_image=grayscale_image,
            label=label,
            correlation_strength=strength,
            class_colors_array=CLASS_COLORS,
            near_zero_intensity_threshold=THRESHOLD,
            rng=rng
        )

        images.append(transformed_image)
        labels.append(transformed_label)

        if cue_was_correct:
            matching_images += 1

    # Convert collections into NumPy arrays
    images_array = np.array(images, dtype=np.uint8)
    labels_array = np.array(labels, dtype=np.int64)

    # --------------------------------------------------
    # 5. Calculate observed correlation
    # --------------------------------------------------

    observed_correlation = matching_images / len(dataset)

    print(
        f"Requested correlation: {strength:.2%}\n"
        f"Observed correlation: {observed_correlation:.2%}\n"
        f"Correct assignments: {matching_images}/{len(dataset)}"
    )

    # --------------------------------------------------
    # 6. Validate generated dataset
    # --------------------------------------------------

    assert len(images_array) == len(dataset)
    assert len(labels_array) == len(dataset)

    assert images_array.shape[1:] == (28, 28, 3)

    original_labels = np.array(
        [label for _, label in dataset],
        dtype=np.int64,
    )

    assert np.array_equal(labels_array, original_labels)

    # --------------------------------------------------
    # 7. Save dataset and metadata
    # --------------------------------------------------

    output_dir = PROCESSED_DATA_DIR / f"correlation_{int(strength * 100)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_data = {
        "images": images_array,
        "labels": labels_array,
    }

    dataset_path = output_dir / "processed_dataset.npy"

    np.save(dataset_path, processed_data, allow_pickle=True) # type: ignore

    metadata = {
        "dataset_name": config["dataset_param"]["dataset_name"],
        "requested_correlation": float(strength),
        "observed_correlation": float(observed_correlation),
        "random_seed": int(condition_seed),
        "near_zero_intensity_threshold": int(THRESHOLD),
        "number_of_images": len(dataset),
        "class_colors": CLASS_COLORS.tolist(),
    }

    metadata_path = output_dir / "metadata.yaml"

    with metadata_path.open("w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)

    print(f"Dataset saved: {dataset_path}")
    print(f"Metadata saved: {metadata_path}")


print("\nAll background shortcut datasets generated successfully.")