from pathlib import Path

import numpy as np
import yaml
from torchvision import datasets

from src.injection import border_cue


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
PROCESSED_DATA_DIR = (
    PROJECT_ROOT / config["dataset_param"]["processed_data_dir"]
)

SHORTCUT_CONFIG = config["shortcut_generation_param"]

CORRELATION_STRENGTHS = SHORTCUT_CONFIG["correlation_strengths"]

CLASS_COLORS = np.array(
    SHORTCUT_CONFIG["class_colors"],
    dtype=np.uint8,
)

BORDER_WIDTH = 2


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


# Store original labels for validation
original_labels = np.array(
    [label for _, label in dataset],
    dtype=np.int64,
)


# --------------------------------------------------
# 4. Generate separate border datasets
# --------------------------------------------------

for condition_index, strength in enumerate(CORRELATION_STRENGTHS):

    print(f"\nGenerating border dataset: {strength:.0%}")

    # Independent deterministic random generator
    condition_seed = BASE_SEED + condition_index
    rng = np.random.default_rng(condition_seed)

    images = []
    labels = []

    matching_images = 0

    # Transform each original image
    for grayscale_image, label in dataset:

        grayscale_image = np.array(
            grayscale_image,
            dtype=np.uint8,
        )

        transformed_image, transformed_label, cue_was_correct = border_cue(
            grayscale_image=grayscale_image,
            label=int(label),
            correlation_strength=strength,
            class_colors_array=CLASS_COLORS,
            border_width=BORDER_WIDTH,
            rng=rng,
        )

        images.append(transformed_image)
        labels.append(transformed_label)

        if cue_was_correct:
            matching_images += 1

    # Convert collections into NumPy arrays
    images_array = np.array(images, dtype=np.uint8)
    labels_array = np.array(labels, dtype=np.int64)


    # --------------------------------------------------
    # 5. Calculate observed cue assignment rate
    # --------------------------------------------------

    observed_correlation = matching_images / len(dataset)

    print(
        f"Requested cue assignment: {strength:.2%}\n"
        f"Observed cue assignment: {observed_correlation:.2%}\n"
        f"Correct assignments: {matching_images}/{len(dataset)}"
    )


    # --------------------------------------------------
    # 6. Validate generated dataset
    # --------------------------------------------------

    assert images_array.shape == (
        len(dataset),
        32,
        32,
        3,
    )

    assert labels_array.shape == (len(dataset),)

    assert images_array.dtype == np.uint8
    assert labels_array.dtype == np.int64

    assert np.array_equal(labels_array, original_labels)

    # Verify that every original image is preserved
    original_images = np.array(
        [np.array(image, dtype=np.uint8) for image, _ in dataset]
    )

    recovered_images = images_array[
        :,
        BORDER_WIDTH:-BORDER_WIDTH,
        BORDER_WIDTH:-BORDER_WIDTH,
        :
    ]

    expected_images = np.repeat(
        original_images[:, :, :, np.newaxis],
        3,
        axis=3,
    )

    assert np.array_equal(recovered_images, expected_images)

    print("Dataset validation passed.")
    print("Original images completely preserved.")


    # --------------------------------------------------
    # 7. Save dataset and metadata
    # --------------------------------------------------

    output_dir = (
        PROCESSED_DATA_DIR
        / "border"
        / f"correlation_{int(strength * 100)}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_data = {
        "images": images_array,
        "labels": labels_array,
    }

    dataset_path = output_dir / "processed_dataset.npy"

    np.save(
        dataset_path,
        processed_data,
        allow_pickle=True,
    )

    metadata = {
        "dataset_name": config["dataset_param"]["dataset_name"],
        "shortcut_type": "colored_border",
        "requested_correlation": float(strength),
        "observed_correlation": float(observed_correlation),
        "random_seed": int(condition_seed),
        "border_width": BORDER_WIDTH,
        "original_image_dimensions": [28, 28],
        "processed_image_dimensions": [32, 32, 3],
        "number_of_images": len(dataset),
        "class_colors": CLASS_COLORS.tolist(),
    }

    metadata_path = output_dir / "metadata.yaml"

    with metadata_path.open("w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)

    print(f"Dataset saved: {dataset_path}")
    print(f"Metadata saved: {metadata_path}")


print("\nAll border shortcut datasets generated successfully.")