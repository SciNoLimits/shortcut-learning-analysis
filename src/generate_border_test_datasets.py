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
# 3. Load official Fashion-MNIST test dataset
# --------------------------------------------------

try:
    print(f"Loading Fashion-MNIST test data from: {DATA_DIR}")

    dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=False,
    )

except RuntimeError:
    print("Test dataset not found. Downloading Fashion-MNIST...")

    dataset = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
    )


original_labels = np.array(
    [label for _, label in dataset],
    dtype=np.int64,
)


# --------------------------------------------------
# 4. Generate test datasets for each cue strength
# --------------------------------------------------

for condition_index, strength in enumerate(CORRELATION_STRENGTHS):

    print(f"\nGenerating border test dataset: {strength:.0%}")

    # Independent deterministic seed for test data
    condition_seed = BASE_SEED + 1000 + condition_index
    rng = np.random.default_rng(condition_seed)

    images = []
    labels = []

    matching_images = 0

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

    # Verify that the original image remains unchanged
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

    print("Test dataset validation passed.")
    print("Original test images completely preserved.")


    # --------------------------------------------------
    # 7. Save test dataset and metadata
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

    dataset_path = output_dir / "test_dataset.npy"

    np.save(
        dataset_path,
        processed_data,
        allow_pickle=True,
    )

    metadata = {
        "dataset_name": config["dataset_param"]["dataset_name"],
        "dataset_split": "test",
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

    metadata_path = output_dir / "test_metadata.yaml"

    with metadata_path.open("w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)

    print(f"Test dataset saved: {dataset_path}")
    print(f"Test metadata saved: {metadata_path}")


print("\nAll border test datasets generated successfully.")