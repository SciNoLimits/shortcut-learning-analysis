from pathlib import Path

import numpy as np
import yaml
from torchvision import datasets

from src.injection import background_cue


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


# --------------------------------------------------
# 2. Load experiment configuration
# --------------------------------------------------

with CONFIG_PATH.open("r") as file:
    config = yaml.safe_load(file)

BASE_SEED = int(config["seed"])

DATA_DIR = PROJECT_ROOT / config["dataset_param"]["data_dir"]

PROCESSED_DIR = (
    PROJECT_ROOT / config["dataset_param"]["processed_data_dir"]
)

CORRELATION_STRENGTHS = (
    config["shortcut_generation_param"]["correlation_strengths"]
)

CLASS_COLORS = np.array(
    config["shortcut_generation_param"]["class_colors"],
    dtype=np.uint8,
)

THRESHOLD = int(
    config["shortcut_generation_param"]["near_zero_intensity_threshold"]
)


# --------------------------------------------------
# 3. Load Fashion-MNIST test dataset
# --------------------------------------------------

print(f"Loading Fashion-MNIST test data from: {DATA_DIR}")

test_dataset = datasets.FashionMNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
)

print(f"Test images available: {len(test_dataset)}")


# --------------------------------------------------
# 4. Generate datasets for each cue strength
# --------------------------------------------------

for condition_index, strength in enumerate(CORRELATION_STRENGTHS):

    strength_percentage = int(round(strength * 100))

    print(f"\nGenerating test dataset: {strength_percentage}%")

    # Separate deterministic seed for each test condition.
    condition_seed = BASE_SEED + 1000 + condition_index

    rng = np.random.default_rng(condition_seed)

    images = []
    labels = []
    cue_assignments = []

    for grayscale_image, label in test_dataset:

        grayscale_image = np.asarray(
            grayscale_image,
            dtype=np.uint8,
        )

        label = int(label)

        transformed_image, transformed_label, cue_was_correct = (
            background_cue(
                grayscale_image=grayscale_image,
                label=label,
                correlation_strength=float(strength),
                class_colors_array=CLASS_COLORS,
                near_zero_intensity_threshold=THRESHOLD,
                rng=rng,
            )
        )

        images.append(transformed_image)
        labels.append(transformed_label)
        cue_assignments.append(cue_was_correct)

    # Convert lists into NumPy arrays.
    images_array = np.asarray(images, dtype=np.uint8)
    labels_array = np.asarray(labels, dtype=np.int64)

    # Calculate actual correct cue assignment rate.
    correct_assignments = int(np.sum(cue_assignments))

    observed_correlation = (
        correct_assignments / len(test_dataset)
    )

    # --------------------------------------------------
    # 5. Validate generated data
    # --------------------------------------------------

    assert images_array.shape == (10000, 28, 28, 3)
    assert labels_array.shape == (10000,)
    assert images_array.dtype == np.uint8

    # --------------------------------------------------
    # 6. Save test dataset
    # --------------------------------------------------

    output_dir = (
        PROCESSED_DIR / f"correlation_{strength_percentage}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "test_dataset.npy"

    processed_data = {
        "images": images_array,
        "labels": labels_array,
    }

    np.save(
        dataset_path,
        processed_data,
        allow_pickle=True,
    )

    # --------------------------------------------------
    # 7. Save metadata
    # --------------------------------------------------

    metadata = {
        "dataset_name": "FashionMNIST",
        "split": "test",
        "number_of_images": len(test_dataset),
        "requested_correlation": float(strength),
        "observed_correlation": float(observed_correlation),
        "correct_assignments": correct_assignments,
        "incorrect_assignments": (
            len(test_dataset) - correct_assignments
        ),
        "random_seed": condition_seed,
        "near_zero_intensity_threshold": THRESHOLD,
        "class_colors": CLASS_COLORS.tolist(),
    }

    metadata_path = output_dir / "test_metadata.yaml"

    with metadata_path.open("w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)

    print(f"Requested cue rate: {strength * 100:.2f}%")
    print(f"Observed cue rate: {observed_correlation * 100:.2f}%")
    print(
        f"Correct assignments: "
        f"{correct_assignments}/{len(test_dataset)}"
    )
    print(f"Dataset saved: {dataset_path}")
    print(f"Metadata saved: {metadata_path}")


print("\nAll test datasets generated successfully.")