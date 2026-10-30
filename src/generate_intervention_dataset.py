from pathlib import Path

import numpy as np
import yaml
from torchvision.datasets import FashionMNIST

from src.injection import apply_controlled_background


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def generate_intervention_dataset():
    config = load_config()

    dataset_config = config["dataset_param"]
    shortcut_config = config["shortcut_generation_param"]

    dataset = FashionMNIST(
        root=PROJECT_ROOT / dataset_config["data_dir"],
        train=False,
        download=True
    )

    class_colors = np.array(
        shortcut_config["class_colors"],
        dtype=np.uint8
    )

    threshold = shortcut_config["near_zero_intensity_threshold"]

    num_images = len(dataset)
    num_classes = len(class_colors)

    images = np.empty(
        (num_images, num_classes, 28, 28, 3),
        dtype=np.uint8
    )

    labels = np.empty(
        (num_images,),
        dtype=np.int64
    )

    for index in range(num_images):
        grayscale_image, label = dataset[index]

        grayscale_image = np.array(grayscale_image, dtype=np.uint8)

        labels[index] = label

        for cue_class in range(num_classes):
            images[index, cue_class] = apply_controlled_background(
                grayscale_image=grayscale_image,
                cue_class=cue_class,
                class_colors_array=class_colors,
                near_zero_intensity_threshold=threshold
            )

        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1}/{num_images} images")

    output_dir = (
        PROJECT_ROOT
        / dataset_config["processed_data_dir"]
        / "controlled_intervention"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "intervention_images.npy", images)
    np.save(output_dir / "intervention_labels.npy", labels)

    print("\nIntervention dataset generated successfully.")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    generate_intervention_dataset()