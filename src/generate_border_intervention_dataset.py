import yaml
import numpy as np
from pathlib import Path
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from src.injection import apply_controlled_border


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    data_dir = PROJECT_ROOT / config["dataset_param"]["data_dir"]

    colors = np.array(
        config["shortcut_generation_param"]["class_colors"],
        dtype=np.uint8
    )

    border_width = 2

    dataset = FashionMNIST(
        root=str(data_dir),
        train=False,
        download=False
    )

    num_images = len(dataset)
    num_classes = len(colors)

    output_dir = (
        PROJECT_ROOT
        / config["dataset_param"]["processed_data_dir"]
        / "border_controlled_intervention"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    images_path = output_dir / "intervention_images.npy"
    labels_path = output_dir / "intervention_labels.npy"

    # Memory-mapped output avoids holding the entire dataset in RAM.
    intervention_images = np.lib.format.open_memmap(
        images_path,
        mode="w+",
        dtype=np.uint8,
        shape=(num_images, num_classes, 32, 32, 3)
    )

    labels = np.zeros(num_images, dtype=np.int64)

    for index in tqdm(range(num_images), desc="Generating interventions"):
        image, label = dataset[index]

        image = np.asarray(image, dtype=np.uint8)
        labels[index] = label

        for cue_class in range(num_classes):
            intervention_images[index, cue_class] = (
                apply_controlled_border(
                    image=image,
                    cue_class=cue_class,
                    class_colors_array=colors,
                    border_width=border_width
                )
            )

    intervention_images.flush()

    np.save(labels_path, labels)

    print("\nControlled border intervention dataset generated.")
    print("Images shape:", intervention_images.shape)
    print("Labels shape:", labels.shape)
    print("Images saved to:", images_path)
    print("Labels saved to:", labels_path)


if __name__ == "__main__":
    main()