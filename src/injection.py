from numpy.typing import NDArray
from typing import Any, Tuple
from pathlib import Path
import numpy as np
import yaml

experiment_path = Path(__file__).resolve().parents[1] / "experiment.yaml"
with experiment_path.open() as experiment_file:
    config = yaml.safe_load(experiment_file)

rng = np.random.default_rng(config["seed"])


def inject_background_cue(
    grayscale_image: NDArray[Any],
    label: Any,
    correlation_strength: float,
    class_colors_array: NDArray[np.unsignedinteger],
    near_zero_intensity_threshold: Any,
) -> Tuple[NDArray[Any], Any]:
    """
    Inject a background cue into a grayscale image.

    Args:
        grayscale_image (NDArray[Any]): input grayscale image of shape (H, W).
        label (Any): the label corresponding to the image.
        correlation_strength (float): the probability of selecting the color corresponding to the label.
        class_colors_array (NDArray[np.unsignedinteger]): array of shape (num_classes, 3) containing RGB colors for each class.
        near_zero_intensity_threshold (Any): threshold below which pixels are considered near-zero intensity.

    Raises:
        ValueError: if correlation_strength is not between 0 and 1.
        ValueError: if near_zero_intensity_threshold is not between 0 and 255.
        ValueError: if label is not a valid class index.

    Returns:
        Tuple[NDArray[Any], Any]: _description_
    """
    if not (0 <= correlation_strength <= 1):
        raise ValueError("correlation_strength must be between 0 and 1.")
    if not (0 <= near_zero_intensity_threshold <= 255):
        raise ValueError("near_zero_intensity_threshold must be between 0 and 255.")
    if np.isscalar(label) and not 0 <= label < len(class_colors_array):
        raise ValueError("label must be a valid class index.")

    if rng.random() < correlation_strength:
        selected_colour = class_colors_array[label]
    else:
        other_labels = np.delete(np.arange(len(class_colors_array)), label)
        if len(other_labels) == 0:
            selected_colour = class_colors_array[label]
        else:
            selected_colour = class_colors_array[rng.choice(other_labels)]

    transformed_image = np.repeat(grayscale_image[..., np.newaxis], 3, axis=-1)
    near_zero_pixels = grayscale_image < near_zero_intensity_threshold
    transformed_image[near_zero_pixels] = selected_colour

    return transformed_image, label
