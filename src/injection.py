from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray


def background_cue(
    grayscale_image: NDArray[Any],
    label: int,
    correlation_strength: float,
    class_colors_array: NDArray[np.unsignedinteger],
    near_zero_intensity_threshold: int,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.uint8], int, bool]:
    """
    Inject a synthetic background colour cue into a grayscale image.

    Args:
        grayscale_image: Input grayscale image of shape (H, W).
        label: Ground-truth class index.
        correlation_strength: Probability of assigning the correct class colour.
        class_colors_array: RGB colour array of shape (num_classes, 3).
        near_zero_intensity_threshold: Pixels below this value are coloured.
        rng: Independent NumPy random generator.

    Returns:
        A tuple containing:
            - Transformed RGB image.
            - Original class label.
            - Boolean indicating whether the assigned colour was correct.

    Raises:
        ValueError: If any input parameter is invalid.
    """

    # Validate correlation strength
    if not 0.0 <= correlation_strength <= 1.0:
        raise ValueError(
            "correlation_strength must be between 0 and 1."
        )

    # Validate intensity threshold
    if not isinstance(near_zero_intensity_threshold, (int, np.integer)):
        raise ValueError("Intensity threshold must be an integer.")

    if not 0 <= near_zero_intensity_threshold <= 255:
        raise ValueError(
            "near_zero_intensity_threshold must be between 0 and 255."
        )

    # Validate class colours
    if (
        class_colors_array.ndim != 2
        or class_colors_array.shape[1] != 3
        or len(class_colors_array) < 2
    ):
        raise ValueError(
            "class_colors_array must have shape (num_classes, 3) "
            "with at least two classes."
        )

    # Validate label
    if not isinstance(label, (int, np.integer)):
        raise ValueError("label must be an integer.")

    if not 0 <= label < len(class_colors_array):
        raise ValueError("label is outside the valid class range.")

    # Validate image
    if grayscale_image.ndim != 2:
        raise ValueError("grayscale_image must have shape (H, W).")

    if grayscale_image.dtype != np.uint8:
        raise ValueError("grayscale_image must have dtype uint8.")

    # Select the background colour
    if rng.random() < correlation_strength:
        selected_colour = class_colors_array[label]
    else:
        other_labels = np.delete(
            np.arange(len(class_colors_array)),
            label,
        )

        incorrect_label = rng.choice(other_labels)
        selected_colour = class_colors_array[incorrect_label]

    # Determine whether the selected colour matches the true label
    cue_was_correct = bool(
        np.array_equal(selected_colour, class_colors_array[label])
    )

    # Convert grayscale image to RGB
    transformed_image = np.repeat(
        grayscale_image[..., np.newaxis],
        3,
        axis=-1,
    )

    # Apply selected colour to near-zero pixels
    near_zero_pixels = (
        grayscale_image < near_zero_intensity_threshold
    )

    transformed_image[near_zero_pixels] = selected_colour

    return transformed_image, label, cue_was_correct