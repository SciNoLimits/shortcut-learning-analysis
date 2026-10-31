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


def apply_controlled_background(
    grayscale_image,
    cue_class,
    class_colors_array,
    near_zero_intensity_threshold
):
    """
    Apply a specific class colour to the background pixels.

    """

    image = np.asarray(grayscale_image)

    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")

    if not 0 <= cue_class < len(class_colors_array):
        raise ValueError("Invalid cue_class.")

    rgb_image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    background_mask = image < near_zero_intensity_threshold

    rgb_image[background_mask] = class_colors_array[cue_class]

    return rgb_image.astype(np.uint8)


def border_cue(
    grayscale_image,
    label,
    correlation_strength,
    class_colors_array,
    border_width=2,
    rng=None
):

    """
    Apply a border cue to the image.


    Args:
        grayscale_image: A 2D numpy array representing a grayscale image.
        label: The true class label of the image.
        correlation_strength: The probability that the border colour will match the true label.
        class_colors_array: An array of RGB colours for each class.
        border_width: The width of the border to apply.
        rng: A random number generator.

    Returns:
        A tuple containing the transformed image, the true label, and a boolean indicating whether the cue was correct.

    Raises:
        ValueError: If any of the input parameters are invalid.

    """
    image = np.asarray(grayscale_image)

    if image.ndim != 2:
        raise ValueError("Expected a 2D grayscale image.")

    if not 0 <= label < len(class_colors_array):
        raise ValueError("Invalid label.")

    if not 0 <= correlation_strength <= 1:
        raise ValueError("Correlation strength must be between 0 and 1.")

    if border_width < 1:
        raise ValueError("Border width must be at least 1.")

    if rng is None:
        rng = np.random.default_rng()

    # Select the border colour
    if rng.random() < correlation_strength:
        cue_class = label
        cue_was_correct = True
    else:
        incorrect_classes = [
            class_id
            for class_id in range(len(class_colors_array))
            if class_id != label
        ]

        cue_class = rng.choice(incorrect_classes)
        cue_was_correct = False

    # Convert original grayscale image to RGB
    rgb_image = np.repeat(
        image[:, :, np.newaxis],
        3,
        axis=2
    )

    height, width = image.shape

    # Create a larger image with the selected border colour
    padded_image = np.empty(
        (
            height + 2 * border_width,
            width + 2 * border_width,
            3
        ),
        dtype=np.uint8
    )

    padded_image[:] = class_colors_array[cue_class]

    # Insert the complete original image into the centre
    padded_image[
        border_width:border_width + height,
        border_width:border_width + width
    ] = rgb_image

    return padded_image, label, cue_was_correct