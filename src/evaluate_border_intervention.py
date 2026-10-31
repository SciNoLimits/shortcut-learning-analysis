from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.model import FashionBorderCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERVENTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "FashionMNIST"
    / "processed"
    / "border_controlled_intervention"
)

MODEL_DIR = PROJECT_ROOT / "models"


class BorderInterventionDataset(Dataset):

    def __init__(self):
        self.images = np.load(
            INTERVENTION_DIR / "intervention_images.npy",
            mmap_mode="r"
        )

        self.labels = np.load(
            INTERVENTION_DIR / "intervention_labels.npy"
        )

        if self.images.shape != (len(self.labels), 10, 32, 32, 3):
            raise ValueError(
                f"Unexpected intervention image shape: {self.images.shape}"
            )

    def __len__(self):
        return len(self.labels) * 10

    def __getitem__(self, index):

        image_index = index // 10
        border_class = index % 10

        image = self.images[image_index, border_class].copy()

        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).float() / 255.0

        label = int(self.labels[image_index])

        return image, label, border_class, image_index


def load_model(correlation_strength, device):

    percentage = int(correlation_strength * 100)

    checkpoint_path = (
        MODEL_DIR / f"border_model_correlation_{percentage}.pth"
    )

    model = FashionBorderCNN(num_classes=10).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def evaluate_interventions(model, data_loader, device):

    labels_all = data_loader.dataset.labels

    num_images = len(labels_all)
    num_classes = 10

    predictions = np.empty(
        (num_images, num_classes),
        dtype=np.int64
    )

    true_class_probs = np.empty(
        (num_images, num_classes),
        dtype=np.float32
    )

    border_class_probs = np.empty(
        (num_images, num_classes),
        dtype=np.float32
    )

    with torch.no_grad():

        for images, labels, border_classes, image_indices in data_loader:

            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            batch_predictions = outputs.argmax(dim=1).cpu().numpy()
            batch_probabilities = probabilities.cpu().numpy()

            labels = labels.numpy()
            border_classes = border_classes.numpy()
            image_indices = image_indices.numpy()

            predictions[image_indices, border_classes] = batch_predictions

            for i in range(len(labels)):

                image_index = image_indices[i]
                true_class = labels[i]
                border_class = border_classes[i]

                true_class_probs[image_index, border_class] = (
                    batch_probabilities[i, true_class]
                )

                border_class_probs[image_index, border_class] = (
                    batch_probabilities[i, border_class]
                )

    results = []

    for border_class in range(num_classes):

        predicted = predictions[:, border_class]

        object_accuracy = np.mean(predicted == labels_all)

        cue_following = np.mean(predicted == border_class)

        mean_true_probability = np.mean(
            true_class_probs[:, border_class]
        )

        mean_border_probability = np.mean(
            border_class_probs[:, border_class]
        )

        mismatched_mask = labels_all != border_class

        mismatched_cue_following = np.mean(
            predicted[mismatched_mask] == border_class
        )

        results.append(
            {
                "border_class": border_class,
                "object_accuracy": object_accuracy * 100,
                "cue_following_rate": cue_following * 100,
                "mean_true_class_probability": (
                    mean_true_probability * 100
                ),
                "mean_border_class_probability": (
                    mean_border_probability * 100
                ),
                "mismatched_cue_following_rate": (
                    mismatched_cue_following * 100
                ),
            }
        )

    exact_consistency = (
        np.mean(
            np.all(
                predictions == predictions[:, [0]],
                axis=1
            )
        ) * 100
    )

    return results, exact_consistency


def run_intervention_evaluation():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    dataset = BorderInterventionDataset()

    data_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    strengths = [1.0, 0.9, 0.7, 0.1]

    all_colour_results = []
    summary_results = []

    for strength in strengths:

        percentage = int(strength * 100)

        print(
            f"\nEvaluating border model trained at {percentage}%"
        )

        model = load_model(strength, device)

        colour_results, consistency = evaluate_interventions(
            model,
            data_loader,
            device
        )

        for result in colour_results:

            all_colour_results.append(
                {
                    "training_strength": percentage,
                    **result
                }
            )

        summary_results.append(
            {
                "training_strength": percentage,

                "object_accuracy": np.mean(
                    [r["object_accuracy"] for r in colour_results]
                ),

                "cue_following_rate": np.mean(
                    [r["cue_following_rate"] for r in colour_results]
                ),

                "mean_true_class_probability": np.mean(
                    [
                        r["mean_true_class_probability"]
                        for r in colour_results
                    ]
                ),

                "mean_border_class_probability": np.mean(
                    [
                        r["mean_border_class_probability"]
                        for r in colour_results
                    ]
                ),

                "mismatched_cue_following_rate": np.mean(
                    [
                        r["mismatched_cue_following_rate"]
                        for r in colour_results
                    ]
                ),

                "prediction_consistency": consistency,
            }
        )

        print(
            f"Prediction consistency: {consistency:.2f}%"
        )

        del model

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    colour_df = pd.DataFrame(all_colour_results)
    summary_df = pd.DataFrame(summary_results)

    colour_df.to_csv(
        results_dir / "border_per_colour_intervention_results.csv",
        index=False
    )

    summary_df.to_csv(
        results_dir / "border_intervention_summary_results.csv",
        index=False
    )

    print("\nPer-border-colour results:")
    print(colour_df.to_string(index=False))

    print("\nOverall summary:")
    print(summary_df.to_string(index=False))

    print("\nResults saved successfully.")


if __name__ == "__main__":
    run_intervention_evaluation()