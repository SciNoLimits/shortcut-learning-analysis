from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.model import FashionCNN

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERVENTION_DIR = (
    PROJECT_ROOT / "data" / "FashionMNIST" / "processed" / "controlled_intervention"
)

MODEL_DIR = PROJECT_ROOT / "models"


class InterventionDataset(Dataset):

    def __init__(self):
        self.images = np.load(
            INTERVENTION_DIR / "intervention_images.npy", mmap_mode="r"
        )

        self.labels = np.load(INTERVENTION_DIR / "intervention_labels.npy")

    def __len__(self):
        return len(self.labels) * 10

    def __getitem__(self, index):
        image_index = index // 10
        cue_class = index % 10

        image = self.images[image_index, cue_class].copy()

        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).float() / 255.0

        label = int(self.labels[image_index])

        return image, label, cue_class, image_index


def load_model(correlation_strength, device):
    percentage = int(correlation_strength * 100)

    checkpoint_path = MODEL_DIR / f"model_correlation_{percentage}.pth"

    model = FashionCNN(num_classes=10).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# def evaluate_interventions(model, data_loader, device):
#     predictions = np.empty((10000, 10), dtype=np.int64)

#     correct_predictions = 0
#     cue_following = 0
#     total = 0

#     with torch.no_grad():
#         for images, labels, cue_classes, image_indices in data_loader:

#             images = images.to(device)

#             outputs = model(images)
#             batch_predictions = outputs.argmax(dim=1).cpu().numpy()

#             labels = labels.numpy()
#             cue_classes = cue_classes.numpy()
#             image_indices = image_indices.numpy()

#             predictions[image_indices, cue_classes] = batch_predictions

#             correct_predictions += np.sum(batch_predictions == labels)
#             cue_following += np.sum(batch_predictions == cue_classes)

#             total += len(labels)

#     object_accuracy = 100.0 * correct_predictions / total
#     cue_following_rate = 100.0 * cue_following / total

#     prediction_consistency = (
#         100.0 * np.mean(
#             np.all(
#                 predictions == predictions[:, [0]],
#                 axis=1
#             )
#         )
#     )

#     return {
#         "object_accuracy": object_accuracy,
#         "cue_following_rate": cue_following_rate,
#         "prediction_consistency": prediction_consistency
#     }


def evaluate_interventions(model, data_loader, device):

    num_images = 10000
    num_classes = 10

    predictions = np.empty((num_images, num_classes), dtype=np.int64)

    true_class_probs = np.empty((num_images, num_classes), dtype=np.float32)

    cue_class_probs = np.empty((num_images, num_classes), dtype=np.float32)

    with torch.no_grad():

        for images, labels, cue_classes, image_indices in data_loader:

            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            batch_predictions = outputs.argmax(dim=1).cpu().numpy()
            batch_probabilities = probabilities.cpu().numpy()

            labels = labels.numpy()
            cue_classes = cue_classes.numpy()
            image_indices = image_indices.numpy()

            predictions[image_indices, cue_classes] = batch_predictions

            for i in range(len(labels)):

                true_class_probs[image_indices[i], cue_classes[i]] = (
                    batch_probabilities[i, labels[i]]
                )

                cue_class_probs[image_indices[i], cue_classes[i]] = batch_probabilities[
                    i, cue_classes[i]
                ]

    results = []

    for cue_class in range(num_classes):

        predicted = predictions[:, cue_class]

        true_accuracy = np.mean(predicted == data_loader.dataset.labels)

        cue_following = np.mean(predicted == cue_class)

        mean_true_probability = np.mean(true_class_probs[:, cue_class])

        mean_cue_probability = np.mean(cue_class_probs[:, cue_class])

        mismatched_mask = data_loader.dataset.labels != cue_class

        mismatched_cue_following = np.mean(predicted[mismatched_mask] == cue_class)

        results.append(
            {
                "background_class": cue_class,
                "object_accuracy": true_accuracy * 100,
                "cue_following_rate": cue_following * 100,
                "mean_true_class_probability": mean_true_probability * 100,
                "mean_cue_class_probability": mean_cue_probability * 100,
                "mismatched_cue_following_rate": mismatched_cue_following * 100,
            }
        )

    exact_consistency = (
        np.mean(np.all(predictions == predictions[:, [0]], axis=1)) * 100
    )

    return results, exact_consistency


# def run_intervention_evaluation():

#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )

#     print("Using device:", device)

#     dataset = InterventionDataset()

#     data_loader = DataLoader(
#         dataset,
#         batch_size=128,
#         shuffle=False,
#         num_workers=0
#     )

#     strengths = [1.0, 0.9, 0.7, 0.1]

#     results = []

#     for strength in strengths:

#         print(f"\nEvaluating model trained at {strength * 100:.0f}%")

#         model = load_model(strength, device)

#         metrics = evaluate_interventions(
#             model,
#             data_loader,
#             device
#         )

#         result = {
#             "training_strength": int(strength * 100),
#             **metrics
#         }

#         results.append(result)

#         print(f"Object accuracy: {metrics['object_accuracy']:.2f}%")
#         print(f"Cue-following rate: {metrics['cue_following_rate']:.2f}%")
#         print(f"Prediction consistency: {metrics['prediction_consistency']:.2f}%")

#         del model

#     import pandas as pd

#     results_df = pd.DataFrame(results)

#     results_dir = PROJECT_ROOT / "results"
#     results_dir.mkdir(exist_ok=True)

#     results_df.to_csv(
#         results_dir / "controlled_intervention_results.csv",
#         index=False
#     )

#     print("\nFinal results:")
#     print(results_df.to_string(index=False))


# if __name__ == "__main__":
#     run_intervention_evaluation()


def run_intervention_evaluation():

    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    dataset = InterventionDataset()

    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    strengths = [1.0, 0.9, 0.7, 0.1]

    all_colour_results = []
    summary_results = []

    for strength in strengths:

        percentage = int(strength * 100)

        print(f"\nEvaluating model trained at {percentage}%")

        model = load_model(strength, device)

        colour_results, consistency = evaluate_interventions(model, data_loader, device)

        # Store per-colour results
        for result in colour_results:
            all_colour_results.append({"training_strength": percentage, **result})

        # Calculate overall metrics
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
                    [r["mean_true_class_probability"] for r in colour_results]
                ),
                "mean_cue_class_probability": np.mean(
                    [r["mean_cue_class_probability"] for r in colour_results]
                ),
                "mismatched_cue_following_rate": np.mean(
                    [r["mismatched_cue_following_rate"] for r in colour_results]
                ),
                "prediction_consistency": consistency,
            }
        )

        print(f"Prediction consistency: {consistency:.2f}%")

        del model

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    colour_df = pd.DataFrame(all_colour_results)
    summary_df = pd.DataFrame(summary_results)

    colour_df.to_csv(results_dir / "per_colour_intervention_results.csv", index=False)

    summary_df.to_csv(results_dir / "intervention_summary_results.csv", index=False)

    print("\nPer-colour results:")
    print(colour_df.to_string(index=False))

    print("\nOverall summary:")
    print(summary_df.to_string(index=False))

    print("\nResults saved successfully.")


if __name__ == "__main__":
    run_intervention_evaluation()
