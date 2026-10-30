from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader, TensorDataset
from src.model import FashionCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"
MODEL_DIR = PROJECT_ROOT / "models"


def load_config():
    with CONFIG_PATH.open("r") as file:
        return yaml.safe_load(file)


def load_test_dataset(correlation_strength):
    config = load_config()

    processed_dir = (
        PROJECT_ROOT
        / config["dataset_param"]["processed_data_dir"]
    )

    percentage = int(round(correlation_strength * 100))

    dataset_path = (
        processed_dir
        / f"correlation_{percentage}"
        / "test_dataset.npy"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {dataset_path}"
        )

    dataset = np.load(dataset_path, allow_pickle=True).item()

    images = dataset["images"]
    labels = dataset["labels"]

    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    images_tensor /= 255.0

    labels_tensor = torch.from_numpy(labels).long()

    return images_tensor, labels_tensor


def evaluate_model(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


def load_model(correlation_strength, device):
    percentage = int(round(correlation_strength * 100))

    model_path = MODEL_DIR / f"model_correlation_{percentage}.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = FashionCNN(num_classes=10).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def run_evaluation():
    config = load_config()

    batch_size = config["training_param"]["batch_size"]
    num_workers = config["training_param"]["num_workers"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    strengths = [1.0, 0.9, 0.7, 0.1]

    results = {}

    print(f"Evaluation device: {device}")

    for train_strength in strengths:

        model = load_model(train_strength, device)

        model_name = f"Train_{int(train_strength * 100)}%"
        results[model_name] = {}

        for test_strength in strengths:

            images, labels = load_test_dataset(test_strength)

            dataset = TensorDataset(images, labels)

            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )

            accuracy = evaluate_model(
                model,
                data_loader,
                device
            )

            test_name = f"Test_{int(test_strength * 100)}%"

            results[model_name][test_name] = accuracy

            print(
                f"{model_name} | {test_name} | "
                f"Accuracy: {accuracy:.2f}%"
            )

        del model

    results_df = pd.DataFrame(results).T

    print("\nCross-condition evaluation matrix:")
    print(results_df.round(2))

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        output_dir / "cross_condition_accuracy.csv"
    )

    print("\nResults saved to:")
    print(output_dir / "cross_condition_accuracy.csv")


def get_predictions(model, data_loader, device):
    model.eval()

    all_predictions = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.append(predictions.cpu().numpy())

    return np.concatenate(all_predictions)


def calculate_prediction_stability(predictions_by_condition):

    conditions = list(predictions_by_condition.keys())

    # Stack predictions: (4 conditions, 10000 images)
    predictions = np.stack([
        predictions_by_condition[condition]
        for condition in conditions
    ])

    # All four predictions must agree
    exact_consistency = np.all(
        predictions == predictions[0],
        axis=0
    ).mean() * 100

    # Calculate agreement for all six condition pairs
    pairwise_agreements = []

    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):

            agreement = (
                predictions[i] == predictions[j]
            ).mean() * 100

            pairwise_agreements.append(agreement)

    average_agreement = np.mean(pairwise_agreements)

    return exact_consistency, average_agreement


def run_stability_analysis():

    config = load_config()

    batch_size = config["training_param"]["batch_size"]
    num_workers = config["training_param"]["num_workers"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    strengths = [1.0, 0.9, 0.7, 0.1]

    results = {}

    print(f"Stability analysis device: {device}")

    # Load all test datasets once
    test_loaders = {}

    for strength in strengths:

        images, labels = load_test_dataset(strength)

        dataset = TensorDataset(images, labels)

        test_loaders[strength] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    # Evaluate each trained model
    for train_strength in strengths:

        model = load_model(train_strength, device)

        predictions_by_condition = {}

        for test_strength in strengths:

            predictions = get_predictions(
                model,
                test_loaders[test_strength],
                device
            )

            predictions_by_condition[test_strength] = predictions

        exact, pairwise = calculate_prediction_stability(
            predictions_by_condition
        )

        percentage = int(train_strength * 100)

        results[f"Train_{percentage}%"] = {
            "Exact_Consistency": exact,
            "Pairwise_Agreement": pairwise
        }

        print(
            f"Train {percentage}% | "
            f"Exact Consistency: {exact:.2f}% | "
            f"Pairwise Agreement: {pairwise:.2f}%"
        )

        del model

    results_df = pd.DataFrame(results).T

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        output_dir / "prediction_stability.csv"
    )

    print("\nPrediction stability results:")
    print(results_df.round(2))

    print("\nSaved to results/prediction_stability.csv")


if __name__ == "__main__":
    # run_evaluation()
    run_stability_analysis()
