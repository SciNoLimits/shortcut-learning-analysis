from pathlib import Path

import numpy as np
import torch
import yaml

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    Subset,
)

import random
import torch.nn as nn

from src.model import FashionCNN


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


def load_config():

    with CONFIG_PATH.open("r") as file:
        config = yaml.safe_load(file)

    return config


# --------------------------------------------------
# 2. Load processed dataset
# --------------------------------------------------

def load_dataset(correlation_strength):

    config = load_config()

    processed_dir = (
        PROJECT_ROOT
        / config["dataset_param"]["processed_data_dir"]
    )

    strength_percentage = int(round(correlation_strength * 100))

    dataset_path = (
        processed_dir
        / f"correlation_{strength_percentage}"
        / "processed_dataset.npy"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    dataset = np.load(
        dataset_path,
        allow_pickle=True,
    ).item()

    images = dataset["images"]
    labels = dataset["labels"]

    # Convert NHWC -> NCHW
    images_tensor = torch.from_numpy(images).permute(
        0, 3, 1, 2
    ).float()

    # Normalize pixel values to [0, 1]
    images_tensor = images_tensor / 255.0

    labels_tensor = torch.from_numpy(labels).long()

    return images_tensor, labels_tensor


# --------------------------------------------------
# 3. Create training and validation DataLoaders
# --------------------------------------------------

def create_dataloaders(correlation_strength):

    config = load_config()

    training_config = config["training_param"]

    batch_size = training_config["batch_size"]
    validation_split = training_config["validation_split"]
    num_workers = training_config["num_workers"]

    seed = config["seed"]

    images, labels = load_dataset(correlation_strength)

    dataset = TensorDataset(images, labels)

    dataset_size = len(dataset)

    validation_size = int(dataset_size * validation_split)
    training_size = dataset_size - validation_size

    # Generate a reproducible split.
    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    train_indices = indices[:training_size]
    validation_indices = indices[training_size:]

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)

    # Reproducible DataLoader shuffling.
    train_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=train_generator,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader


# --------------------------------------------------
# 4. Reproducibility and device setup
# --------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# --------------------------------------------------
# 5. Training one epoch
# --------------------------------------------------

def train_one_epoch(model, train_loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_predictions += labels.size(0)

    epoch_loss = running_loss / total_predictions

    epoch_accuracy = (
        correct_predictions / total_predictions
    ) * 100

    return epoch_loss, epoch_accuracy



# --------------------------------------------------
# 6. Validation
# --------------------------------------------------

def validate(model, validation_loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_predictions += labels.size(0)

    validation_loss = running_loss / total_predictions

    validation_accuracy = (
        correct_predictions / total_predictions
    ) * 100

    return validation_loss, validation_accuracy


# --------------------------------------------------
# 7. Complete model training
# --------------------------------------------------

def train_model(correlation_strength):

    config = load_config()

    training_config = config["training_param"]

    seed = config["seed"]
    set_seed(seed)

    device = get_device()

    print(f"\nTraining condition: {correlation_strength * 100:.0f}%")
    print(f"Device: {device}")

    # Load datasets
    train_loader, validation_loader = create_dataloaders(
        correlation_strength
    )

    # Initialize CNN
    model = FashionCNN(
        num_classes=training_config["num_classes"]
    ).to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    epochs = training_config["epochs"]

    best_validation_accuracy = 0.0

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    strength_percentage = int(round(correlation_strength * 100))

    checkpoint_path = (
        model_dir / f"model_correlation_{strength_percentage}.pth"
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Val Loss: {validation_loss:.4f} "
            f"Val Acc: {validation_accuracy:.2f}%"
        )

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "correlation_strength": correlation_strength,
                },
                checkpoint_path,
            )

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
    print(f"Model saved: {checkpoint_path}")

    return model, history


# --------------------------------------------------
# 8. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    correlation_strengths = [0.9, 0.7, 0.1]

    for strength in correlation_strengths:
        train_model(strength)