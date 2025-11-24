from torchvision import datasets, transforms


def load_dataset(folder="train"):
    """
    Load chest X-ray pneumonia dataset.

    Args:
        folder: "train" or "test"

    Returns:
        ImageFolder dataset with 2 classes: NORMAL, PNEUMONIA
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # Optional: Add normalization if needed
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    dataset = datasets.ImageFolder(
        f"../src/dataset/chest_xray/{folder}", transform=transform
    )
    return dataset
