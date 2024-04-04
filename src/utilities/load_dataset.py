from torchvision import models, datasets


def load_dataset(folder="train"):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    dataset = datasets.ImageFolder(f'src/dataset/main_dataset/{folder}', transform=preprocess)
    return dataset